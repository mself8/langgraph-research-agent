"""LangGraph 연구 에이전트 — 계획→실행→평가→재계획 루프.

/research(클로드 코드 활용판)의 "직접 구현" 버전. LLM 두뇌는 기본이
Claude Code CLI(`claude -p`, 구독으로 처리)이고 로컬 vLLM으로 교체 가능 —
어느 쪽도 API 키가 필요 없다.

사용법:
    python agent.py "sandbox/ 안의 csv 행 수를 세라"   # 기본: 클로드 CLI 두뇌
    python agent.py --mock "smoke test 과제"          # LLM 없이 그래프 검증
    python agent.py --vllm "..."                      # 로컬 vLLM 두뇌 (vllm serve 필요)

환경변수:
    CLAUDE_BIN      클로드 CLI 경로 (기본 claude) / CLAUDE_TIMEOUT 호출 타임아웃 초 (기본 180)
    VLLM_BASE_URL   기본 http://localhost:8000/v1
    VLLM_MODEL      기본 Qwen/Qwen2.5-7B-Instruct
    AGENT_WORKDIR   명령이 실행될 작업 디렉터리 (기본: ./sandbox)
    AGENT_CMD_TIMEOUT  명령 타임아웃 초 (기본 120)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_ITERS = 3

# ---------------------------------------------------------------- state


class AgentState(TypedDict):
    task: str
    plan: dict | None       # {"command", "success_criteria", "rationale"}
    history: list[dict]     # 실행 기록 [{iteration, plan, stdout, stderr, returncode, verdict, reason}]
    result: str
    verdict: str            # "" | done | retry | fail
    iteration: int


# ---------------------------------------------------------------- LLM 어댑터

PLANNER_SYSTEM = (
    "너는 연구 작업을 셸 명령으로 진행하는 계획자다. 과제와 지금까지의 실행 기록을 보고 "
    "다음에 실행할 셸 명령 '하나'와 성공 기준을 제안하라. 반드시 JSON 하나만 출력하라: "
    '{"command": "...", "success_criteria": "...", "rationale": "..."} '
    "명령은 작업 디렉터리 안에서만 동작해야 하며 파괴적 명령(rm -rf, sudo 등)은 금지다."
)

EVALUATOR_SYSTEM = (
    "너는 실행 결과 평가자다. 성공 기준과 명령의 stdout/stderr/returncode를 보고 판정하라. "
    '반드시 JSON 하나만 출력하라: {"verdict": "done|retry|fail", "reason": "..."} '
    "done=기준 충족, retry=다른 명령으로 재시도할 가치 있음, fail=이 과제는 이 루프로 달성 불가."
)


class MockLLM:
    """GPU/서버 없이 그래프 역학을 검증하기 위한 스크립트드 LLM."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if "계획자" in system:
            return json.dumps(
                {
                    "command": "echo 'hello from mock plan' && pwd",
                    "success_criteria": "stdout에 hello 문자열이 포함되고 returncode가 0",
                    "rationale": "스모크 테스트: 에코와 작업 디렉터리 확인",
                },
                ensure_ascii=False,
            )
        return json.dumps({"verdict": "done", "reason": "hello 출력과 returncode 0 확인"}, ensure_ascii=False)


class VllmLLM:
    """로컬 vLLM(OpenAI 호환)을 langchain-openai로 호출."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI  # 실제 모드에서만 임포트

        self.client = ChatOpenAI(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="EMPTY",
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=0.2,
        )

    def complete(self, system: str, user: str) -> str:
        resp = self.client.invoke([("system", system), ("user", user)])
        return resp.content


class ClaudeCliLLM:
    """Claude Code CLI headless(`claude -p`)를 두뇌로 사용 — 구독으로 처리, API 키 불필요.

    호출당 CLI 기동 오버헤드(수 초~수십 초)가 있으므로 노드 판단처럼 드문 호출에 적합.
    """

    def __init__(self) -> None:
        self.bin = os.environ.get("CLAUDE_BIN", "claude")

    def complete(self, system: str, user: str) -> str:
        prompt = (
            f"{system}\n\n입력:\n{user}\n\n"
            "주의: 도구를 사용하지 말고, 설명 없이 위 형식의 JSON 오브젝트 하나만 출력하라."
        )
        proc = subprocess.run(
            [self.bin, "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True,
            timeout=int(os.environ.get("CLAUDE_TIMEOUT", "180")),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI 실패 (rc={proc.returncode}): {proc.stderr[-500:]}")
        return proc.stdout


LLM: MockLLM | VllmLLM | ClaudeCliLLM  # main()에서 주입


def parse_json_block(text: str) -> dict:
    """응답에서 첫 JSON 오브젝트를 추출 (모델이 앞뒤로 말을 붙여도 견딤)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"JSON을 찾지 못함: {text[:200]!r}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------- 노드

DENY_PATTERNS = [
    r"\brm\s+-[a-zA-Z]*r",      # rm -r / -rf
    r"\bsudo\b",
    r"\bshutdown\b|\breboot\b|\bmkfs\b|\bdd\s+if=",
    r"\bcurl\b|\bwget\b",       # 외부 네트워크 금지 (v1 정책)
    r"\bgit\s+push\b",
    r"/etc/|/root\b|~/|\.\./",  # 작업 디렉터리 탈출 휴리스틱
    r":\(\)\s*\{",              # fork bomb
]


def planner(state: AgentState) -> dict:
    user = json.dumps(
        {"task": state["task"], "history": state["history"][-3:]}, ensure_ascii=False
    )
    plan = parse_json_block(LLM.complete(PLANNER_SYSTEM, user))
    print(f"[planner] iter {state['iteration'] + 1}: {plan.get('command')!r}")
    return {"plan": plan, "iteration": state["iteration"] + 1}


def executor(state: AgentState) -> dict:
    plan = state["plan"] or {}
    command = plan.get("command", "")
    workdir = os.environ.get("AGENT_WORKDIR", os.path.join(os.path.dirname(__file__), "sandbox"))
    os.makedirs(workdir, exist_ok=True)
    record = {"iteration": state["iteration"], "plan": plan}

    blocked = next((p for p in DENY_PATTERNS if re.search(p, command)), None)
    if blocked:
        record.update(stdout="", stderr=f"안전장치 거부 (패턴: {blocked})", returncode=-1)
        print(f"[executor] BLOCKED: {command!r}")
    else:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=workdir, capture_output=True, text=True,
                timeout=int(os.environ.get("AGENT_CMD_TIMEOUT", "120")),
            )
            record.update(
                stdout=proc.stdout[-4000:], stderr=proc.stderr[-2000:], returncode=proc.returncode
            )
            print(f"[executor] rc={proc.returncode}, stdout {len(proc.stdout)}B")
        except subprocess.TimeoutExpired:
            record.update(stdout="", stderr="타임아웃", returncode=-2)
            print("[executor] TIMEOUT")

    return {"history": state["history"] + [record]}


def evaluator(state: AgentState) -> dict:
    last = state["history"][-1]
    user = json.dumps(
        {
            "success_criteria": (state["plan"] or {}).get("success_criteria"),
            "stdout": last["stdout"], "stderr": last["stderr"], "returncode": last["returncode"],
        },
        ensure_ascii=False,
    )
    judged = parse_json_block(LLM.complete(EVALUATOR_SYSTEM, user))
    verdict = judged.get("verdict", "fail")
    if verdict == "retry" and state["iteration"] >= MAX_ITERS:
        verdict, judged["reason"] = "fail", f"최대 반복({MAX_ITERS}) 도달: {judged.get('reason')}"
    last.update(verdict=verdict, reason=judged.get("reason"))
    print(f"[evaluator] {verdict}: {judged.get('reason')}")
    return {"verdict": verdict, "history": state["history"]}


def finish(state: AgentState) -> dict:
    ok = state["verdict"] == "done"
    lines = [f"과제: {state['task']}", f"결과: {'성공' if ok else '실패'} ({state['iteration']}회 시도)"]
    for record in state["history"]:
        lines.append(
            f"  #{record['iteration']} {record['plan'].get('command')!r}"
            f" → rc={record['returncode']} [{record.get('verdict', '?')}] {record.get('reason', '')}"
        )
    result = "\n".join(lines)
    print(f"[finish]\n{result}")
    return {"result": result}


def route_after_eval(state: AgentState) -> str:
    return {"done": "finish", "retry": "planner", "fail": "finish"}[state["verdict"]]


# ---------------------------------------------------------------- 그래프


def build_app():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)
    graph.add_node("finish", finish)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges("evaluator", route_after_eval, {"planner": "planner", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def main() -> int:
    global LLM
    parser = argparse.ArgumentParser(description="LangGraph 연구 에이전트")
    parser.add_argument("task", help="수행할 과제 (자연어)")
    parser.add_argument("--mock", action="store_true", help="LLM 없이 그래프 스모크 테스트")
    parser.add_argument("--vllm", action="store_true", help="클로드 CLI 대신 로컬 vLLM 두뇌 사용")
    args = parser.parse_args()

    LLM = MockLLM() if args.mock else (VllmLLM() if args.vllm else ClaudeCliLLM())
    app = build_app()
    final = app.invoke(
        {"task": args.task, "plan": None, "history": [], "result": "", "verdict": "", "iteration": 0}
    )
    return 0 if final["verdict"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
