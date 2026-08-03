"""LangGraph 문제 정의 에이전트 — 초안→근거검증→다중해석→발산해소 루프.

`agent.py`(계획→실행→평가)의 **앞단**. 모호한 요청을 받아, 실행 에이전트가
정확히 풀 수 있는 스펙으로 수렴시킨다.

"잘 정의됨"의 조작적 정의: **스펙만 읽은 독립 해석자들이 "무엇을 만들지"와
"합격 판정"에 같은 답을 내놓는다.** 갈리면 그 스펙은 아직 모호하다 — 이
발산 테스트가 루프의 종료 조건이다.

사용법:
    python definer.py "승률 예측 좀 개선해줘"          # 기본: 클로드 CLI 두뇌, 헤드리스
    python definer.py --ask "..."                     # 발산 지점을 사용자에게 질문
    python definer.py --generic "..."                 # 도메인 무관 (형식 검사만)
    python definer.py --mock "..."                    # LLM 없이 그래프 검증
    python definer.py --vllm "..."                    # 로컬 vLLM 두뇌

    # 정의 → 실행 연결
    python definer.py "..." --out spec.md && python agent.py --spec spec.md

환경변수:
    DEFINER_BASE    근거 검증의 기준 디렉터리 (기본: 현재 작업 디렉터리)
    BACKLOG_PATH    백로그 파일 (기본 /workspace/RESEARCH_BACKLOG.md)
    그 외 LLM 관련 변수는 agent.py와 공유 (CLAUDE_BIN, VLLM_BASE_URL 등)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent import ClaudeCliLLM, VllmLLM, parse_json_block

MAX_ROUNDS = 3
DEFAULT_INTERPRETERS = 3

# ---------------------------------------------------------------- state


class DefinerState(TypedDict):
    request: str                 # 원문 요청 (모호해도 됨)
    mode: str                    # "workspace" | "generic"
    interactive: bool            # --ask 여부
    n_interpreters: int
    context: dict                # intake가 모은 주변 정보
    spec: dict | None            # 현재 초안
    grounding: list[dict]        # 근거 검증 결과 [{kind, detail, ok}]
    interpretations: list[dict]  # [{builds, passes_if}]
    divergences: list[dict]      # [{point, kind, question}]
    assumptions: list[str]       # 사용자 답변 / 헤드리스에서 명시한 가정
    round: int
    status: str                  # "" | converged | assumed
    out_path: str


# ---------------------------------------------------------------- 프롬프트

SPEC_FIELDS = (
    '{"goal": "이 문제가 풀리면 무엇이 달라지는가 (왜)", '
    '"inputs": ["실재해야 하는 입력·전제 — 경로·데이터·선행조건"], '
    '"deliverable": "산출물 하나를 정확히 (경로와 형식까지)", '
    '"verification": "합격 판정 방법 — 무엇을 보면 완수를 아는가", '
    '"verification_command": "위 판정을 수행하는 셸 명령 (불가능하면 빈 문자열)", '
    '"out_of_scope": ["명시적으로 범위 밖인 것"], '
    '"assumptions": ["확인하지 못한 채 택한 해석"]}'
)

DRAFTER_SYSTEM = (
    "너는 문제 정의자다. 모호한 요청을 실행 에이전트가 오해 없이 수행할 수 있는 스펙으로 만든다. "
    "규칙: (1) 요청에 없는 기능을 추가하지 마라 — 범위를 넓히는 것은 정의가 아니라 왜곡이다. "
    "(2) deliverable은 산출물 '하나'를 경로와 형식까지 지정하라. "
    "(3) verification은 사람의 주관이 아니라 확인 가능한 사실로 써라. 가능하면 verification_command에 "
    "실제 셸 명령을 넣어라. (4) 확인할 수 없어 택한 해석은 반드시 assumptions에 적어라 — "
    "모르는 것을 아는 척 채우지 마라. (5) 근거 검증(grounding)에서 존재하지 않는다고 나온 경로는 "
    "전제로 쓰지 말고, 그 경로를 만들거나 찾는 것을 스펙에 반영하라. "
    f"반드시 JSON 하나만 출력하라: {SPEC_FIELDS}"
)

INTERPRETER_SYSTEM = (
    "너는 스펙 해석자다. 아래 스펙만 보고(원래 요청은 모른다고 가정하고) 네가 실제로 무엇을 만들지, "
    "그리고 무엇을 확인하면 합격이라고 판단할지 각각 한 문장으로 답하라. 스펙에 없는 내용을 "
    "선의로 보충하지 마라 — 비어 있으면 비어 있는 대로 드러내라. "
    "그리고 수행하려면 필요한데 스펙에 정의돼 있지 않아 네가 임의로 정해야 했던 것을 undefined에 "
    "모두 나열하라 (용어의 판정 기준, 단위를 나누는 규칙, 값이 지정 안 된 항목 등). "
    '반드시 JSON 하나만 출력하라: {"builds": "...", "passes_if": "...", "undefined": ["..."]}'
)

DIVERGENCE_SYSTEM = (
    "너는 해석 비교자다. 같은 스펙을 읽은 해석자들의 답을 비교해 '실질적으로' 갈린 지점만 찾아라. "
    "실질적 발산 = 만들 산출물이 다르거나, 같은 결과물을 놓고 합격/불합격 판정이 갈릴 수 있는 경우. "
    "표현 차이나 상세도 차이는 발산이 아니다 — 억지로 만들어내지 마라. "
    "단, 해석자들의 답이 서로 일치하더라도 그들이 undefined에 신고한 항목은 별개로 판단하라: "
    "그 항목이 미정의인 탓에 합격 판정이 갈릴 수 있다면 그것도 발산이다. "
    "해석자들이 우연히 같은 임의 선택을 했다는 사실은 스펙이 그 선택을 지정했다는 뜻이 아니다. "
    "각 발산을 분류하라: kind='resolvable'(스펙을 더 명확히 쓰면 해소됨) 또는 "
    "kind='user'(요청자의 의도를 물어야만 해소됨 — 어느 쪽도 합리적인 선택지일 때). "
    "kind='user'면 question에 물어볼 질문 한 문장을 써라. "
    '반드시 JSON 하나만 출력하라: {"verdict": "converged|revise|ask", '
    '"divergences": [{"point": "...", "kind": "resolvable|user", "question": "..."}]} '
    "verdict: converged=실질 발산 없음, revise=resolvable만 있음, ask=user 발산이 있음."
)


class MockDefinerLLM:
    """LLM 없이 그래프 역학(발산→재작성→수렴)을 검증하는 스크립트드 두뇌."""

    def __init__(self) -> None:
        self.calls = 0
        self.compare_rounds = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        # 비교자 프롬프트에도 '해석자'라는 낱말이 들어가므로 판정 순서가 중요하다
        if "비교자" in system:
            self.compare_rounds += 1
            if self.compare_rounds == 1:
                return json.dumps(
                    {
                        "verdict": "revise",
                        "divergences": [
                            {
                                "point": "산출물이 표준출력인지 파일인지 갈림 (1라운드 발산 반영 필요)",
                                "kind": "resolvable",
                                "question": "",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            return json.dumps({"verdict": "converged", "divergences": []}, ensure_ascii=False)
        if "정의자" in system:
            resolved = "1라운드 발산 반영" in user
            return json.dumps(
                {
                    "goal": "sandbox 안 csv의 행 수를 알아 데이터 규모를 파악한다",
                    "inputs": ["sandbox/ 디렉터리의 csv 파일"],
                    "deliverable": ("sandbox/rowcount.md" if resolved else "행 수 리포트"),
                    "verification": "리포트 파일이 존재하고 각 csv마다 한 줄씩 행 수가 적혀 있다",
                    "verification_command": "test -s sandbox/rowcount.md && cat sandbox/rowcount.md",
                    "out_of_scope": ["csv 내용 분석"],
                    "assumptions": ["헤더 행은 데이터 행에서 제외한다"],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "builds": "csv 행 수 리포트",
                "passes_if": "리포트에 파일별 행 수가 있다",
                "undefined": ["헤더 행을 셀지 여부"],
            },
            ensure_ascii=False,
        )


LLM: MockDefinerLLM | VllmLLM | ClaudeCliLLM  # main()에서 주입


# ---------------------------------------------------------------- 노드

EXTENSIONS = "csv|json|md|py|txt|parquet|yaml|yml|ipynb|pkl|pt"

PATH_RE = re.compile(
    r"(?:/?[\w.~-]+(?:/[\w.~-]*)+)"                    # 슬래시 포함 경로 (끝의 / 도 허용)
    rf"|(?:\b[\w-]+\.(?:{EXTENSIONS})\b)"              # 확장자 있는 파일명
)

# "로그·.json/.csv 결과" 같은 확장자 나열이 경로로 오인되는 것을 막는다.
# 유령 경로 하나가 스펙을 오염시키고 재작성 루프를 낭비시킨 실동 사례가 있었다.
BARE_EXT_RE = re.compile(rf"^\.(?:{EXTENSIONS})$", re.I)


def intake(state: DefinerState) -> dict:
    """근거 검증에 쓸 주변 컨텍스트를 수집 (LLM 없이)."""
    if state["mode"] == "generic":
        print("[intake] generic 모드 — 컨텍스트 수집 생략")
        return {"context": {"mode": "generic"}}

    base = os.environ.get("DEFINER_BASE", os.getcwd())
    context: dict = {"mode": "workspace", "base": base}
    try:
        context["base_entries"] = sorted(os.listdir(base))[:40]
    except OSError as exc:
        context["base_entries"] = f"읽기 실패: {exc}"

    backlog = os.environ.get("BACKLOG_PATH", "/workspace/RESEARCH_BACKLOG.md")
    if os.path.exists(backlog):
        context["backlog_excerpt"] = open(backlog, encoding="utf-8").read()[:1500]

    print(f"[intake] base={base}, 항목 {len(context.get('base_entries', []))}개")
    return {"context": context}


def drafter(state: DefinerState) -> dict:
    payload = {
        "request": state["request"],
        "context": state["context"],
        "previous_spec": state["spec"],
        "grounding": state["grounding"],
        "divergences_to_fix": state["divergences"],
        "assumptions_or_answers": state["assumptions"],
    }
    spec = parse_json_block(LLM.complete(DRAFTER_SYSTEM, json.dumps(payload, ensure_ascii=False)))
    print(f"[drafter] round {state['round'] + 1}: {spec.get('deliverable')!r}")
    return {"spec": spec, "round": state["round"] + 1, "divergences": []}


def grounder(state: DefinerState) -> dict:
    """스펙이 전제한 것들이 실재하는지 코드로 확인 — LLM 추론이 아니라 파일시스템 사실."""
    spec = state["spec"] or {}
    checks: list[dict] = []

    for field in ("goal", "deliverable", "verification"):
        if not str(spec.get(field, "")).strip():
            checks.append({"kind": "형식", "detail": f"필수 필드 '{field}'가 비어 있음", "ok": False})
    if not str(spec.get("verification_command", "")).strip():
        checks.append(
            {"kind": "형식", "detail": "verification_command 없음 — 판정이 사람 주관에 의존한다", "ok": False}
        )

    if state["mode"] == "workspace":
        base = state["context"].get("base", os.getcwd())
        seen: set[str] = set()
        for raw in PATH_RE.findall(" ".join(str(v) for v in spec.get("inputs", []))):
            candidate = raw.strip("'\"`,.)")
            # 한글만으로 된 토큰("입력/출력" 같은 서술)은 경로가 아니다 — 오탐 차단
            if candidate in seen or len(candidate) < 3 or not re.search(r"[A-Za-z0-9]", candidate):
                continue
            if any(BARE_EXT_RE.match(seg) for seg in candidate.split("/") if seg):
                continue  # ".json/.csv" 같은 확장자 나열
            seen.add(candidate)
            full = candidate if os.path.isabs(candidate) else os.path.join(base, candidate)
            exists = os.path.exists(full)
            checks.append(
                {
                    "kind": "경로",
                    "detail": f"입력 전제 {candidate!r} → {'존재함' if exists else '존재하지 않음'} ({full})",
                    "ok": exists,
                }
            )

    failed = [c for c in checks if not c["ok"]]
    print(f"[grounder] 검사 {len(checks)}건, 문제 {len(failed)}건")
    for check in failed:
        print(f"  ✗ {check['detail']}")
    return {"grounding": checks}


def interpreters(state: DefinerState) -> dict:
    """스펙만 보여주고 '무엇을 만들고 어떻게 합격 판정하나'를 독립적으로 받는다."""
    spec_only = json.dumps(state["spec"], ensure_ascii=False)

    def ask_one(index: int) -> dict:
        try:
            return parse_json_block(LLM.complete(INTERPRETER_SYSTEM, spec_only))
        except Exception as exc:  # 해석자 1명 실패가 라운드 전체를 죽이면 안 됨
            print(f"[interpreters] #{index + 1} 실패: {exc}")
            return {}

    count = state["n_interpreters"]
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = [r for r in pool.map(ask_one, range(count)) if r]

    for i, r in enumerate(results):
        print(f"[interpreters] #{i + 1} builds={r.get('builds')!r} passes_if={r.get('passes_if')!r}")
        for item in r.get("undefined", []):
            print(f"    ? 미정의: {item}")
    return {"interpretations": results}


def divergence(state: DefinerState) -> dict:
    if len(state["interpretations"]) < 2:
        print("[divergence] 해석자 응답 부족 — 비교 생략, 가정 하에 확정")
        return {"divergences": [], "status": "assumed"}

    payload = {"spec": state["spec"], "interpretations": state["interpretations"]}
    judged = parse_json_block(LLM.complete(DIVERGENCE_SYSTEM, json.dumps(payload, ensure_ascii=False)))
    found = judged.get("divergences", [])
    verdict = judged.get("verdict", "converged")

    print(f"[divergence] {verdict} — 발산 {len(found)}건")
    for d in found:
        print(f"  · [{d.get('kind')}] {d.get('point')}")

    if verdict == "converged" or not found:
        return {"divergences": [], "status": "converged"}
    if state["round"] >= MAX_ROUNDS:
        print(f"[divergence] 최대 라운드({MAX_ROUNDS}) 도달 — 남은 발산을 가정으로 기록하고 확정")
        return {
            "divergences": found,
            "status": "assumed",
            "assumptions": state["assumptions"]
            + [f"미해소 발산: {d.get('point')}" for d in found],
        }
    return {"divergences": found, "status": ""}


def resolve(state: DefinerState) -> dict:
    """사용자 판단이 필요한 발산 처리 — 대화형이면 질문, 헤드리스면 가정으로 명시."""
    added: list[str] = []
    for d in state["divergences"]:
        if d.get("kind") != "user":
            continue
        question = d.get("question") or d.get("point", "")
        if state["interactive"]:
            print(f"\n[resolve] ❓ {question}")
            answer = input("    답변 (엔터 = 에이전트 판단에 맡김): ").strip()
            added.append(f"{question} → {answer}" if answer else f"{question} → 사용자 위임, 합리적 해석 채택")
        else:
            added.append(f"{question} → 확인 불가, 가장 합리적인 해석을 택하고 스펙에 명시할 것")
    if added:
        print(f"[resolve] {len(added)}건 반영 ({'대화형' if state['interactive'] else '헤드리스 가정'})")
    return {"assumptions": state["assumptions"] + added}


def finalize(state: DefinerState) -> dict:
    spec = state["spec"] or {}
    status = state["status"] or "assumed"

    def bullets(items: object) -> str:
        values = items if isinstance(items, list) else ([items] if items else [])
        return "\n".join(f"- {v}" for v in values) or "- (없음)"

    command = str(spec.get("verification_command", "")).strip()
    lines = [
        f"# 문제 정의: {spec.get('deliverable', '(미지정)')}",
        "",
        f"> 원문 요청: {state['request']}",
        f"> definer.py · {state['round']}라운드 · 상태: "
        f"{'수렴 (해석 일치)' if status == 'converged' else '가정 하에 확정'}",
        "",
        "## 목표",
        spec.get("goal", "(미지정)"),
        "",
        "## 입력·전제",
        bullets(spec.get("inputs")),
        "",
        "## 산출물",
        spec.get("deliverable", "(미지정)"),
        "",
        "## 검증 방법",
        spec.get("verification", "(미지정)"),
    ]
    if command:
        lines += ["", "```bash", command, "```"]
    lines += [
        "",
        "## 범위 밖",
        bullets(spec.get("out_of_scope")),
        "",
        "## 가정 (확인되지 않음)",
        bullets(list(spec.get("assumptions") or []) + state["assumptions"]),
        "",
        "## 근거 검증 (grounder)",
        "\n".join(f"- {'✅' if c['ok'] else '⚠️'} {c['detail']}" for c in state["grounding"]) or "- (없음)",
        "",
    ]
    document = "\n".join(lines)

    with open(state["out_path"], "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"\n[finalize] {state['out_path']} 작성 ({status}, {state['round']}라운드)\n")
    print(document)
    return {"status": status}


def route_after_grounding(state: DefinerState) -> str:
    """근거 검증 실패는 해석 비교보다 먼저 고친다 — 존재하지 않는 전제 위에 세운 스펙은
    해석이 완벽히 일치해도 실행 시점에 무너진다."""
    failed = [c for c in state["grounding"] if not c["ok"]]
    if failed and state["round"] < MAX_ROUNDS:
        print(f"[grounder] 문제 {len(failed)}건 → 스펙 재작성")
        return "drafter"
    return "interpreters"


def route_after_divergence(state: DefinerState) -> str:
    if state["status"]:
        return "finalize"
    if any(d.get("kind") == "user" for d in state["divergences"]):
        return "resolve"
    return "drafter"


# ---------------------------------------------------------------- 그래프


def build_app():
    graph = StateGraph(DefinerState)
    graph.add_node("intake", intake)
    graph.add_node("drafter", drafter)
    graph.add_node("grounder", grounder)
    graph.add_node("interpreters", interpreters)
    graph.add_node("divergence", divergence)
    graph.add_node("resolve", resolve)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "drafter")
    graph.add_edge("drafter", "grounder")
    graph.add_conditional_edges(
        "grounder",
        route_after_grounding,
        {"drafter": "drafter", "interpreters": "interpreters"},
    )
    graph.add_edge("interpreters", "divergence")
    graph.add_conditional_edges(
        "divergence",
        route_after_divergence,
        {"drafter": "drafter", "resolve": "resolve", "finalize": "finalize"},
    )
    graph.add_edge("resolve", "drafter")
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> int:
    global LLM
    parser = argparse.ArgumentParser(description="LangGraph 문제 정의 에이전트")
    parser.add_argument("request", help="정의할 요청 (모호해도 됨)")
    parser.add_argument("--ask", action="store_true", help="사용자 판단이 필요한 발산을 직접 질문 (기본: 헤드리스)")
    parser.add_argument("--generic", action="store_true", help="도메인 무관 모드 — 경로 실재 검증 없이 형식 검사만")
    parser.add_argument("--out", default="spec.md", help="스펙 출력 경로 (기본 spec.md)")
    parser.add_argument("--interpreters", type=int, default=DEFAULT_INTERPRETERS, help="해석자 수 (기본 3)")
    parser.add_argument("--mock", action="store_true", help="LLM 없이 그래프 스모크 테스트")
    parser.add_argument("--vllm", action="store_true", help="클로드 CLI 대신 로컬 vLLM 두뇌 사용")
    args = parser.parse_args()

    LLM = MockDefinerLLM() if args.mock else (VllmLLM() if args.vllm else ClaudeCliLLM())
    app = build_app()
    final = app.invoke(
        {
            "request": args.request,
            "mode": "generic" if args.generic else "workspace",
            "interactive": args.ask,
            "n_interpreters": max(2, args.interpreters),
            "context": {},
            "spec": None,
            "grounding": [],
            "interpretations": [],
            "divergences": [],
            "assumptions": [],
            "round": 0,
            "status": "",
            "out_path": args.out,
        },
        {"recursion_limit": 50},
    )
    return 0 if final["status"] == "converged" else 1


if __name__ == "__main__":
    sys.exit(main())
