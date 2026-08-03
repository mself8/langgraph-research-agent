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
    frame: str                   # "task"(만들 것) | "research"(알아낼 것)
    interactive: bool            # --ask 여부
    n_interpreters: int
    context: dict                # intake가 모은 주변 정보
    evidence: str                # surveyor.py가 모은 근거 목록 (--evidence)
    spec: dict | None            # 현재 초안
    grounding: list[dict]        # 근거 검증 결과 [{kind, detail, ok}]
    inventory: list[str]         # input_paths 하위에 실제로 있는 파일 목록
    grounding_seen: list[str]    # 라운드별 실패 서명 — 같은 실패 반복 시 재작성 포기
    interpretations: list[dict]  # [{builds, passes_if}]
    divergences: list[dict]      # [{point, kind, question}]
    assumptions: list[str]       # 사용자 답변 / 헤드리스에서 명시한 가정
    round: int
    status: str                  # "" | converged | assumed
    out_path: str


# ---------------------------------------------------------------- 프롬프트

RESEARCH_FIELDS = (
    '{"question": "답을 알아내려는 질문 하나. 측정 가능한 형태로 (\'좋아지나?\'가 아니라 '
    '\'무엇이 얼마나 달라지나?\')", '
    '"hypothesis": "예상하는 답과 그렇게 보는 근거", '
    '"falsification": "가설이 틀렸다면 무엇이 관측되는가 — 이 값이 나오면 가설을 버린다는 구체적 기준. '
    '이게 없으면 연구가 아니라 그냥 작업이다", '
    '"data": ["쓸 데이터와 그 출처"], '
    '"input_paths": ["실재해야 하는 파일·디렉터리 경로만. 경로 문자열 하나씩, 설명 금지. '
    '\'로그/CSV/JSON\' 같은 형식 나열은 경로가 아니다. 없으면 빈 배열"], '
    '"baseline": "무엇과 비교해서 판단하는가 — 비교 대상이 없으면 결론이 남지 않는다", '
    '"metric": "무엇을 재는가 + 유의미하다고 볼 임계값", '
    '"method": "가설을 검정하기 위해 실제로 무엇을 실행하는가", '
    '"out_of_scope": ["이 연구에서 답하지 않는 것"], '
    '"assumptions": ["확인하지 못한 채 택한 해석"]}'
)

SPEC_FIELDS = (
    '{"goal": "이 문제가 풀리면 무엇이 달라지는가 (왜)", '
    '"inputs": ["실재해야 하는 입력·전제 — 경로·데이터·선행조건 (서술 가능)"], '
    '"input_paths": ["위 전제 중 실재해야 하는 파일·디렉터리 경로만. 경로 문자열 하나씩, 설명 금지. '
    '\'로그/CSV/JSON\' 같은 나열이나 형식 이름은 경로가 아니므로 넣지 마라. 없으면 빈 배열"], '
    '"deliverable": "산출물 하나를 정확히 (경로와 형식까지)", '
    '"verification": "합격 판정 방법 — 무엇을 보면 완수를 아는가", '
    '"verification_command": "위 판정을 수행하는 셸 명령 (불가능하면 빈 문자열)", '
    '"out_of_scope": ["명시적으로 범위 밖인 것"], '
    '"assumptions": ["확인하지 못한 채 택한 해석"]}'
)

DRAFTER_SYSTEM = (
    "너는 문제 정의자다. 모호한 요청을 실행 에이전트가 오해 없이 수행할 수 있는 스펙으로 만든다. "
    "규칙: (1) 요청에 없는 기능을 추가하지 마라 — 범위를 넓히는 것은 정의가 아니라 왜곡이다. "
    "(2) deliverable은 산출물 '하나'의 경로와 형식만 한 문장으로 써라 — 세부 규칙은 verification과 "
    "assumptions로 나눠 담아라. deliverable에 문단을 밀어넣지 마라. "
    "(3) verification은 사람의 주관이 아니라 확인 가능한 사실로 써라. 가능하면 verification_command에 "
    "실제 셸 명령을 넣어라. (4) 확인할 수 없어 택한 해석은 반드시 assumptions에 적어라 — "
    "모르는 것을 아는 척 채우지 마라. (5) 근거 검증(grounding)에서 존재하지 않는다고 나온 경로는 "
    "전제로 쓰지 말고, 그 경로를 만들거나 찾는 것을 스펙에 반영하라. "
    f"반드시 JSON 하나만 출력하라: {SPEC_FIELDS}"
)

RESEARCH_DRAFTER_SYSTEM = (
    "너는 연구 문제 정의자다. 막연한 연구 의향을 '무엇을 알아낼 것인가'가 분명한 정의로 만든다. "
    "규칙: (1) question은 답이 측정 가능한 형태여야 한다 — '성능이 좋아지나?'는 질문이 아니다. "
    "(2) falsification이 이 정의의 핵심이다. 가설이 틀렸을 때 무엇이 관측되는지를 구체적 수치·조건으로 "
    "써라. 어떤 결과가 나와도 '역시 맞았다'로 읽히는 정의는 실패한 정의다. "
    "(3) baseline 없이는 어떤 수치도 의미가 없다 — 무엇과 비교하는지 반드시 정하라. "
    "(4) 요청에 없는 연구 범위를 넓히지 마라. 한 번에 답할 수 있는 질문 하나로 좁혀라. "
    "(5) 파일 목록(inventory)이 주어지면 그것에 근거해 구체적으로 써라 — 있지도 않은 데이터를 "
    "전제하지 말고, 확인 못 한 것은 assumptions에 적어라. "
    "(6) 근거(evidence)가 주어지면 baseline·metric·falsification은 반드시 그 안의 출처에 기대어 "
    "정하고, 어느 근거를 썼는지 해당 필드에 함께 밝혀라. 근거에 없는 수치를 지어내지 마라. "
    f"반드시 JSON 하나만 출력하라: {RESEARCH_FIELDS}"
)

INTERPRETER_SYSTEM = (
    "너는 스펙 해석자다. 아래 스펙만 보고(원래 요청은 모른다고 가정하고) 네가 실제로 무엇을 만들지, "
    "그리고 무엇을 확인하면 합격이라고 판단할지 각각 한 문장으로 답하라. 스펙에 없는 내용을 "
    "선의로 보충하지 마라 — 비어 있으면 비어 있는 대로 드러내라. "
    "그리고 스펙에 정의돼 있지 않아 네가 임의로 정해야 했던 것 중 **다르게 정했다면 합격/불합격 "
    "판정이 뒤집혔을 것만** undefined에 최대 3개까지 적어라. 서식·표기·정렬·소수점 자릿수·문서 "
    "언어처럼 판정을 바꾸지 않는 세부는 절대 넣지 마라. 없으면 빈 배열로 두라. "
    '반드시 JSON 하나만 출력하라: {"builds": "...", "passes_if": "...", "undefined": ["..."]}'
)

RESEARCH_INTERPRETER_SYSTEM = (
    "너는 연구 정의 해석자다. 아래 정의만 보고(원래 요청은 모른다고 가정하고) 네가 실제로 무엇을 "
    "실행할지, 그리고 **어떤 결과가 나오면 가설이 틀렸다고 결론낼지** 각각 한 문장으로 답하라. "
    "정의에 없는 내용을 선의로 보충하지 마라 — 비어 있으면 비어 있는 대로 드러내라. "
    "그리고 정의돼 있지 않아 네가 임의로 정해야 했던 것 중 **다르게 정했다면 결론이 뒤집혔을 것만** "
    "undefined에 최대 3개까지 적어라. 표기·서식처럼 결론을 바꾸지 않는 세부는 넣지 마라. "
    '반드시 JSON 하나만 출력하라: {"builds": "실행할 것", "passes_if": "가설을 버리는 조건", '
    '"undefined": ["..."]}'
)

DIVERGENCE_SYSTEM = (
    "너는 해석 비교자다. 같은 스펙을 읽은 해석자들의 답을 비교해 '실질적으로' 갈린 지점만 찾아라. "
    "실질적 발산 = 만들 산출물이 다르거나, 같은 결과물을 놓고 합격/불합격 판정이 갈릴 수 있는 경우. "
    "표현 차이나 상세도 차이는 발산이 아니다 — 억지로 만들어내지 마라. "
    "각 발산을 분류하라: kind='resolvable'(스펙을 더 명확히 쓰면 해소됨) 또는 "
    "kind='user'(요청자의 의도를 물어야만 해소됨 — 어느 쪽도 합리적인 선택지일 때). "
    "kind='user'면 question에 물어볼 질문 한 문장을 써라. "
    "그리고 별도로, 해석자들이 undefined에 신고한 항목 중 **합격 판정을 실제로 뒤집을 수 있는 것만** "
    "중복을 합쳐 최대 3개까지 blocking_undefined에 담아라(발산과 같은 형식으로 kind도 분류하라). "
    "해석자들의 답이 우연히 일치했다는 사실은 스펙이 그 선택을 지정했다는 뜻이 아니다 — "
    "답의 일치와 무관하게 판단하라. "
    '반드시 JSON 하나만 출력하라: {"divergences": [{"point": "...", "kind": "resolvable|user", '
    '"question": "..."}], "blocking_undefined": [{"point": "...", "kind": "resolvable|user", '
    '"question": "..."}]}'
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
                        "divergences": [
                            {
                                "point": "산출물이 표준출력인지 파일인지 갈림 (1라운드 발산 반영 필요)",
                                "kind": "resolvable",
                                "question": "",
                            }
                        ],
                        "blocking_undefined": [
                            {
                                "point": "헤더 행을 데이터 행에 포함하는지",
                                "kind": "user",
                                "question": "헤더 행도 데이터 행으로 셀까요?",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            return json.dumps({"divergences": [], "blocking_undefined": []}, ensure_ascii=False)
        if "정의자" in system:
            resolved = "1라운드 발산 반영" in user
            return json.dumps(
                {
                    "goal": "sandbox 안 csv의 행 수를 알아 데이터 규모를 파악한다",
                    "inputs": ["sandbox/ 디렉터리의 csv 파일 (로그/CSV/JSON 형식 나열은 경로가 아니다)"],
                    "input_paths": ["sandbox"],
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


def load_evidence(path: str) -> str:
    """surveyor.py가 만든 근거 목록 — 베이스라인·평가지표·임계값을 지어내지 않게 하는 재료."""
    if not path:
        return ""
    text = open(path, encoding="utf-8").read()
    print(f"[intake] 근거 {path} 로드 ({len(text)}자)")
    return text


def drafter(state: DefinerState) -> dict:
    payload = {
        "request": state["request"],
        "context": state["context"],
        "previous_spec": state["spec"],
        "evidence": state["evidence"],
        "grounding": state["grounding"],
        "inventory": state["inventory"],
        "divergences_to_fix": state["divergences"],
        "assumptions_or_answers": state["assumptions"],
    }
    system = RESEARCH_DRAFTER_SYSTEM if state["frame"] == "research" else DRAFTER_SYSTEM
    spec = parse_json_block(LLM.complete(system, json.dumps(payload, ensure_ascii=False)))
    headline = spec.get("question") if state["frame"] == "research" else spec.get("deliverable")
    print(f"[drafter] round {state['round'] + 1}: {headline!r}")
    return {"spec": spec, "round": state["round"] + 1, "divergences": []}


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def explore(path: str, limit: int = 60) -> list[str]:
    """경로 하위에 실제로 뭐가 있는지 목록화 (LLM 없이).

    실동에서 수렴을 막은 것은 문구의 모호함이 아니라 '폴더 안에 뭐가 있는지 아무도 모른다'는
    사실이었다. 정의만 다듬어선 영영 풀리지 않고, ls 한 번이면 사라지는 질문이었다.
    """
    if os.path.isfile(path):
        return [os.path.basename(path)]
    found: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(files):
            found.append(os.path.relpath(os.path.join(root, name), path))
            if len(found) >= limit:
                return found
    return found


def grounder(state: DefinerState) -> dict:
    """스펙이 전제한 것들이 실재하는지 코드로 확인 — LLM 추론이 아니라 파일시스템 사실."""
    spec = state["spec"] or {}
    checks: list[dict] = []
    inventory: list[str] = []

    required = (
        ("question", "falsification", "baseline")
        if state["frame"] == "research"
        else ("goal", "deliverable", "verification")
    )
    for field in required:
        if not str(spec.get(field, "")).strip():
            checks.append({"kind": "형식", "detail": f"필수 필드 '{field}'가 비어 있음", "ok": False})
    if state["frame"] == "task" and not str(spec.get("verification_command", "")).strip():
        checks.append(
            {"kind": "형식", "detail": "verification_command 없음 — 판정이 사람 주관에 의존한다", "ok": False}
        )

    if state["mode"] == "workspace":
        # 경로는 산문에서 뽑지 않고 drafter가 내놓은 전용 필드만 본다. 정규식으로 한국어 서술에서
        # 경로를 추출하려다 '.json/.csv', '입력/출력', '로그/CSV/JSON/노트북'을 차례로 경로로
        # 오인했고, 그 유령 하나가 스펙에 기입되어 라운드를 태웠다.
        base = state["context"].get("base", os.getcwd())
        for candidate in dict.fromkeys(str(p).strip() for p in spec.get("input_paths", [])):
            if not candidate:
                continue
            full = candidate if os.path.isabs(candidate) else os.path.join(base, candidate)
            exists = os.path.exists(full)
            checks.append(
                {
                    "kind": "경로",
                    "detail": f"입력 전제 {candidate!r} → {'존재함' if exists else '존재하지 않음'} ({full})",
                    "ok": exists,
                }
            )
            if exists:
                inventory += [f"{candidate}/{item}" for item in explore(full)]

    failed = [c for c in checks if not c["ok"]]
    print(f"[grounder] 검사 {len(checks)}건, 문제 {len(failed)}건, 탐색된 파일 {len(inventory)}개")
    for check in failed:
        print(f"  ✗ {check['detail']}")

    signature = "|".join(sorted(c["detail"] for c in failed))
    return {
        "grounding": checks,
        "inventory": inventory,
        "grounding_seen": state["grounding_seen"] + [signature],
    }


def interpreters(state: DefinerState) -> dict:
    """스펙만 보여주고 '무엇을 만들고 어떻게 합격 판정하나'를 독립적으로 받는다."""
    spec_only = json.dumps(state["spec"], ensure_ascii=False)
    system = RESEARCH_INTERPRETER_SYSTEM if state["frame"] == "research" else INTERPRETER_SYSTEM

    def ask_one(index: int) -> dict:
        try:
            return parse_json_block(LLM.complete(system, spec_only))
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

    # 수렴 판정은 코드가 한다. 비교자에게 verdict 토큰을 맡겼더니, 판정을 뒤집는 미정의 항목을
    # 잔뜩 쥐고도 "해석이 일치한다"는 이유로 converged를 내는 실동 사례가 있었다.
    found = list(judged.get("divergences", []))
    for item in judged.get("blocking_undefined", []):
        entry = item if isinstance(item, dict) else {"point": item, "kind": "resolvable"}
        found.append({**entry, "point": f"미정의: {entry.get('point')}"})

    # 두 라운드를 고쳐 썼는데도 남은 항목은 '스펙을 더 명확히 쓰면 해소'되는 것이 아니다.
    # 실동에서 비교자가 '어떤 파일이 하나의 실험인가'를 계속 resolvable로 분류했고, drafter가
    # 규칙을 덧붙일수록 새 경계 케이스가 생겨 발산이 3→5건으로 늘었다. 요청자 판단으로 올린다.
    if state["round"] >= 2 and found:
        found = [d if d.get("kind") == "user" else {**d, "kind": "user"} for d in found]
        print(f"[divergence] {state['round']}라운드째 미해소 → 요청자 판단 항목으로 승격")

    print(f"[divergence] 발산 {len(found)}건")
    for d in found:
        print(f"  · [{d.get('kind')}] {d.get('point')}")

    if not found:
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
            try:
                answer = input("    답변 (엔터 = 에이전트 판단에 맡김): ").strip()
            except EOFError:  # 비대화 환경(파이프·크론)에서 --ask가 죽지 않도록
                print("    (stdin 없음 — 헤드리스로 강등)")
                answer = ""
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

    research = state["frame"] == "research"
    command = str(spec.get("verification_command", "")).strip()
    headline = spec.get("question") if research else spec.get("deliverable")
    lines = [
        f"# {'연구 문제 정의' if research else '문제 정의'}: {headline or '(미지정)'}",
        "",
        f"> 원문 요청: {state['request']}",
        f"> definer.py · {state['round']}라운드 · 상태: "
        f"{'수렴 (해석 일치)' if status == 'converged' else '가정 하에 확정'}",
        "",
    ]
    if research:
        lines += [
            "## 연구 질문",
            spec.get("question", "(미지정)"),
            "",
            "## 가설",
            spec.get("hypothesis", "(미지정)"),
            "",
            "## 반증 조건 — 이게 관측되면 가설을 버린다",
            spec.get("falsification", "(미지정)"),
            "",
            "## 데이터",
            bullets(spec.get("data")),
            "",
            "## 베이스라인",
            spec.get("baseline", "(미지정)"),
            "",
            "## 평가 지표·임계값",
            spec.get("metric", "(미지정)"),
            "",
            "## 방법",
            spec.get("method", "(미지정)"),
        ]
    else:
        lines += [
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
        f"## 탐색된 파일 ({len(state['inventory'])}개)",
        bullets(state["inventory"][:30]),
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
    해석이 완벽히 일치해도 실행 시점에 무너진다.

    단 '같은 실패가 다시 나오면' 되돌리지 않는다. drafter가 못 고치는 실패(대개 grounder
    오탐)를 붙들고 라운드를 태우다 스펙만 비대해진 실동 사례가 있었다.
    """
    failed = [c for c in state["grounding"] if not c["ok"]]
    if not failed or state["round"] >= MAX_ROUNDS:
        return "interpreters"
    if state["grounding_seen"].count(state["grounding_seen"][-1]) > 1:
        print(f"[grounder] 같은 문제 {len(failed)}건이 재발 → 재작성 포기하고 진행")
        return "interpreters"
    print(f"[grounder] 문제 {len(failed)}건 → 스펙 재작성")
    return "drafter"


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
    parser.add_argument("--research", action="store_true", help="'만들 것'이 아니라 '알아낼 것'을 정의 (질문·가설·반증조건)")
    parser.add_argument("--evidence", default="", help="surveyor.py가 만든 근거 파일 (베이스라인·지표를 근거로 정하게 한다)")
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
            "frame": "research" if args.research else "task",
            "interactive": args.ask,
            "n_interpreters": max(2, args.interpreters),
            "context": {},
            "evidence": load_evidence(args.evidence),
            "spec": None,
            "grounding": [],
            "inventory": [],
            "grounding_seen": [],
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
