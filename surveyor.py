"""LangGraph 조사 에이전트 — 논문·기술 서칭 + 다관점 조사 → 근거 목록.

`definer.py`(문제 정의)의 **앞단**. 막연한 연구 의향을 받아 바깥 근거를 모은다.

왜 필요한가: 정의 단계에서 계속 막힌 것이 "베이스라인은 뭘로?", "평가 지표는?",
"임계값은?" 이었다. 이건 스펙 문구를 다듬어서 풀리는 게 아니라 **선행연구가 답해주는
것**이다. 근거 없이 정의하면 LLM이 그럴듯하게 지어내거나 가정으로 쌓인다.

파이프라인:
    surveyor.py → evidence.md → definer.py --evidence → spec.md → agent.py --spec
    (조사)                      (정의)                    (해결·실험)

사용법:
    python surveyor.py "GNN으로 추천 라인업 연구를 하려고 해"
    python surveyor.py "..." --web 0          # 웹검색 끄기 (arXiv + 로컬만, 빠름)
    python surveyor.py "..." --web 4          # 렌즈 4개 전부 웹검색 (느림·비쌈)
    python surveyor.py "..." --mock           # LLM 없이 그래프 검증

환경변수:
    SURVEY_BASE     로컬 문서 검색 기준 디렉터리 (기본 /workspace)
    CODEX_BIN       codex CLI 경로 (기본 codex) / CODEX_TIMEOUT 초 (기본 240)
    그 외 LLM 관련 변수는 agent.py와 공유
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent import ClaudeCliLLM, VllmLLM, parse_json_block

MAX_LENSES = 4
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}


class SurveyState(TypedDict):
    topic: str
    context: dict           # 워크스페이스 맥락 — 이게 없으면 주제가 표류한다
    lenses: list[dict]      # [{name, why, arxiv_query, web_query, terms}]
    raw: list[dict]         # [{lens, origin, source, title, snippet}]
    findings: list[dict]    # [{claim, source, lens, why}]
    gaps: list[str]         # 조사로도 안 밝혀진 것 — 정의 단계로 넘길 질문
    web_budget: int
    out_path: str


# ---------------------------------------------------------------- 프롬프트

SCOPE_SYSTEM = (
    "너는 연구 조사 설계자다. 막연한 연구 의향을 받아 '서로 다른 각도'의 조사 렌즈를 최대 4개 만든다. "
    "**먼저 도메인을 확정하라.** 주어진 워크스페이스 맥락(디렉터리 목록·백로그·관련 문서)을 보고 이 "
    "요청이 어느 분야의 이야기인지 판단한 뒤 검색어를 만들어라. 요청 문구만 보고 일반적인 분야로 "
    "넘어가지 마라 — 도메인을 놓치면 조사 전체가 엉뚱한 곳을 판다. 검색어에는 도메인 용어를 반드시 "
    "포함하라. terms에는 워크스페이스 문서에서 실제로 쓰일 한국어 용어를 넣어라. "
    "렌즈는 겹치면 안 된다 — 같은 걸 네 번 찾으면 조사가 아니라 반복이다. 다음 축을 기본으로 삼되 "
    "주제에 맞게 조정하라: (1) 선행연구·방법론(이미 뭐가 됐나), (2) 평가·베이스라인(뭐랑 비교하고 "
    "뭘 재나), (3) 데이터·재현성(어떤 데이터로, 구할 수 있나), (4) 실패사례·한계(왜 안 됐나). "
    "arxiv_query는 arXiv API용 영문 검색어(따옴표로 구절 묶기 가능), web_query는 웹검색용 자연어 질의, "
    "terms는 로컬 문서에서 찾을 핵심어 3~6개(한글·영문 섞어도 됨)다. "
    '반드시 JSON 하나만 출력하라: {"lenses": [{"name": "...", "why": "이 각도가 왜 필요한가", '
    '"arxiv_query": "...", "web_query": "...", "terms": ["..."]}]}'
)

DISTILL_SYSTEM = (
    "너는 조사 결과 정제자다. 수집된 자료에서 '문제 정의에 실제로 쓸 수 있는' 근거만 뽑는다. "
    "규칙: (1) 모든 claim은 주어진 자료에 근거해야 한다. 자료에 없는 내용을 배경지식으로 채우지 마라 "
    "— 그건 근거가 아니라 추측이다. (2) source에는 반드시 주어진 자료의 출처 문자열을 그대로 옮겨라. "
    "(3) 정의에 쓸모없는 일반론('이 분야는 활발히 연구된다')은 버려라. 베이스라인·평가지표·데이터·"
    "구체적 수치·알려진 실패 원인처럼 **정의를 좁히는 데 쓰이는 것**만 남겨라. "
    "(4) 자료가 부실하면 findings를 적게 내라. 개수를 채우려 하지 마라. "
    '반드시 JSON 하나만 출력하라: {"findings": [{"claim": "...", "source": "...", '
    '"why": "이 근거가 문제 정의를 어떻게 좁히는가"}]}'
)

GAPS_SYSTEM = (
    "너는 조사 감사자다. 모인 근거를 보고 '연구 문제를 정의하려면 알아야 하는데 아직 모르는 것'을 "
    "최대 5개 짚어라. 근거가 답해준 것은 적지 마라 — 빈 곳만 적는다. "
    "특히 이것들이 근거로 정해졌는지 확인하라: 비교할 베이스라인, 평가 지표와 임계값, 쓸 데이터의 "
    "실재 여부, 선행연구가 이미 답한 질문인지 여부. "
    '반드시 JSON 하나만 출력하라: {"gaps": ["..."]}'
)


class MockSurveyLLM:
    """LLM 없이 그래프 역학을 검증하는 스크립트드 두뇌."""

    def complete(self, system: str, user: str) -> str:
        if "설계자" in system:
            return json.dumps(
                {
                    "lenses": [
                        {
                            "name": "선행연구·방법론",
                            "why": "이미 된 것을 다시 하지 않기 위해",
                            "arxiv_query": 'all:"graph neural network" AND all:soccer',
                            "web_query": "GNN soccer lineup recommendation",
                            "terms": ["GNN", "lineup", "라인업"],
                        },
                        {
                            "name": "평가·베이스라인",
                            "why": "비교 대상 없이는 결론이 안 남는다",
                            "arxiv_query": 'all:"win probability" AND all:football',
                            "web_query": "soccer win probability baseline RPS",
                            "terms": ["RPS", "베이스라인", "baseline"],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "정제자" in system:
            return json.dumps(
                {
                    "findings": [
                        {
                            "claim": "승률 예측의 통상 베이스라인은 RPS 0.19~0.21 구간이다",
                            "source": "mock://local/v2_design_survey.md",
                            "why": "임계값을 근거로 정할 수 있게 한다",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps({"gaps": ["K리그 라인업 데이터의 실재 여부가 미확인"]}, ensure_ascii=False)


LLM: MockSurveyLLM | VllmLLM | ClaudeCliLLM  # main()에서 주입


# ---------------------------------------------------------------- 수집기 (코드)


def search_arxiv(query: str, limit: int = 5) -> list[dict]:
    """arXiv API — 무료·무인증. http는 차단되고 https만 통한다."""
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": limit, "sortBy": "relevance"}
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            root = ET.fromstring(resp.read().decode())
    except Exception as exc:  # 조사 실패가 파이프라인을 죽이면 안 됨
        print(f"  [arxiv] 실패: {exc}")
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in root.findall("a:entry", ns):
        entries.append(
            {
                "origin": "arxiv",
                "source": (entry.findtext("a:id", "", ns) or "").strip(),
                "title": " ".join((entry.findtext("a:title", "", ns) or "").split()),
                "snippet": " ".join((entry.findtext("a:summary", "", ns) or "").split())[:700],
            }
        )
    return entries


def search_local(terms: list[str], base: str, limit: int = 4) -> list[dict]:
    """워크스페이스의 기존 서베이·메모를 근거로 재활용 — 공짜이고 맥락이 정확하다."""
    scored: list[tuple[int, str, str]] = []
    lowered = [t.lower() for t in terms if t]
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            low = text.lower()
            score = sum(1 for t in lowered if t in low)
            if score >= 2:
                scored.append((score, path, text))
    scored.sort(key=lambda item: -item[0])

    hits = []
    for _, path, text in scored[:limit]:
        low = text.lower()
        pos = min((low.find(t) for t in lowered if low.find(t) >= 0), default=0)
        hits.append(
            {
                "origin": "local",
                "source": path,
                "title": os.path.basename(path),
                "snippet": " ".join(text[max(0, pos - 200) : pos + 700].split()),
            }
        )
    return hits


def search_web(query: str) -> list[dict]:
    """codex CLI 웹검색. 호출당 수십초~수분, 토큰도 크므로 렌즈 수를 제한해 쓴다."""
    prompt = (
        f"Search the web for: {query}\n"
        "Reply ONLY with up to 5 lines, each 'TITLE | URL or DOI | one-sentence finding'. "
        "No preamble, no commentary."
    )
    try:
        proc = subprocess.run(
            [
                os.environ.get("CODEX_BIN", "codex"), "--search", "exec",
                "--skip-git-repo-check", "-s", "read-only", prompt,
            ],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=int(os.environ.get("CODEX_TIMEOUT", "240")),
        )
    except Exception as exc:
        print(f"  [web] 실패: {exc}")
        return []
    # codex는 진행로그를 함께 뱉는다. 최종 답변은 마지막 'tokens used' 블록 뒤에 온다.
    tail = proc.stdout.split("tokens used")[-1]
    entries = []
    for line in tail.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        # 출처는 URL/DOI여야 한다. 검색어를 출처로 쓰면 인용 검증이 무의미해진다.
        url = next((p for p in parts[1:] if "http" in p or p.startswith("10.")), "")
        if not url:
            continue
        entries.append(
            {"origin": "web", "source": url, "title": parts[0], "snippet": " | ".join(parts[2:]) or line.strip()}
        )
    return entries[:5]


# ---------------------------------------------------------------- 노드


def intake(state: SurveyState) -> dict:
    """도메인 확정용 워크스페이스 맥락 수집 (LLM 없이).

    첫 실동에서 이게 없어 'GNN 추천 라인업'을 일반 추천시스템(LightGCN·번들추천)으로 읽었고,
    조사 29건 전부가 축구와 무관했다. 요청 문구만으로는 도메인이 결정되지 않는다.
    """
    base = os.environ.get("SURVEY_BASE", "/workspace")
    context: dict = {"base": base}
    try:
        context["base_entries"] = sorted(os.listdir(base))[:40]
    except OSError as exc:
        context["base_entries"] = f"읽기 실패: {exc}"

    backlog = os.environ.get("BACKLOG_PATH", "/workspace/RESEARCH_BACKLOG.md")
    if os.path.exists(backlog):
        context["backlog_excerpt"] = open(backlog, encoding="utf-8").read()[:2000]

    # 요청어 자체로 로컬 문서를 훑어 도메인 힌트를 준다
    hint_terms = [t for t in state["topic"].split() if len(t) >= 2][:6]
    context["related_docs"] = [h["source"] for h in search_local(hint_terms, base, limit=6)]

    print(f"[intake] base={base}, 관련 문서 {len(context['related_docs'])}건")
    for path in context["related_docs"][:4]:
        print(f"  · {path}")
    return {"context": context}


def scope(state: SurveyState) -> dict:
    payload = json.dumps(
        {"topic": state["topic"], "workspace_context": state["context"]}, ensure_ascii=False
    )
    parsed = parse_json_block(LLM.complete(SCOPE_SYSTEM, payload))
    lenses = parsed.get("lenses", [])[:MAX_LENSES]
    print(f"[scope] 조사 렌즈 {len(lenses)}개")
    for lens in lenses:
        print(f"  · {lens.get('name')} — {lens.get('why')}")
    return {"lenses": lenses}


def gather(state: SurveyState) -> dict:
    """렌즈별로 arXiv·로컬·웹을 병렬 수집 (LLM 없이)."""
    base = os.environ.get("SURVEY_BASE", "/workspace")
    budget = state["web_budget"]

    def one(item: tuple[int, dict]) -> list[dict]:
        index, lens = item
        found = search_arxiv(lens.get("arxiv_query", ""), limit=5)
        found += search_local(lens.get("terms", []), base)
        if index < budget:
            found += search_web(lens.get("web_query", ""))
        for entry in found:
            entry["lens"] = lens.get("name", "")
        origins = {o: sum(1 for e in found if e["origin"] == o) for o in ("arxiv", "local", "web")}
        print(f"[gather] {lens.get('name')}: arxiv {origins['arxiv']} · 로컬 {origins['local']} · 웹 {origins['web']}")
        return found

    with ThreadPoolExecutor(max_workers=max(1, len(state["lenses"]))) as pool:
        results = list(pool.map(one, enumerate(state["lenses"])))
    raw = [entry for group in results for entry in group]
    print(f"[gather] 총 {len(raw)}건 수집")
    return {"raw": raw}


def distill(state: SurveyState) -> dict:
    """렌즈별 수집물을 '출처가 붙은 근거'로 정제. 출처 없는 주장은 코드가 버린다."""
    by_lens: dict[str, list[dict]] = {}
    for entry in state["raw"]:
        by_lens.setdefault(entry.get("lens", ""), []).append(entry)

    def one(name: str) -> list[dict]:
        payload = json.dumps({"lens": name, "materials": by_lens[name]}, ensure_ascii=False)
        try:
            parsed = parse_json_block(LLM.complete(DISTILL_SYSTEM, payload))
        except Exception as exc:
            print(f"[distill] {name} 실패: {exc}")
            return []
        known = {e["source"] for e in by_lens[name]}
        kept = []
        for finding in parsed.get("findings", []):
            source = str(finding.get("source", "")).strip()
            # 출처가 수집물에 없으면 지어낸 것 — LLM 판단이 아니라 코드가 거른다
            if source and any(source in k or k in source for k in known):
                kept.append({**finding, "source": source, "lens": name})
            else:
                print(f"[distill] {name}: 출처 불명으로 폐기 — {str(finding.get('claim'))[:60]}")
        return kept

    names = [n for n in by_lens if by_lens[n]]
    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        results = list(pool.map(one, names))
    findings = [f for group in results for f in group]
    print(f"[distill] 근거 {len(findings)}건 확보")
    return {"findings": findings}


def gaps(state: SurveyState) -> dict:
    payload = json.dumps(
        {"topic": state["topic"], "findings": state["findings"]}, ensure_ascii=False
    )
    try:
        parsed = parse_json_block(LLM.complete(GAPS_SYSTEM, payload))
    except Exception as exc:
        print(f"[gaps] 실패: {exc}")
        return {"gaps": []}
    found = parsed.get("gaps", [])[:5]
    print(f"[gaps] 조사로 못 채운 것 {len(found)}건")
    for item in found:
        print(f"  ? {item}")
    return {"gaps": found}


def finalize(state: SurveyState) -> dict:
    by_lens: dict[str, list[dict]] = {}
    for finding in state["findings"]:
        by_lens.setdefault(finding.get("lens", "기타"), []).append(finding)

    lines = [
        f"# 조사 결과: {state['topic']}",
        "",
        f"> surveyor.py · 렌즈 {len(state['lenses'])}개 · 수집 {len(state['raw'])}건 → 근거 {len(state['findings'])}건",
        "",
    ]
    for lens in state["lenses"]:
        name = lens.get("name", "")
        lines += [f"## {name}", f"*{lens.get('why', '')}*", ""]
        items = by_lens.get(name, [])
        if not items:
            lines += ["- (근거 없음 — 이 각도는 조사가 비었다)", ""]
            continue
        for finding in items:
            lines += [
                f"- **{finding.get('claim')}**",
                f"  - 출처: `{finding.get('source')}`",
                f"  - 정의에 미치는 영향: {finding.get('why', '')}",
            ]
        lines.append("")
    lines += [
        "## 조사로도 못 채운 것 (정의 단계에서 결정 필요)",
        "\n".join(f"- {g}" for g in state["gaps"]) or "- (없음)",
        "",
    ]
    document = "\n".join(lines)

    with open(state["out_path"], "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"\n[finalize] {state['out_path']} 작성 (근거 {len(state['findings'])}건)\n")
    print(document)
    return {}


# ---------------------------------------------------------------- 그래프


def build_app():
    graph = StateGraph(SurveyState)
    for name, node in (
        ("intake", intake), ("scope", scope), ("gather", gather), ("distill", distill),
        ("gaps", gaps), ("finalize", finalize),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "scope")
    graph.add_edge("scope", "gather")
    graph.add_edge("gather", "distill")
    graph.add_edge("distill", "gaps")
    graph.add_edge("gaps", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> int:
    global LLM
    parser = argparse.ArgumentParser(description="LangGraph 조사 에이전트")
    parser.add_argument("topic", help="조사할 연구 주제 (막연해도 됨)")
    parser.add_argument("--web", type=int, default=2, help="웹검색을 돌릴 렌즈 수 (기본 2, 0=끔). 느리고 비싸다")
    parser.add_argument("--out", default="evidence.md", help="근거 목록 출력 경로 (기본 evidence.md)")
    parser.add_argument("--mock", action="store_true", help="LLM 없이 그래프 스모크 테스트")
    parser.add_argument("--vllm", action="store_true", help="클로드 CLI 대신 로컬 vLLM 두뇌 사용")
    args = parser.parse_args()

    LLM = MockSurveyLLM() if args.mock else (VllmLLM() if args.vllm else ClaudeCliLLM())
    app = build_app()
    final = app.invoke(
        {
            "topic": args.topic,
            "context": {},
            "lenses": [],
            "raw": [],
            "findings": [],
            "gaps": [],
            "web_budget": max(0, args.web),
            "out_path": args.out,
        }
    )
    return 0 if final["findings"] else 1


if __name__ == "__main__":
    sys.exit(main())
