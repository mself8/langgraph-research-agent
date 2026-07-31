# LangGraph 연구 에이전트

`/research`(클로드 코드 활용판)와 같은 기능 — **계획→실행→평가→재계획 루프** — 을
LangGraph로 직접 구현한 버전. "에이전트 도구 활용 → 프레임워크 직접 개발"로
발전시킨 포트폴리오 프로젝트다. LLM 두뇌는 교체 가능하며 **어느 쪽도 API 키가
필요 없다**:

- **기본 = Claude Code CLI** (`claude -p`, 구독으로 처리) — 준비물 0
- 옵션 = 로컬 vLLM (`--vllm`, OpenAI 호환 서버) — 자가호스팅 검증용

두뇌를 어댑터(`complete(system, user)`) 하나로 추상화해서 CLI↔vLLM↔(원하면 API)를
플래그로 스왑한다 — "LLM 백엔드 독립적인 하네스"가 이 프로젝트의 설계 포인트.

## 그래프 설계

```
START → selector → planner → executor → evaluator ─┬─ done/fail → finish → slack_report → END
                     ▲                             │
                     └──────── retry (최대 3회) ────┘

selector     = 과제 인자가 없으면 RESEARCH_BACKLOG.md 최상위 항목 자동 선정
slack_report = SLACK_WEBHOOK_URL 설정 시 결과를 슬랙 DM으로 발송 (미설정 시 무시)
```

| 노드 | 역할 | 두뇌 |
|---|---|---|
| `planner` | 과제+실행 기록을 보고 다음 셸 명령 1개와 성공 기준을 JSON으로 제안 | LLM |
| `executor` | 명령을 subprocess로 실행. 안전장치: 금지 패턴 거부(rm -rf/sudo/curl/디렉터리 탈출 등), 작업 디렉터리 제한, 타임아웃 | 코드 |
| `evaluator` | 출력이 성공 기준을 충족하는지 판정 `{done\|retry\|fail}` | LLM |
| `finish` | 실행 기록을 결과 리포트로 정리 | 코드 |

**왜 LangGraph인가**: 이 루프의 핵심은 "평가 결과에 따라 흐름이 갈라지고
(조건부 엣지), 상태(history)를 들고 되돌아간다(루프)"는 것 — 고정 파이프라인
(n8n류)으로는 표현할 수 없고, LangGraph의 `StateGraph` + `add_conditional_edges`가
정확히 이 모양을 위한 도구다.

## 실행

```bash
# 1) 실제 실행 — 기본: 클로드 CLI 두뇌 (claude 로그인만 돼 있으면 준비물 0)
.venv/bin/python agent.py "이 폴더의 csv 파일 각각의 데이터 행 수를 세서 보고하라"
.venv/bin/python agent.py          # 과제 생략 → 백로그 최상위 항목 자동 선정
# ✅ 2026-07-31 실동 검증 2회: 1회 시도 성공 / 경로 착오→retry 자가복구 후 성공+슬랙 보고

# 2) 스모크 테스트 (LLM 불필요 — 그래프 역학만 검증)
.venv/bin/python agent.py --mock "스모크 테스트"

# 3) 로컬 vLLM 두뇌 (자가호스팅 검증용)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000   # 별도 터미널
.venv/bin/python agent.py --vllm "sandbox/ 안에 1~100 합을 계산하는 스크립트를 만들어 실행하라"
```

환경변수: `CLAUDE_BIN`/`CLAUDE_TIMEOUT` · `VLLM_BASE_URL`(기본 localhost:8000/v1) ·
`VLLM_MODEL` · `AGENT_WORKDIR`(명령 실행 디렉터리, 기본 `./sandbox`) · `AGENT_CMD_TIMEOUT`(초).

## /research 커맨드와의 관계

| | `/research` (클로드 코드) | 이 프로젝트 (LangGraph) |
|---|---|---|
| 루프 주체 | Claude Code 하네스 | **내가 짠 StateGraph** |
| LLM | 구독 (Claude) | 교체형: 클로드 CLI(기본)/로컬 vLLM |
| 용도 | 실전 연구 진행 | 구조 학습 + 포트폴리오 |

핵심 구분: `/research`는 **남의 하네스에 절차서(CLAUDE.md·커맨드)를 얹는 것**,
이 프로젝트는 **루프·도구·안전장치·종료조건을 코드로 직접 소유하는 것**.

## 로드맵

1. ~~`selector` 노드 — 백로그 자동 선정~~ ✅ (07-31)
2. ~~Slack webhook 보고 노드~~ ✅ (07-31)
3. 장시간 실험 지원 — 학습을 백그라운드 발사 + 주기 폴링 노드 (연구 주력화의 관문)
4. 다단계 계획 — plan을 명령 목록으로 확장
5. 체크포인터로 중단·재개 (LangGraph `checkpointer`)

### 실전에서 배운 하네스 설계 교훈
- evaluator가 "계획의 성공기준"만 보면 과제 일부 달성을 done으로 오판한다 →
  원래 과제를 판정 입력에 포함하고 "부분 달성=retry"를 명시 (07-31 수정)
