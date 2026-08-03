# LangGraph 연구 에이전트

두 개의 그래프로 구성된다:

| | 파일 | 루프 | 종료 조건 |
|---|---|---|---|
| **문제 정의** | `definer.py` | 초안→근거검증→다중해석→발산해소 | 해석자들의 해석이 일치 |
| **문제 해결** | `agent.py` | 계획→실행→평가→재계획 | 과제 완수 판정 |

`definer.py`가 만든 스펙을 `agent.py --spec`으로 넘기면 정의→해결이 이어진다.

## 문제 해결 에이전트 (`agent.py`)

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

## 문제 정의 에이전트 (`definer.py`)

모호한 요청을 실행 에이전트가 오해 없이 풀 수 있는 스펙으로 수렴시킨다. 전제는
"요즘 AI는 문제가 정확히 정의되면 풀 수 있다 — 병목은 정의다".

**핵심 설계 — "잘 정의됨"을 측정 가능하게:** 대부분의 문제정의 도구는 LLM에게 스펙
템플릿을 채우게 시키고 끝낸다. 실패 신호가 없어서 모호한 스펙도 그럴듯하게 통과한다.
대신 조작적 정의를 둔다:

> 스펙이 잘 정의됐다 ⟺ **그 스펙만 읽은 독립 해석자들이 "무엇을 만들지"와
> "합격 판정"에 같은 답을 내놓는다.**

특히 합격 판정 일치가 중요하다 — 두 평가자가 같은 출력물을 놓고 통과/불통과가
갈리면 그 스펙은 깨진 것이다. 이 **발산 테스트**가 루프의 종료 조건이다.

```
START → intake → drafter → grounder → interpreters ×3 → divergence
        (코드)   (LLM)     (코드)      (LLM·병렬)        (LLM)
                   ▲                                       │
                   ├── resolvable 발산 → 재작성 (최대 3라운드) ┤
                   └── user 발산 → resolve(질문/가정 명시) ────┤
                                          수렴 → finalize → END┘
```

| 노드 | 역할 | 두뇌 |
|---|---|---|
| `intake` | 근거 검증용 컨텍스트 수집 (기준 디렉터리 목록·백로그) | 코드 |
| `drafter` | 목표/입력/산출물/검증방법/범위밖/가정 스펙 작성 | LLM |
| `grounder` | **스펙이 전제한 경로가 실재하는지 확인** + 형식 검사 | 코드 |
| `interpreters` | 스펙만 보고 "무엇을 만들고 어떻게 합격 판정하나" 독립 답변 | LLM ×3 (스레드 병렬) |
| `divergence` | 실질적 발산만 추출·분류 → `converged\|revise\|ask` | LLM |
| `resolve` | `--ask`면 사용자에게 질문, 헤드리스면 가정으로 명시 | 코드 |
| `finalize` | `spec.md` 작성 (검증 명령 포함) | 코드 |

**`grounder`가 코드인 이유**: 실전에서 가장 흔한 실패는 스펙이 모호한 게 아니라
**존재하지 않는 데이터·경로를 전제**하는 것이고, 이건 LLM 추론이 아니라
`os.path.exists` 한 줄로 잡힌다. `agent.py`의 `DENY_PATTERNS`와 같은 성격의 코드측 검사.

**한계 (중요)**: 이 설계가 잡는 것은 *스펙 내부의 모호성*이지 *그 문제를 푸는 게
옳은지*가 아니다. 발산 테스트를 통과한 아름다운 스펙이 엉뚱한 문제일 수 있다.
목표의 타당성은 사용자나 데이터만 공급할 수 있고, 에이전트가 이를 보증하는 척해선 안 된다.

```bash
# 헤드리스 (기본) — 사용자 판단이 필요한 지점은 가정으로 명시하고 진행
.venv/bin/python definer.py "승률 예측 좀 개선해줘"

.venv/bin/python definer.py --ask "..."       # 발산 지점을 직접 질문
.venv/bin/python definer.py --generic "..."   # 도메인 무관 — 경로 실재 검증 없이 형식 검사만
.venv/bin/python definer.py --mock "..."      # LLM 없이 그래프 검증 (발산→재작성→수렴 2라운드)
.venv/bin/python definer.py --interpreters 2 "..."   # 해석자 수 조정 (기본 3, 최소 2)

# 정의 → 해결 연결
.venv/bin/python definer.py "..." --out spec.md && .venv/bin/python agent.py --spec spec.md
```

환경변수: `DEFINER_BASE`(근거 검증 기준 디렉터리, 기본 cwd) · `BACKLOG_PATH` ·
LLM 관련 변수는 `agent.py`와 공유.

비용 주의: 라운드당 클로드 CLI 호출 = 초안 1 + 해석자 N + 비교 1. 해석자는 스레드로
병렬이라 체감은 라운드당 3콜 수준이지만, 느리면 `--interpreters 2`로 줄이면 된다.

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
2-1. ~~`definer.py` — 문제 정의 그래프~~ ✅ (08-03)
2-2. **결정론적 evaluator** — `--spec`으로 받은 스펙의 `verification_command`를 실제로
   실행해 그 종료코드를 판정 근거로 삼는다. 지금은 evaluator가 LLM 의견으로 판정하는데,
   이러면 아래 "부분 달성을 done으로 오판" 문제가 구조적으로 사라진다.
3. 장시간 실험 지원 — 학습을 백그라운드 발사 + 주기 폴링 노드 (연구 주력화의 관문)
4. 다단계 계획 — plan을 명령 목록으로 확장
5. 체크포인터로 중단·재개 (LangGraph `checkpointer`)

### 실전에서 배운 하네스 설계 교훈
- evaluator가 "계획의 성공기준"만 보면 과제 일부 달성을 done으로 오판한다 →
  원래 과제를 판정 입력에 포함하고 "부분 달성=retry"를 명시 (07-31 수정)
- **해석 일치 ≠ 정의 완료.** 첫 실동에서 해석자 3명이 같은 답을 냈지만, 그중 하나가
  "'승률 예측 관련'의 판정 기준이 스펙에 없다"고 따로 적었는데도 비교자는 converged로
  판정했다 — 해석자들이 *우연히 같은 임의 선택*을 한 것뿐인데 스펙이 그걸 지정했다고
  오인한 것. → 해석자 출력에 `undefined`(내가 임의로 정해야 했던 것) 필드를 추가하고,
  답이 일치해도 undefined 항목이 합격 판정을 가를 수 있으면 발산으로 취급 (08-03 수정)
- **근거 검증이 자문에 그치면 안 된다.** grounder가 drafter 뒤에 있어서, 1라운드에
  수렴하면 검증 결과가 스펙에 한 번도 반영되지 않았다(첫 실동에서 "grounding이 비어 있어
  확인 못 했다"는 가정이 그대로 남음). → grounder 실패 시 해석 비교로 넘어가지 않고
  drafter로 되돌리는 조건부 엣지 추가 (08-03 수정)
