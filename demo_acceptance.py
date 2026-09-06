"""Reproducible scripted-planner demo: actual graph/executor/checks, no live LLM."""
import html
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import agent
from acceptance import load_contract


class DemoPlanner:
    def complete(self, system, user):
        payload = json.loads(user)
        if '계획자' not in system:
            raise AssertionError('완료 판정은 결과물 검사로 수행해야 합니다.')
        if not payload['history']:
            command = "printf 'season,matches\n2021,12\n' > season_counts.csv"
        else:
            command = "printf 'season,matches\n2021,12\n2022,13\n2023,14\n2024,15\n2025,16\n' > season_counts.csv; printf '## 결과\n예제 데이터 5개 시즌\n## 한계\n실제 K리그 데이터가 아닌 데모입니다.\n' > summary.md"
        return json.dumps({'command':command,'success_criteria':'완료','rationale':'첫 시도 부분 완료, 다음 시도 수정'}, ensure_ascii=False)


def main():
    root = Path(__file__).parent
    checks = load_contract(str(root/'examples/acceptance.json'))
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AGENT_WORKDIR':tmp}), patch.object(agent,'LLM',DemoPlanner(),create=True):
        final = agent.build_app().invoke(dict(task='5개 시즌 집계 CSV와 결과·한계 요약을 작성하세요.',plan=None,history=[],result='',verdict='',iteration=0,acceptance=checks,notify_slack=False))
    final['test_mode'] = 'scripted planner; real graph, executor and artifact validation; synthetic data'
    out = root/'docs/acceptance_demo.json'
    out.write_text(json.dumps(final,ensure_ascii=False,indent=2))
    cards = []
    for run in final['history']:
        rows = ''.join(f'<tr><td>{html.escape(x["id"])}</td><td>{"통과" if x["passed"] else "미충족"}</td><td>{html.escape(x["detail"])}</td></tr>' for x in run['validation']['checks'])
        cards.append(f'<section><h2>{run["iteration"]}차 실행 · {run["verdict"]}</h2><table><tr><th>조건</th><th>검사</th><th>근거</th></tr>{rows}</table><details><summary>실행 명령 보기</summary><pre>{html.escape(run["plan"]["command"])}</pre></details></section>')
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>에이전트 완료 조건 검증</title><style>body{font:17px/1.7 system-ui;background:#f4f6fa;color:#17243b;max-width:960px;margin:48px auto;padding:0 24px}section{background:white;border-radius:16px;padding:24px;margin:24px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;font-size:13px}small{color:#526079}</style><h1>파일이 완성됐을 때 작업을 완료합니다</h1><p>요청: 5개 시즌 집계 CSV + 결과와 한계를 포함한 요약</p><p>첫 시도에서 빠진 조건을 확인하고, 다음 계획에 검사 결과를 전달합니다.</p>'''+''.join(cards)+'''<small>재현용 시나리오: 계획은 고정된 예제이며, 그래프·파일 작성·검사는 실제 실행했습니다. LLM 성능 평가나 실제 K리그 분석 결과가 아닙니다. 이 검사는 행 수·컬럼·문구 등 지정 조건만 확인하며 내용의 사실성은 보증하지 않습니다.</small></html>'''
    (root/'docs/acceptance_demo.html').write_text(document)
    print('Report:',out)

if __name__ == '__main__': main()
