import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from acceptance import check_artifacts, validate_contract


class ScriptedPlanner:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.inputs = []

    def complete(self, system, user):
        if '계획자' not in system:
            raise AssertionError('Structured acceptance must not call the LLM evaluator')
        self.inputs.append(json.loads(user))
        return json.dumps({'command': next(self.commands), 'success_criteria': '완료라고 출력', 'rationale': 'fixture'})


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def check(self, spec):
        return check_artifacts(validate_contract([spec]), str(self.root))

    def test_missing_and_empty(self):
        spec = dict(id='file', kind='nonempty_file', path='out.txt')
        self.assertFalse(self.check(spec)['passed'])
        (self.root/'out.txt').touch()
        self.assertFalse(self.check(spec)['passed'])

    def test_csv_checks_rows_columns_and_shape(self):
        spec = dict(id='table', kind='csv_table', path='out.csv', columns=['season', 'count'], exact_rows=5)
        for text in ('season\n2021\n', 'season,count\n2021,1\n', 'season,count\n2021,1,extra\n'):
            (self.root/'out.csv').write_text(text)
            self.assertFalse(self.check(spec)['passed'])
        (self.root/'out.csv').write_text('season,count\n'+''.join(f'{year},1\n' for year in range(2021,2026)))
        self.assertTrue(self.check(spec)['passed'])

    def test_json_and_text(self):
        spec = dict(id='json', kind='json_object', path='out.json', equals={'count': 5})
        for value in ('broken', '[]', '{"count":true}', '{"count":4}'):
            (self.root/'out.json').write_text(value)
            self.assertFalse(self.check(spec)['passed'])
        (self.root/'out.json').write_text('{"count":5}')
        self.assertTrue(self.check(spec)['passed'])
        (self.root/'out.md').write_text('## 결과\n## 한계')
        self.assertTrue(self.check(dict(id='text',kind='text_contains',path='out.md',contains=['## 결과','## 한계']))['passed'])

    def test_config_rejects_escape_unknown_keys_and_duplicates(self):
        base = dict(id='x', kind='nonempty_file', path='x')
        for value in ([], [base,base], [{**base,'path':'../x'}], [{**base,'path':'/tmp/x'}], [{**base,'typo':True}]):
            with self.assertRaises(ValueError): validate_contract(value)

    def test_symlink_escape_and_large_file(self):
        outside = self.root.parent / (self.root.name+'-outside')
        outside.write_text('private')
        self.addCleanup(outside.unlink)
        (self.root/'link').symlink_to(outside)
        self.assertFalse(self.check(dict(id='link',kind='nonempty_file',path='link'))['passed'])
        (self.root/'large').write_bytes(b'x'*2_000_001)
        self.assertFalse(self.check(dict(id='large',kind='nonempty_file',path='large'))['passed'])

    def run_graph(self, commands):
        planner = ScriptedPlanner(commands)
        criteria = validate_contract([dict(id='result',kind='text_contains',path='result.md',contains=['완료'])])
        with patch.dict(os.environ, {'AGENT_WORKDIR':str(self.root)}), patch.object(agent, 'LLM', planner, create=True):
            final = agent.build_app().invoke(dict(task='결과 파일 만들기',plan=None,history=[],result='',verdict='',iteration=0,acceptance=criteria,notify_slack=False))
        return final, planner

    def test_false_completion_retries_with_feedback(self):
        final, planner = self.run_graph(["echo done", "printf '완료' > result.md"])
        self.assertEqual(final['verdict'],'done')
        self.assertEqual(final['iteration'],2)
        self.assertFalse(planner.inputs[1]['history'][0]['validation']['passed'])
        self.assertEqual(planner.inputs[0]['acceptance'], planner.inputs[1]['acceptance'])

    def test_failure_stops_at_limit(self):
        final, _ = self.run_graph(['echo done']*3)
        self.assertEqual(final['verdict'],'fail')
        self.assertEqual(final['iteration'],3)

    def test_command_failure_cannot_use_leftover_output(self):
        (self.root/'result.md').write_text('완료')
        final, _ = self.run_graph(['exit 1']*3)
        self.assertEqual(final['verdict'],'fail')
        self.assertTrue(final['history'][0]['validation']['passed'])

if __name__ == '__main__': unittest.main()
