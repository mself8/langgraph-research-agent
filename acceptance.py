"""User-owned artifact acceptance checks. No shell commands or model-generated checks.

Passing means the configured artifact conditions passed, not semantic correctness.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

MAX_BYTES = 2_000_000
KINDS = {'nonempty_file', 'text_contains', 'csv_table', 'json_object'}


def validate_contract(value: object) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError('완료 조건은 1~30개의 검사 목록이어야 합니다.')
    result = []
    ids = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('검사는 JSON 객체여야 합니다.')
        check = dict(item)
        kind = check.get('kind')
        ident = check.get('id')
        path = check.get('path')
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError('검사 id는 비어 있지 않은 고유 문자열이어야 합니다.')
        ids.add(ident)
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f'{ident}: 지원하지 않는 검사 종류')
        if not isinstance(path, str) or not path or Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError(f'{ident}: 작업 폴더 기준 상대 경로가 필요합니다.')
        if kind == 'text_contains':
            terms = check.get('contains')
            if not isinstance(terms, list) or not terms or not all(isinstance(x, str) and x for x in terms):
                raise ValueError(f'{ident}: contains에 확인할 문자열 목록이 필요합니다.')
        if kind == 'csv_table':
            columns = check.get('columns')
            if not isinstance(columns, list) or not columns or not all(isinstance(x, str) and x for x in columns):
                raise ValueError(f'{ident}: columns에 필수 컬럼 목록이 필요합니다.')
            if len(columns) != len(set(columns)):
                raise ValueError(f'{ident}: columns가 중복됐습니다.')
            for key in ('min_rows', 'exact_rows'):
                if key in check and (type(check[key]) is not int or check[key] < 0):
                    raise ValueError(f'{ident}: {key}는 0 이상의 정수여야 합니다.')
            if 'min_rows' in check and 'exact_rows' in check:
                raise ValueError(f'{ident}: min_rows와 exact_rows 중 하나만 지정하세요.')
        if kind == 'json_object':
            expected = check.get('equals')
            if not isinstance(expected, dict) or not expected:
                raise ValueError(f'{ident}: equals에 필수 최상위 키와 기대값이 필요합니다.')
        allowed = {'id', 'kind', 'path'} | {
            'nonempty_file': set(), 'text_contains': {'contains'},
            'csv_table': {'columns', 'min_rows', 'exact_rows'}, 'json_object': {'equals'},
        }[kind]
        if set(check) - allowed:
            raise ValueError(f'{ident}: 알 수 없는 설정 {sorted(set(check) - allowed)}')
        result.append(check)
    return result


def load_contract(path: str) -> list[dict]:
    return validate_contract(json.loads(Path(path).read_text(encoding='utf-8')))


def check_artifacts(checks: list[dict], workdir: str) -> dict:
    """Read-only checks; reject missing files, symlink escapes and oversized files."""
    root = Path(workdir).resolve()
    results = []
    for check in checks:
        item = {'id': check['id'], 'path': check['path'], 'kind': check['kind'], 'passed': False}
        try:
            path = (root / check['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('작업 폴더 밖으로 연결된 경로입니다.')
            if not path.is_file():
                raise ValueError('결과 파일이 없습니다.')
            size = path.stat().st_size
            if size > MAX_BYTES:
                raise ValueError(f'검사 크기 제한({MAX_BYTES}바이트)을 초과했습니다.')
            if size == 0:
                raise ValueError('파일이 비어 있습니다.')
            # Bound the actual read, too, in case a file grew after stat().
            with path.open('rb') as handle:
                raw = handle.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError('검사 중 파일 크기 제한을 초과했습니다.')
            if check['kind'] == 'nonempty_file':
                detail = f'{len(raw)}바이트 파일 확인'
            else:
                text = raw.decode('utf-8-sig')
                if check['kind'] == 'text_contains':
                    missing = [x for x in check['contains'] if x not in text]
                    if missing:
                        raise ValueError(f'필수 문구 누락: {missing}')
                    detail = f'필수 문구 {len(check["contains"])}개 확인'
                elif check['kind'] == 'csv_table':
                    import io
                    rows = list(csv.reader(io.StringIO(text), strict=True))
                    if not rows:
                        raise ValueError('CSV 헤더가 없습니다.')
                    header, data = rows[0], rows[1:]
                    if len(header) != len(set(header)):
                        raise ValueError('CSV 헤더가 중복됐습니다.')
                    missing = [x for x in check['columns'] if x not in header]
                    if missing:
                        raise ValueError(f'필수 컬럼 누락: {missing}')
                    if any(len(row) != len(header) for row in data):
                        raise ValueError('헤더와 데이터의 컬럼 수가 다릅니다.')
                    count = len(data)
                    if 'exact_rows' in check and count != check['exact_rows']:
                        raise ValueError(f'데이터 행 수 {count}, 필요 {check["exact_rows"]}')
                    if count < check.get('min_rows', 0):
                        raise ValueError(f'데이터 행 수 {count}, 최소 {check["min_rows"]}')
                    detail = f'필수 컬럼 확인, 데이터 {count}행'
                else:
                    data = json.loads(text)
                    if not isinstance(data, dict):
                        raise ValueError('JSON 최상위 값이 객체가 아닙니다.')
                    mismatch = [key for key, expected in check['equals'].items()
                                if key not in data or type(data[key]) is not type(expected) or data[key] != expected]
                    if mismatch:
                        raise ValueError(f'키 누락 또는 기대값 불일치: {mismatch}')
                    detail = f'키·기대값 {len(check["equals"])}개 확인'
            item.update(passed=True, detail=detail)
        except (OSError, ValueError, csv.Error, RuntimeError) as exc:
            item['detail'] = str(exc)
        results.append(item)
    return {'passed': bool(results) and all(x['passed'] for x in results), 'checks': results}
