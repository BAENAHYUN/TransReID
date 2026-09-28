#!/usr/bin/env python
"""Offline image review index and shared, standard-library-only report helpers.

File identities and timestamps can detect stale inputs, but cannot prove a common
run: these artifacts have no shared run ID or immutable DB snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path


import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import (
    normalize_sources,
    file_info,
    common_summary,
    warning,
    write_json,
    result_markers,
    path_href,
    local_reference,
    is_noise,
    iter_assignments,
    size_histogram,
    assignment_stats,
    load_json,
    load_manifests,
    manifest_time,
    latest_build,
    esc,
    cell,
    table,
    manifest_html,
    cluster_details,
    histogram_html,
)


def build_parser():
    parser = argparse.ArgumentParser(description='이미지 DB · 클러스터 결과 오프라인 인덱스')
    parser.add_argument('--db-html', default='outputs/image_db_html/image_db_PRW.html')
    parser.add_argument('--db-summary', default=None)
    parser.add_argument('--cluster-dir', default='outputs/clustering/leiden_image_prw')
    parser.add_argument('--manifest-dir', default='data/build_manifests')
    parser.add_argument('--person-gallery-report', default=None)
    parser.add_argument('--object-gallery-report', default=None)
    parser.add_argument('--title', default='이미지 DB · 클러스터 결과 (PRW)')
    parser.add_argument('--out', default='outputs/image_review/index_PRW.html')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out, db_html, cluster_dir = Path(args.out).resolve(), Path(args.db_html).resolve(), Path(args.cluster_dir).resolve()
    warnings, inputs, checks, targets = [], [], [], {}

    def read(role, path, target=None, expected=dict):
        inputs.append(file_info(role, path))
        return load_json(path, warnings, target, expected)

    def check(name, state, detail, target=None):
        checks.append(dict(name=name, state=state, detail=detail))
        if state != '확인':
            warnings.append(warning('CHECK_MISMATCH' if state == '불일치' else 'CHECK_UNKNOWN', f'{name}: {detail}', target))

    def compare(name, left, right, detail, target=None):
        state = '확인 불가' if left is None or right is None else ('확인' if left == right else '불일치')
        check(name, state, f'{detail}: {left} / {right}', target)

    db = read('db_summary', args.db_summary or db_html.with_suffix('.summary.json'))
    db_valid = isinstance(db, dict) and isinstance(db.get('total'), (int, float)) and not isinstance(db.get('total'), bool)
    if db is not None and not db_valid:
        warnings.append(warning('INVALID_DB_SUMMARY', 'DB summary의 total이 없거나 유효하지 않음'))
    db = db if db_valid else {}
    provenance = [('DB', (db.get('config') or {}).get('settings') or {})]
    summary = read('cluster_summary', cluster_dir / 'leiden_pipeline_summary.json', expected=list)
    summary_by_target = {item.get('target') or (item.get('config') or {}).get('target'): item
                         for item in (summary or []) if isinstance(item, dict)}
    links = dict(db_html=str(db_html) if db_html.is_file() else None, person_gallery=None, object_gallery=None)
    inputs.append(file_info('db_html', db_html))
    sections = [f'<h2>DB 데이터 범위</h2>{table(db.get("filters") or {"상태": "없음"})}']
    sections.append('<h2>DB 요약</h2>' + table({key: db.get(key) for key in
                    ('total', 'images', 'n_person', 'n_object', 'n_unknown', 'build_ids', 'legacy')}))
    sections.append(table({'스캔 상태': db.get('per_collection'), '썸네일 상태': db.get('thumbnails')}))
    if db.get('legacy') and db.get('legacy') == db.get('total'):
        sections.append('<p>정보: DB point는 전부 legacy (embedding_build_id 없음)입니다.</p>')
    for name, condition in [('DB 부분 스캔', any(v.get('truncated') for v in (db.get('per_collection') or {}).values() if isinstance(v, dict))),
                            ('DB 썸네일 누락', (db.get('thumbnails') or {}).get('missing', 0) > 0)]:
        if condition:
            check(name, '불일치', '부분 스캔 또는 선택 표본 누락이 기록됨')
            sections.append(f'<p class="warning">{name}</p>')
    if (db.get('thumbnails') or {}).get('enabled') is False:
        sections.append('<p>썸네일 생성 비활성화: 이미지 누락 여부를 검증하지 않았습니다.</p>')
    manifests = load_manifests(args.manifest_dir, db.get('build_ids') or {}, warnings, inputs)
    sections.append('<h2>참조 manifest</h2>' + manifest_html(manifests))

    identity = next((v for v in db.get('inputs', []) if v.get('role') in ('out_html', 'db_html')), None)
    actual = file_info('db_html', db_html)
    if db.get('out_html') and identity and identity.get('sha256') and identity.get('size') is not None and actual['exists']:
        same = (Path(db['out_html']).resolve() == db_html and Path(identity.get('path', '')).resolve() == db_html
                and identity['sha256'] == actual['sha256'] and identity['size'] == actual['size'])
        check('DB HTML 식별', '확인' if same else '불일치', 'out_html 경로 및 HTML 크기/sha256 대조')
    else:
        check('DB HTML 식별', '확인 불가', 'HTML 없음 또는 구형 summary: 크기/sha256 식별 정보 없음')

    valid_reports = 0
    for target in ('person', 'object'):
        folder = cluster_dir / target
        rp = folder / f'{target}_leiden_report.json'
        ap = folder / f'{target}_leiden_assignments.jsonl'
        gp = Path(getattr(args, f'{target}_gallery_report') or folder / 'gallery' / 'gallery_report.json').resolve()
        report = read('cluster_report', rp, target)
        report_valid = isinstance(report, dict) and isinstance(report.get('config'), dict) and isinstance(report.get('stats'), dict) and isinstance(report['stats'].get('points'), int)
        # Pipeline summary entries carry the same report schema and are valid primary inputs.
        entry = summary_by_target.get(target)
        if not report_valid and isinstance(entry, dict) and isinstance(entry.get('config'), dict) and isinstance(entry.get('stats'), dict) and isinstance(entry['stats'].get('points'), int):
            report, report_valid = entry, True
        if report_valid:
            valid_reports += 1
        report = report if report_valid else {}
        gallery = read('gallery_report', gp, target) or {}
        provenance.append((f'{target} cluster', report.get('config') or {}))
        provenance.append((f'{target} gallery', (gallery.get('config') or {}).get('settings') or {}))
        if entry and entry.get('skipped'):
            provenance.append((f'{target} skipped', entry.get('config') or entry))
        inputs.append(file_info('assignments', ap))
        astats = assignment_stats(ap) if ap.is_file() else None
        if astats is None:
            warnings.append(warning('ASSIGNMENTS_MISSING', f'{ap}: 없음', target))
        config, stats = report.get('config') or {}, report.get('stats') or {}
        gallery_html = None
        if gallery.get('html'):
            raw = str(gallery['html'])
            if raw.lower().startswith(('http:', 'https:', 'data:')):
                warnings.append(warning('EXTERNAL_GALLERY', f'로컬 HTML 아님: {raw}', target))
            else:
                # Old reports used paths relative to the project cwd; newer reports are absolute.
                candidate = Path(raw)
                candidates = ([local_reference(raw, gp.parent)] if raw.lower().startswith('file:') else
                              [candidate.resolve(), (gp.parent / candidate).resolve()])
                gallery_html = next((p for p in candidates if p.is_file()), None)
        if gallery_html is None:
            warnings.append(warning('GALLERY_HTML_MISSING', '갤러리 HTML 없음', target))
        else:
            inputs.append(file_info('gallery_html', gallery_html))
            links[f'{target}_gallery'] = str(gallery_html)
        skipped = bool(entry and entry.get('skipped'))
        if skipped:
            check(f'{target} summary', '불일치' if rp.is_file() or gp.is_file() else '확인',
                  '이번 summary 에서 skipped' + (' / 이전 산출물 잔존' if rp.is_file() or gp.is_file() else ''), target)
        elif entry is None:
            check(f'{target} summary', '확인 불가', '이번 summary 에 없음', target)
        else:
            check(f'{target} summary', '확인', '이번 summary에 포함 (동일 실행 증명은 아님)', target)
        db_sources = normalize_sources((db.get('filters') or {}).get('source')) if 'source' in (db.get('filters') or {}) else None
        sources = normalize_sources(config['sources']) if 'sources' in config else None
        compare(f'{target} source', set(db_sources) if db_sources is not None else None,
                set(sources) if sources is not None else None, 'DB source 집합 / 클러스터 sources 집합', target)
        collection = config.get('collection')
        compare(f'{target} collection', collection in db['collections'] if collection and isinstance(db.get('collections'), list) else None,
                True, f'클러스터 {collection} / DB {db.get("collections")}', target)
        compare(f'{target} assignments 수', stats.get('points'), astats['valid_assignments'] if astats else None, 'report points / 유효 레코드', target)
        compare(f'{target} gallery 수', gallery.get('assignments'), stats.get('points'), 'gallery assignments / report points', target)
        for role, path in (('assignments', ap), ('report', rp)):
            current = file_info(role, path)
            recorded = next((v for v in gallery.get('inputs', []) if v.get('role') == role), None)
            if not recorded or not recorded.get('sha256') or not current['exists']:
                check(f'{target} {role} sha256', '확인 불가', '구형 report 또는 현재 파일 없음', target)
            else:
                same = current['sha256'] == recorded['sha256']
                check(f'{target} {role} sha256', '확인' if same else '불일치',
                      '현재 파일과 동일' if same else '갤러리가 다른 입력으로 생성됨', target)
        times = {}
        for key, path in [('DB HTML', db_html), ('report', rp), ('assignments', ap), ('gallery HTML', gallery_html)]:
            times[key] = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat() if path and path.is_file() else '없음'
        if gallery_html and rp.is_file():
            stale = gallery_html.stat().st_mtime < rp.stat().st_mtime
            check(f'{target} 시간 순서', '불일치' if stale else '확인',
                  '갤러리 HTML이 report보다 오래됨' if stale else '갤러리 HTML mtime >= report mtime (동일 실행 증명은 아님)', target)
        else:
            check(f'{target} 시간 순서', '확인 불가', '시간 대조 파일 없음', target)
        if gallery.get('missing_images', 0) > 0:
            check(f'{target} 갤러리 이미지', '불일치', f'선택 표본 누락 {gallery["missing_images"]}', target)
        if astats and astats['parse_errors']['count']:
            check(f'{target} assignments 파싱', '불일치', f'깨진 줄: {astats["parse_errors"]}', target)
        if (gallery.get('parse_errors') or {}).get('count', 0):
            check(f'{target} gallery 생성 시 파싱', '불일치', f'깨진 줄: {gallery["parse_errors"]}', target)
        targets[target] = dict(stats=stats, assignments=astats, coverage=gallery.get('coverage'),
                               missing_images=gallery.get('missing_images'), skipped=skipped,
                               config=config, dry_run=report.get('dry_run'))
        scope = {'sources': config.get('sources'), 'collection': collection, 'media_type': '제한 없음',
                 'max_points': f'제한된 대상 ({config["max_points"]})' if config.get('max_points') else '제한 없음'}
        skip_label = '— skipped / 이전 산출물' if skipped and (rp.is_file() or gp.is_file()) else '— skipped' if skipped else ''
        sections.append(f'<h2>{target} 클러스터 {skip_label}</h2>')
        metrics = {key: stats.get(key) for key in ('points', 'kept_clusters', 'clustered_points', 'noise_points')}
        metrics['noise 비율'] = astats['noise_ratio'] if astats else (stats.get('noise_points', 0) / stats['points'] if stats.get('points') else None)
        metrics['최종 최대 (assignments)'] = astats['largest_cluster'] if astats else None
        # 범위(scope)와 설정(cluster_details)에 겹치는 키(sources/collection/max_points)는 한 번만 보여준다.
        settings = {**cluster_details(report), 'media_type': scope['media_type']}
        sections.append('<div class="grid"><div><h3>수량</h3>' + table(metrics) + '</div>'
                        '<div><h3>범위 · 설정</h3>' + table(settings) + '</div></div>')
        sections.append('<h3>크기 히스토그램 (noise 제외)</h3>' + (histogram_html(astats['size_histogram']) if astats else '없음'))
        sections.append('<h3>갤러리 커버리지</h3>' + table(gallery.get('coverage') or {'상태': '미기록'}))
        sections.append('<h3>수정 시각 (mtime)</h3>' + table(times))

    for name, key in (('config 경로 일치', 'config_path'), ('config 내용 일치', 'config_sha256')):
        recorded = [(label, value[key]) for label, value in provenance if value.get(key)]
        state = '확인 불가' if len(recorded) < 2 else ('확인' if len({v for _, v in recorded}) == 1 else '불일치')
        detail = ('확인 불가(구형 산출물): 기록 2개 미만' if len(recorded) < 2 else
                  ' / '.join(f'{label}: {value}' for label, value in recorded))
        check(name, state, detail)
    sections.append('<h2>YAML 설정 출처</h2>' + table({
        label: {'yaml 파일': Path(value['config_path']).name if value.get('config_path') else None,
                'sha256 (앞 12자)': str(value['config_sha256'])[:12] if value.get('config_sha256') else None}
        for label, value in provenance}))

    if not db_valid and not valid_reports:
        print('오류: 유효한 주 입력(DB summary 또는 cluster report)이 없습니다.')
        return 1
    link_labels = {'db_html': 'DB 리포트 HTML', 'person_gallery': 'person 클러스터 갤러리', 'object_gallery': 'object 클러스터 갤러리'}
    sections.insert(0, '<h2>링크</h2><ul class="links">' + ''.join(
        f'<li>{esc(link_labels.get(key, key))}: <a href="{path_href(value, out)}" target="_blank">열기</a> '
        f'<span class="muted">{esc(Path(value).name)}</span></li>' if value else f'<li>{esc(link_labels.get(key, key))}: 없음</li>'
        for key, value in links.items()) + '</ul>')
    state_class = {'확인': 'ok', '불일치': 'bad', '확인 불가': 'unknown'}
    counts = Counter(c['state'] for c in checks)
    badge = ' · '.join(f'<span class="state {state_class.get(s, "unknown")}">{esc(s)} {counts[s]}</span>'
                       for s in ('확인', '불일치', '확인 불가') if counts.get(s))
    sections.append('<h2>정합성 점검</h2><p>' + (badge or '점검 항목 없음') + '</p><table>' + ''.join(
        f'<tr><th>{esc(c["name"])}</th><td><span class="state {state_class.get(c["state"], "unknown")}">{esc(c["state"])}</span></td>'
        f'<td>{esc(c["detail"])}</td></tr>' for c in checks) + '</table>')
    sections.append('<h2>경고</h2>' + ('<ul>' + ''.join(
        f'<li>{esc(w["target"] or "공통")}: {esc(w["message"])}</li>' for w in warnings) + '</ul>' if warnings
        else '<p class="muted">경고 없음</p>'))
    result = common_summary('build_image_review_index.py', out, inputs, warnings)
    result.update(checks=checks, links=links, targets=targets)
    document = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{esc(args.title)}</title><style>body{{background:#1a1a19;color:#fff;font:15px/1.6 system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem}}'
                'table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #35352f;padding:.45rem;text-align:left;overflow-wrap:anywhere}'
                'th{color:#c3c2b7;width:34%}a{color:#3987e5}.bar{background:#3987e5}.warning{color:#eb6834}'
                'h2{margin-top:2.2rem;padding-bottom:.3rem;border-bottom:1px solid #35352f}h3{margin:.8rem 0 .2rem;color:#c3c2b7;font-size:1rem}'
                '.muted{color:#8f8e84}.grid{display:grid;gap:1rem;grid-template-columns:1fr 1fr}@media(max-width:800px){.grid{grid-template-columns:1fr}}'
                'table table{margin:0}td td,td th{border-color:#2a2a27;padding:.25rem .4rem}'
                '.state{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.85rem;font-weight:600}'
                '.state.ok{background:#1f4d2b;color:#9be3ad}.state.bad{background:#5a2317;color:#ffb39e}.state.unknown{background:#3a3a36;color:#d6d6cf}'
                'ul.links li{margin:.25rem 0}footer{margin-top:2.5rem;color:#8f8e84;font-size:.9rem}</style></head><body>'
                f'<h1>{esc(args.title)}</h1><p>생성 시각: {esc(result["generated_at"])}</p>' + ''.join(sections) +
                '<footer>한계: DB snapshot/공통 run id가 없어 동일 실행을 증명할 수 없습니다. compat.total은 입력 stats 수량이며 현재 필터의 DB point 수가 아닙니다.</footer></body></html>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding='utf-8', newline='\n')
    sidecar = out.with_suffix('.summary.json')
    write_json(sidecar, result)
    result_markers(sidecar, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
