"""Shared report helpers and lazy pipeline settings (standard library at import time)."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from urllib.request import url2pathname


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_ROOT / 'pipeline.yaml'
CONFIG_HELP = '기본: pipeline.yaml (스크립트 폴더 기준)'


@dataclass
class PipelineSettings:
    config_path: str
    config_sha256: str
    collection_prefix: str
    person_collection: str
    object_collection: str
    qdrant_url: str
    retrievers: dict
    clustering: dict
    inferred: dict

    def collection_for(self, target):
        if target not in ('person', 'object'):
            raise ValueError(f'unknown target: {target}')
        return getattr(self, f'{target}_collection')

    def vector_for(self, target):
        return self.clustering[target]['vector']

    def threshold_for(self, target):
        return self.clustering[target]['threshold']

    def summary(self):
        return {key: getattr(self, key) for key in (
            'config_path', 'config_sha256', 'collection_prefix', 'person_collection',
            'object_collection', 'qdrant_url', 'clustering', 'inferred')}


def validate_vector(retrievers, vector, target, name):
    candidates = [key for key, spec in retrievers.items() if spec['scope'] in ('all', target)]
    if not isinstance(vector, str) or vector not in candidates:
        raise ValueError(f'{name}: {vector!r} is not a {target} retriever; 후보: {candidates}')
    return vector


def validate_threshold(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name}: finite float in [-1, 1] required') from exc
    if isinstance(value, bool) or not math.isfinite(result) or not -1 <= result <= 1:
        raise ValueError(f'{name}: finite float in [-1, 1] required')
    return result


def load_pipeline_settings(config_path=None, require=True):
    path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        if require:
            raise FileNotFoundError(f'config file not found: {path}') from None
        return None
    # Parse, validate and fingerprint precisely the same bytes, including any BOM.
    import yaml
    from config import PipelineConfig

    raw = yaml.safe_load(content.decode('utf-8-sig'))
    if not isinstance(raw, Mapping):
        raise ValueError('pipeline config must be a mapping')
    cfg = PipelineConfig.from_dict(raw, base_dir=path.parent)

    def mapping(value, allowed, name):
        if not isinstance(value, Mapping):
            raise ValueError(f'{name}: mapping required')
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f'{name}: unknown keys {sorted(unknown, key=str)}')
        return value

    section = raw.get('clustering')
    section = mapping({} if section is None else section,
                      {'person', 'object', 'secondary_vector', 'sources'}, 'clustering')
    retrievers = {name: dict(scope=spec.scope, dim=spec.dim, supports_text=spec.supports_text)
                  for name, spec in cfg.retrievers.items()}
    clustering, inferred = {}, {}
    for target, preferred, threshold in (('person', 'solider', .95), ('object', 'dinov2', .90)):
        item = mapping(section.get(target, {}), {'vector', 'threshold'}, f'clustering.{target}')
        candidates = [name for name, spec in cfg.retrievers.items()
                      if getattr(spec, f'accepts_{target}')()]
        if 'vector' in item:
            vector = item['vector']
        else:
            first = next((name for name, spec in cfg.retrievers.items()
                          if spec.scope == target and (target != 'person' or not spec.supports_text)), None)
            vector = preferred if preferred in candidates else first or next(iter(candidates), None)
            inferred[f'{target}.vector'] = f'default {preferred}' if vector == preferred else f'first compatible retriever: {vector}'
        vector = validate_vector(retrievers, vector, target, f'{target}.vector')
        if 'threshold' not in item:
            inferred[f'{target}.threshold'] = f'default {threshold}'
        clustering[target] = dict(vector=vector, threshold=validate_threshold(
            item.get('threshold', threshold), f'{target}.threshold'))

    if 'secondary_vector' in section:
        secondary = validate_vector(retrievers, section['secondary_vector'], 'person', 'secondary_vector')
    else:
        secondary = 'irra' if 'irra' in cfg.retrievers and cfg.retrievers['irra'].accepts_person() else None
        inferred['secondary_vector'] = 'default irra' if secondary else 'no compatible irra; disabled'
    sources = section.get('sources')
    if 'sources' in section:
        if not isinstance(sources, (list, str)):
            raise ValueError('clustering.sources: list or comma-separated string required')
        sources = normalize_sources(sources)
        if not sources:
            raise ValueError('clustering.sources: empty sources are ambiguous')
    else:
        inferred['sources'] = 'unspecified; consumer fallback'
    clustering.update(secondary_vector=secondary, sources=sources)
    return PipelineSettings(str(path), hashlib.sha256(content).hexdigest(), cfg.collection_prefix,
                            cfg.person_collection(), cfg.object_collection(), cfg.qdrant.url,
                            retrievers, clustering, inferred)


def resolve(cli_value, settings_value, name):
    if cli_value is not None and cli_value != '':
        return cli_value
    if settings_value is not None and settings_value != '':
        return settings_value
    raise ValueError(f'{name}: --config 또는 플래그로 지정')


def applied_config(settings, **applied):
    return {'settings': settings.summary() if settings else None, 'applied': applied}


def normalize_sources(value):
    values = value.split(',') if isinstance(value, str) else (value or [])
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))


def file_info(role, path):
    path = Path(path).resolve()
    result = dict(role=role, path=str(path), exists=path.is_file(), size=None)
    if result['exists']:
        result['size'] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result['sha256'] = digest.hexdigest()
    return result


def common_summary(producer, out, inputs, warnings):
    return dict(schema_version=1, producer=producer,
                generated_at=datetime.now().astimezone().isoformat(),
                status='warning' if warnings else 'ok', warnings=warnings,
                out_html=str(Path(out).resolve()), inputs=inputs)


def warning(code, message, target=None):
    return dict(code=code, severity='warning', target=target, message=message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')


def result_markers(sidecar, out):
    print(f'RESULT_SUMMARY: {Path(sidecar).resolve()}')
    print(f'RESULT_HTML: {Path(out).resolve()}')


def path_href(target, out):
    """Return an HTML-escaped local URL; URLs must not be treated as filenames."""
    raw = str(target)
    if raw.lower().startswith(('https://', 'http://', 'data:', 'file:')):
        return html.escape(raw, quote=True)
    path = Path(target).resolve()
    try:
        value = quote(os.path.relpath(path, Path(out).resolve().parent).replace('\\', '/'), safe='/')
    except ValueError:
        value = path.as_uri()
    return html.escape(value, quote=True)


def local_reference(raw, base):
    """Decode an HTML local reference. None means an external URL."""
    value = html.unescape(str(raw))
    if value.lower().startswith(('http:', 'https:', '//', 'data:')):
        return None
    if value.lower().startswith('file:'):
        parts = urlsplit(value)
        value = url2pathname(('//' + parts.netloc if parts.netloc else '') + parts.path)
    else:
        value = unquote(value).replace('\\', '/')
    path = Path(value)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def is_noise(row):
    return bool(row.get('noise')) or bool(row.get('is_noise')) or row.get('cluster_id') is None


def iter_assignments(path, errors):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or 'point_id' not in row:
                    raise ValueError('point_id missing')
            except (ValueError, TypeError):
                errors['count'] += 1
                if errors['first_line'] is None:
                    errors['first_line'] = number
                continue
            yield row


def size_histogram(counts):
    hist = dict.fromkeys(('1', '2–4', '5–19', '20–99', '100–499', '500+'), 0)
    for size in counts:
        key = ('1' if size == 1 else '2–4' if size < 5 else '5–19' if size < 20
               else '20–99' if size < 100 else '100–499' if size < 500 else '500+')
        hist[key] += 1
    return hist


def assignment_stats(path):
    """Stream assignments with O(number of clusters) additional memory."""
    errors = dict(count=0, first_line=None)
    counts = Counter()
    total = noise = 0
    for row in iter_assignments(path, errors):
        total += 1
        if is_noise(row):
            noise += 1
        else:
            counts[str(row['cluster_id'])] += 1
    return dict(valid_assignments=total, noise=noise, noise_ratio=noise / total if total else None,
                clusters=len(counts), largest_cluster=max(counts.values(), default=0),
                size_histogram=size_histogram(counts.values()), parse_errors=errors)


def load_json(path, warnings, target=None, expected=dict):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, expected):
            raise ValueError(f'expected {expected.__name__}')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(warning('INPUT_UNAVAILABLE', f'{path}: {exc}', target))
        return None


def load_manifests(directory, build_ids, warnings, inputs):
    manifests = {}
    for build_id in build_ids:
        # IDs are payload data, not paths. Never allow directory traversal.
        if Path(str(build_id)).name != str(build_id) or '/' in str(build_id) or '\\' in str(build_id):
            warnings.append(warning('INVALID_BUILD_ID', f'manifest 없음: {build_id}'))
            manifests[build_id] = None
            continue
        path = Path(directory) / f'{build_id}.json'
        inputs.append(file_info('manifest', path))
        manifests[build_id] = load_json(path, warnings)
    return manifests


def manifest_time(manifest):
    try:
        value = datetime.fromisoformat(manifest['created_at'])
        return value.astimezone(timezone.utc) if value.tzinfo else None
    except (TypeError, KeyError, ValueError):
        return None


def latest_build(manifests):
    dated = [(manifest_time(value), key) for key, value in manifests.items() if manifest_time(value)]
    return max(dated)[1] if dated else None


def esc(value):
    return html.escape(str(value) if value is not None else '미기록', quote=True)


def cell(value):
    """표 셀: dict 는 중첩 표, list/tuple 은 쉼표 나열, 그 외는 escape 문자열."""
    if isinstance(value, dict):
        return table(value) if value else esc('없음')
    if isinstance(value, (list, tuple)):
        return esc(', '.join(str(v) for v in value)) if value else esc('없음')
    if isinstance(value, bool):
        return esc('예' if value else '아니오')
    if isinstance(value, float):
        return esc(f'{value:.4f}'.rstrip('0').rstrip('.'))
    return esc(value)


def table(values):
    return '<table>' + ''.join(f'<tr><th>{esc(k)}</th><td>{cell(v)}</td></tr>' for k, v in values.items()) + '</table>'


def manifest_html(manifests):
    if not manifests:
        return '<p class="muted">참조 build 없음 — embedding_build_id 를 가진 point 가 없습니다 (전부 legacy).</p>'
    rows = []
    newest = latest_build(manifests)
    for key, value in manifests.items():
        if value is None:
            rows.append(f'<tr><td>{esc(key)}</td><td colspan="3">manifest 없음 · 시각 미확인</td></tr>')
            continue
        compat = value.get('compat') or {}
        dims = ', '.join(f'{name}: {item.get("dim", "미기록")}'
                         for name, item in (compat.get('retrievers') or {}).items() if isinstance(item, dict))
        date = value.get('created_at') if manifest_time(value) else '시각 미확인'
        rows.append(f'<tr><td>{esc(key)}{" (최신 확인 시각)" if key == newest else ""}</td>'
                    f'<td>{esc(date)}</td><td>{esc(compat.get("total"))}</td><td>{esc(dims)}</td></tr>')
    return ('<table><tr><th>임베딩 build</th><th>created_at</th><th>입력 stats 수량</th>'
            '<th>retriever dim</th></tr>' + ''.join(rows) + '</table>')


def cluster_details(report):
    report = report or {}
    config, stats = report.get('config') or {}, report.get('stats') or {}
    result = {key: config.get(key, '미기록') for key in
              ('sources', 'collection', 'vector', 'knn', 'score_threshold', 'resolution', 'mutual_knn')}
    result['max_cluster_size'] = ('상한 없음' if config.get('max_cluster_size') is None
                                  else config['max_cluster_size']) if 'max_cluster_size' in config else '미기록'
    result['max_points'] = f'제한된 대상 ({config["max_points"]})' if config.get('max_points') else '제한 없음'
    result['dry_run'] = ('Qdrant payload 미기록' if report['dry_run'] else '기록됨') if isinstance(report.get('dry_run'), bool) else '미기록'
    result['largest_community (재분할 전)'] = stats.get('largest_community', '미기록')
    for key in ('largest_after_refine', 'oversized_remaining'):
        if key in stats:
            result[key] = stats[key]
    return result


def histogram_html(hist):
    maximum = max(hist.values(), default=0) or 1
    return '<table>' + ''.join(f'<tr><th>{esc(k)}</th><td>{v}</td><td style="width:60%">'
                               f'<div class="bar" style="width:{v / maximum * 100:.2f}%">&nbsp;</div></td></tr>'
                               for k, v in hist.items()) + '</table>'

