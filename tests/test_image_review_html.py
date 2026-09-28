"""Bounded offline report tests. All generated files stay in TemporaryDirectory."""
import contextlib
import builtins
from collections import Counter
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from PIL import Image
import requests
from report import build_image_db_html as db
from report import build_leiden_gallery as gallery
from report import build_image_review_index as review
from report import inline_gallery_html as inline
from report_common import load_pipeline_settings, applied_config


ROOT = Path(__file__).resolve().parents[1]


class OfflineReviewTests(unittest.TestCase):
    def setUp(self):
        # Fail immediately for requests and any underlying network socket.
        for name in ('requests.Session.post', 'requests.Session.get', 'requests.Session.request',
                     'socket.socket', 'socket.create_connection', 'socket.getaddrinfo'):
            guard = patch(name, side_effect=AssertionError('NETWORK FORBIDDEN'))
            guard.start()
            self.addCleanup(guard.stop)
        temp = tempfile.TemporaryDirectory(prefix='image_review_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # Also reject any accidental generated artifact outside this fixture tree.
        def guarded_open(original):
            def call(file, mode='r', *args, **kwargs):
                if not isinstance(file, int) and any(flag in mode for flag in 'wax+'):
                    self.assertTrue(Path(file).resolve().is_relative_to(self.root), f'WRITE OUTSIDE TEMP: {file}')
                return original(file, mode, *args, **kwargs)
            return call
        for module in (builtins, io):
            guard = patch.object(module, 'open', guarded_open(module.open))
            guard.start()
            self.addCleanup(guard.stop)
        self.config_path = self.root / 'pipeline.yaml'
        # Fixture settings keep all report tests independent of the production YAML.
        self.config_path.write_text(json.dumps(dict(collection_prefix='forensic', retrievers={
            'solider': dict(scope='person', dim=768, supports_text=False, module='fake', **{'class': 'Fake'}),
            'dinov2': dict(scope='object', dim=1024, supports_text=False, module='fake', **{'class': 'Fake'}),
            'irra': dict(scope='person', dim=512, supports_text=True, module='fake', **{'class': 'Fake'}),
        })), encoding='utf-8')
        self.settings = load_pipeline_settings(self.config_path)
        for module in (db, gallery):
            guard = patch.object(module, 'DEFAULT_CONFIG_PATH', self.config_path)
            guard.start()
            self.addCleanup(guard.stop)
        self.crops = []
        for index in range(4):
            path = self.root / '공백 한글 # % &' / f'crop {index} # &%.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (18, 24), (30 * index, 80, 120)).save(path)
            self.crops.append(path)
        self.cluster = self.root / 'cluster # 한글 & %'
        self.assignments = self.cluster / 'person' / 'person_leiden_assignments.jsonl'
        self.assignments.parent.mkdir(parents=True)
        rows = [dict(point_id='p0', cluster_id='a', noise=False),
                dict(point_id='p1', cluster_id='a', noise=False),
                dict(point_id='p2', cluster_id='b', noise=False),
                dict(point_id='p3', cluster_id='wrong', noise=True),
                dict(point_id='p4', cluster_id='wrong', is_noise=True),
                dict(point_id='p5', cluster_id=None, noise=False)]
        self.assignments.write_text('\n'.join(json.dumps(r) for r in rows) + '\nBAD JSON\n\n', encoding='utf-8')
        self.report_path = self.assignments.with_name('person_leiden_report.json')
        self.report = dict(config=dict(target='person', sources=['prw_image'], collection='forensic_person',
                                      vector='solider', knn=30, score_threshold=.95, resolution=1.0,
                                      mutual_knn=True, max_cluster_size=None, max_points=500),
                           stats=dict(points=6, kept_clusters=2, clustered_points=3, noise_points=3,
                                      largest_community=100, largest_after_refine=4, oversized_remaining=1), dry_run=True)
        self.report['config'].update(config_path=self.settings.config_path, config_sha256=self.settings.config_sha256)
        review.write_json(self.report_path, self.report)
        review.write_json(self.cluster / 'leiden_pipeline_summary.json', [self.report, dict(target='object', skipped=True)])
        self.payloads = {f'p{i}': dict(image_id=f'image_{i}', crop_path=str(self.crops[i % 4]),
                                      bbox=[1.2, 2.4, 10.7, 20.6], score=0, detection_id=0, label='person') for i in range(6)}
        self.manifest_dir = self.root / 'manifests'
        review.write_json(self.manifest_dir / 'z_old.json', dict(embedding_build_id='z_old', created_at='2026-09-17T22:56:55+0900', compat=dict(total=900, retrievers={'siglip2': {'dim': 768}})))
        review.write_json(self.manifest_dir / 'a_new.json', dict(embedding_build_id='a_new', created_at='2026-09-18T22:56:55+0900', compat=dict(total=1000, retrievers={'siglip2': {'dim': 768}})))
        self.db_html = self.root / 'DB # 공백 & %' / 'db # & %.html'
        self.db_html.parent.mkdir()
        self.db_html.write_text('<html>DB fixture</html>', encoding='utf-8')
        self.db_summary = review.common_summary('build_image_db_html.py', self.db_html, [review.file_info('out_html', self.db_html)], [])
        self.db_summary['config'] = applied_config(self.settings)
        self.db_summary.update(total=6, images=4, n_person=6, n_object=0, n_unknown=0,
                               collections=['forensic_person', 'forensic_object'], filters=dict(source=['prw_image'], media_type='image'),
                               build_ids={'z_old': 2, 'a_new': 2, 'no_manifest': 1}, legacy=1,
                               thumbnails=dict(enabled=True, sampled=6, ok=6, missing=0, fallback_used=0),
                               per_collection={'forensic_person': dict(count=6, scanned=6, truncated=False, stop_reason='exhausted')})
        review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        self.gui = json.loads((ROOT / 'gui_pipelines.json').read_text(encoding='utf-8'))['image_pipeline']

    def gui_argv(self, step_id, overrides=None):
        step = next(s for s in self.gui if s['id'] == step_id)
        result = []
        for arg in step['args']:
            defaults = {'--config': self.config_path}
            value = (overrides or {}).get(arg['flag'], defaults.get(arg['flag'], arg.get('default')))
            if arg['type'] == 'bool':
                if value:
                    result.append(arg['flag'])
            elif value is not None and value != '':
                result.extend([arg['flag'], str(value)])
        return result

    def call(self, func, argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = func(argv)
        return code, output.getvalue()

    def assert_markers(self, output, sidecar, html):
        markers = [line for line in output.splitlines() if line.startswith('RESULT_')]
        self.assertEqual(markers, [f'RESULT_SUMMARY: {sidecar.resolve()}', f'RESULT_HTML: {html.resolve()}'])
        self.assertTrue(sidecar.is_file())
        self.assertTrue(html.is_file())

    def run_gallery(self, target='person', inline_images=True, extra=None):
        output = self.cluster / target / 'gallery'
        args = self.gui_argv('image_gallery_' + target, {'--assignments': self.assignments, '--output-dir': output,
                                                       '--inline-images': inline_images})
        args += ['--project-root', str(self.root)] + (extra or [])
        with patch.object(gallery.QdrantHTTP, 'retrieve_points', return_value=self.payloads):
            code, text = self.call(gallery.main, args)
        self.assertEqual(code, 0)
        html = output / f'leiden_{target}_gallery.html'
        sidecar = output / 'gallery_report.json'
        self.assert_markers(text, sidecar, html)
        return html, json.loads(sidecar.read_text(encoding='utf-8'))

    def index_args(self):
        return ['--db-html', str(self.db_html), '--cluster-dir', str(self.cluster),
                '--manifest-dir', str(self.manifest_dir), '--out', str(self.root / 'new out # & % 한글' / 'index.html')]

    def run_index(self, extra=None):
        args = self.index_args() + (extra or [])
        code, text = self.call(review.main, args)
        out = Path(args[args.index('--out') + 1])
        self.assertEqual(code, 0)
        self.assert_markers(text, out.with_suffix('.summary.json'), out)
        return out.read_text(encoding='utf-8'), json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))

    def test_network_guard(self):
        for action in (lambda: requests.Session().get('http://localhost:6333'),
                       lambda: requests.Session().post('http://localhost:6333'), socket.socket):
            with self.assertRaisesRegex(AssertionError, 'NETWORK FORBIDDEN'):
                action()

    def test_gui_default_parsers_and_help(self):
        for step_id, module in [('image_db_html', db), ('image_gallery_person', gallery),
                                ('image_gallery_object', gallery), ('image_review_index', review)]:
            args = self.gui_argv(step_id)
            parser = module.build_parser()
            parser.parse_args(args)  # No main(), no default path IO.
            help_text = parser.format_help()
            for flag in args:
                if flag.startswith('--'):
                    self.assertIn(flag, help_text)
        self.assertTrue(inline.parse_args(['--html', 'x']).drop_links)
        self.assertFalse(inline.parse_args(['--html', 'x', '--keep-links']).drop_links)

    def test_invalid_args_precede_io(self):
        cases = [(db, flag, value) for flag, value in [('--chart-labels', '0'), ('--samples', '-1'),
                 ('--thumb-size', '15'), ('--thumb-quality', '96'), ('--scroll-batch', '0'), ('--max-scan', '-1')]]
        cases += [(gallery, flag, value) for flag, value in [('--top-clusters', '-1'), ('--medium-clusters', '-1'),
                  ('--small-clusters', '-1'), ('--noise-samples', '-1'), ('--images-per-cluster', '0'),
                  ('--thumb-size', '15'), ('--qdrant-batch-size', '0'), ('--thumb-quality', '0')]]
        cases += [(inline, '--thumb-size', '15'), (inline, '--thumb-quality', '96')]
        for module, flag, value in cases:
            with self.subTest(module=module.__name__, flag=flag), patch.object(Path, 'open', side_effect=AssertionError('IO BEFORE VALIDATION')), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    module.main((['--html', 'missing'] if module is inline else []) + [flag, value])
                self.assertEqual(result.exception.code, 2)

    def test_gallery_gui_inline_and_counts(self):
        path, report = self.run_gallery()
        content = path.read_text(encoding='utf-8')
        for value in ['image_0', 'crop 0 # &amp;%.jpg', '1,2,11,21', '<b>score</b> 0.000', '<b>detection_id</b> 0']:
            self.assertIn(value, content)
        for field in ('track', 'canonical', 'frame', 'video'):
            self.assertNotIn(f'<b>{field}</b>', content)
        for key in ('assignments', 'clusters', 'noise', 'selected_clusters', 'copied_images', 'missing_images', 'html'):
            self.assertIn(key, report)
        self.assertEqual(report['valid_assignments'], 6)
        self.assertEqual(report['noise'], 3)
        self.assertEqual(report['noise_ratio'], .5)
        self.assertEqual(report['largest_cluster'], 2)
        self.assertEqual(report['size_histogram'], {'1': 1, '2–4': 1, '5–19': 0, '20–99': 0, '100–499': 0, '500+': 0})
        self.assertEqual(report['coverage'], dict(selected_clusters=2, total_clusters=2, sampled_points=6, shown_points=6, total_points=6))
        self.assertEqual(report['parse_errors'], dict(count=1, first_line=7))
        self.assertEqual(report['schema_version'], 1)
        self.assertIs(report['dry_run'], True)
        self.assertIsNone(report['max_cluster_size'])
        self.assertTrue(Path(report['html']).is_absolute())
        self.assertIn('sha256', report['inputs'][0])
        self.assertIn('상한 없음', content)
        self.assertIn('제한된 대상', content)

    def test_gallery_video_zero_and_missing(self):
        self.payloads['p0'].update(frame_idx=0, track_id=0, canonical_person_id=0, video_path='movie # 한글.mp4')
        self.payloads['p1']['crop_path'] = str(self.root / 'missing.jpg')
        path, report = self.run_gallery()
        content = path.read_text(encoding='utf-8')
        for field in ('frame', 'track', 'canonical'):
            self.assertIn(f'<b>{field}</b> 0', content)
        self.assertIn('<b>video</b>', content)
        self.assertEqual(report['copied_images'], 5)
        self.assertEqual(report['coverage']['sampled_points'], 6)
        self.assertEqual(report['missing_images'], 1)
        self.assertIn('MISSING_IMAGES', [w['code'] for w in report['warnings']])

    def test_gallery_external_refreshes_mtime(self):
        path, report = self.run_gallery(inline_images=False)
        copied = path.parent / 'assets' / 'cluster_001' / 'p0.jpg'
        self.assertEqual(copied.read_bytes(), self.crops[0].read_bytes())
        Image.new('RGB', (18, 24), (250, 0, 0)).save(self.crops[0])
        old = copied.stat().st_mtime
        os.utime(self.crops[0], (old + 20, old + 20))
        self.run_gallery(inline_images=False)
        self.assertEqual(copied.read_bytes(), self.crops[0].read_bytes())
        self.assertEqual(copied.stat().st_mtime_ns, self.crops[0].stat().st_mtime_ns)
        self.assertEqual(report['copied_images'], 6)

    def test_report_optional_and_variants(self):
        for dry, limit, expected in [(False, 4, '기록됨'), (None, 'absent', '미기록')]:
            data = json.loads(json.dumps(self.report))
            if dry is None:
                data.pop('dry_run')
                data['config'].pop('max_cluster_size')
            else:
                data['dry_run'] = dry
                data['config']['max_cluster_size'] = limit
            review.write_json(self.report_path, data)
            path, report = self.run_gallery()
            self.assertIn(expected, path.read_text(encoding='utf-8'))
            self.assertEqual(report['dry_run'], dry)
        path, report = self.run_gallery(extra=['--report', str(self.root / 'absent.json')])
        self.assertIn('미기록', path.read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'warning')

    def test_assignment_stream_and_empty(self):
        result = review.assignment_stats(self.assignments)
        self.assertEqual(result['valid_assignments'], 6)
        self.assertEqual(result['noise'], 3)
        self.assertEqual(review.size_histogram([1, 2, 4, 5, 19, 20, 99, 100, 499, 500]),
                         {'1': 1, '2–4': 2, '5–19': 2, '20–99': 2, '100–499': 2, '500+': 1})
        self.assignments.write_text('\n[]\n{}\nBAD\n', encoding='utf-8')
        result = review.assignment_stats(self.assignments)
        self.assertIsNone(result['noise_ratio'])
        self.assertEqual(result['parse_errors'], dict(count=3, first_line=2))
        _, result = self.run_gallery()
        self.assertEqual(result['valid_assignments'], 0)

    @staticmethod
    def models():
        return SimpleNamespace(FieldCondition=lambda **k: k, MatchValue=lambda **k: k,
                               MatchAny=lambda **k: k, Filter=lambda **k: k)

    def fake_client(self, size=7):
        points = [SimpleNamespace(id=i, payload=dict(self.payloads['p0'], embedding_build_id='a_new' if i % 2 else None,
                                                     is_person=True, media_type='image', source='prw_image')) for i in range(size)]
        calls = []
        def scroll(**kw):
            calls.append(kw['limit'])
            start = kw['offset'] or 0
            end = min(size, start + kw['limit'])
            return points[start:end], end if end < size else None
        return SimpleNamespace(scroll=scroll, collection_exists=lambda _: True), calls

    def test_sources_and_exact_max_scan(self):
        models = self.models()
        self.assertEqual(review.normalize_sources(' A, ,B,A, a '), ['A', 'B', 'a'])
        self.assertEqual(db.build_filter(models, 'any', ' A, A ')['must'][0]['match'], {'value': 'A'})
        self.assertEqual(db.build_filter(models, 'any', ' A,B, A ')['must'][0]['match'], {'any': ['A', 'B']})
        self.assertIsNone(db.build_filter(models, 'any', ' , '))
        for limit, expected, stop, calls_expected in [(5, 5, 'max_scan', [4, 1]), (7, 7, 'exhausted', [4, 3]), (0, 7, 'exhausted', [4, 4])]:
            client, calls = self.fake_client()
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                rows, scanned = db.scroll_collection(client, models, 'collection', 'image', 'prw_image', limit, 4, metadata)
            self.assertEqual(scanned, expected)
            self.assertEqual(len(rows), expected)
            self.assertEqual(metadata['stop_reason'], stop)
            self.assertEqual(metadata['truncated'], stop == 'max_scan')
            self.assertEqual(calls, calls_expected)

    def test_db_stats_sampling_bars_escaping(self):
        client, _ = self.fake_client()
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _ = db.scroll_collection(client, self.models(), 'x', 'image', None, 0, 4)
        stats = db.summarize(rows)
        self.assertEqual(stats['build_ids'], {'a_new': 3})
        self.assertEqual(stats['legacy'], 4)
        self.assertEqual(db.stratified_sample(rows, 3, 42), db.stratified_sample(rows, 3, 42))
        self.assertEqual(len(db.stratified_sample(rows, 3, 42)), 3)
        self.assertEqual(db.stratified_sample(rows, 0, 42), [])
        self.assertIn('없습니다', db.bar_rows(Counter(), 25))
        self.assertIn('없습니다', db.bar_rows(Counter({'x': 2}), 0))
        rows[0]['label'] = '<>&"\''
        content = db.build_html(db.summarize(rows), rows[:1], {})
        self.assertIn('&lt;&gt;&amp;&quot;&#x27;', content)
        self.assertNotIn('data-label="<', content)

    def test_db_main_sidecar_and_no_thumbs(self):
        client, _ = self.fake_client()
        module = SimpleNamespace(QdrantClient=Mock(return_value=client), models=self.models())
        out = self.root / 'new db' / 'report.html'
        args = self.gui_argv('image_db_html', {'--out': out}) + ['--manifest-dir', str(self.manifest_dir), '--max-scan', '5']
        with patch.dict(sys.modules, {'qdrant_client': module}):
            code, output = self.call(db.main, args)
        self.assertEqual(code, 0)
        sidecar = out.with_suffix('.summary.json')
        self.assert_markers(output, sidecar, out)
        result = json.loads(sidecar.read_text(encoding='utf-8'))
        self.assertEqual(result['total'], 10)
        self.assertEqual(result['filters']['source'], ['prw_image'])
        self.assertEqual(result['thumbnails']['ok'], 10)
        self.assertTrue(all(v['truncated'] for v in result['per_collection'].values()))
        self.assertEqual(result['schema_version'], 1)
        self.assertEqual(result['inputs'][-1]['sha256'], review.file_info('x', out)['sha256'])
        self.assertIn('부분 스캔', out.read_text(encoding='utf-8'))
        with patch.dict(sys.modules, {'qdrant_client': module}), patch.object(db, 'load_thumbs', side_effect=AssertionError('no thumbs')):
            self.call(db.main, args + ['--no-thumbs'])
        result = json.loads(sidecar.read_text(encoding='utf-8'))
        self.assertFalse(result['thumbnails']['enabled'])
        self.assertEqual(result['thumbnails']['sampled'], 0)

    def test_thumb_missing_and_fallback(self):
        rows = [dict(crop_path=str(self.root / 'absent' / self.crops[0].name)), dict(crop_path=str(self.root / 'missing.jpg'))]
        meta = {}
        with contextlib.redirect_stdout(io.StringIO()):
            ok, miss = db.load_thumbs(rows, 32, 75, self.crops[0].parent, meta)
        self.assertEqual((ok, miss, meta['fallback_used']), (1, 1, 1))

    def test_index_checks_links_and_manifest_dates(self):
        self.run_gallery()
        # A skipped target may still contain artifacts from a previous run.
        object_report = self.cluster / 'object' / 'object_leiden_report.json'
        data = json.loads(json.dumps(self.report))
        data['config'].update(target='object', collection='forensic_object', sources=['different'])
        review.write_json(object_report, data)
        review.write_json(object_report.parent / 'gallery' / 'gallery_report.json', dict(assignments=12, html='missing.html'))
        content, result = self.run_index()
        states = {v['name']: v for v in result['checks']}
        self.assertEqual(states['person source']['state'], '확인')
        self.assertEqual(states['person assignments 수']['state'], '확인')
        self.assertEqual(states['person assignments sha256']['state'], '확인')
        self.assertEqual(states['DB HTML 식별']['state'], '확인')
        self.assertEqual(states['object source']['state'], '불일치')
        self.assertEqual(states['object assignments sha256']['state'], '확인 불가')
        self.assertIn('이전 산출물 잔존', states['object summary']['detail'])
        self.assertIn('a_new (최신 확인 시각)', content)
        self.assertIn('manifest 없음', content)
        self.assertIn('입력 stats 수량', content)
        self.assertIn('동일 실행을 증명할 수 없습니다', content)
        expected = quote(os.path.relpath(self.db_html, self.root / 'new out # & % 한글').replace('\\', '/'), safe='/')
        self.assertIn(f'href="{expected}"', content)
        self.assertIn('%23', content)
        self.assertIn('%25', content)
        self.assertIn('%26', content)
        self.assertEqual(result['targets']['person']['assignments']['largest_cluster'], 2)

    def test_index_mismatch_sha_points_and_time(self):
        path, _ = self.run_gallery()
        self.report['stats']['points'] = 99
        review.write_json(self.report_path, self.report)
        os.utime(self.report_path, (path.stat().st_mtime + 100, path.stat().st_mtime + 100))
        self.db_html.write_text('modified', encoding='utf-8')
        _, result = self.run_index()
        states = {v['name']: v['state'] for v in result['checks']}
        for key in ('person assignments 수', 'person gallery 수', 'person report sha256', 'person 시간 순서', 'DB HTML 식별'):
            self.assertEqual(states[key], '불일치')

    def test_index_partial_warnings_and_legacy(self):
        self.db_summary.update(build_ids={}, legacy=6)
        self.db_summary['thumbnails']['missing'] = 2
        self.db_summary['per_collection']['forensic_person']['truncated'] = True
        review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        content, result = self.run_index()
        self.assertEqual(result['status'], 'warning')
        self.assertIn('전부 legacy', content)
        self.assertIn('DB 부분 스캔', content)
        self.assertFalse(any('legacy' in w['message'] for w in result['warnings']))

    def test_index_none_no_markers(self):
        args = ['--db-html', str(self.root / 'none.html'), '--cluster-dir', str(self.root / 'no_cluster'),
                '--manifest-dir', str(self.root / 'none'), '--out', str(self.root / 'not-created' / 'index.html')]
        code, text = self.call(review.main, args)
        self.assertEqual(code, 1)
        self.assertNotIn('RESULT_', text)
        self.assertFalse((self.root / 'not-created').exists())
        review.write_json(self.root / 'none.summary.json', {})
        code, text = self.call(review.main, args)
        self.assertEqual(code, 1)

    def test_index_gallery_override_inline_filename(self):
        path, report = self.run_gallery()
        alternate = self.root / 'other gallery' / 'actual_inline.html'
        alternate.parent.mkdir()
        alternate.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
        report['html'] = str(alternate)
        custom = alternate.parent / 'custom.json'
        review.write_json(custom, report)
        _, result = self.run_index(['--person-gallery-report', str(custom)])
        self.assertEqual(result['links']['person_gallery'], str(alternate))

    def test_cross_drive_fallback_and_url(self):
        with patch.object(review.os.path, 'relpath', side_effect=ValueError('different drive')):
            self.assertEqual(review.path_href(self.crops[0], self.db_html), self.crops[0].as_uri())
        self.assertEqual(review.path_href('https://example.invalid/a?x=1&y=2', self.db_html),
                         'https://example.invalid/a?x=1&amp;y=2')

    def test_inline_entities_percent_external_links(self):
        src = self.root / 'inline source' / 'gallery.html'
        crop = src.parent / 'assets' / '한국 # % & image.jpg'
        crop.parent.mkdir(parents=True)
        crop.write_bytes(self.crops[0].read_bytes())
        ref = 'assets/한국%20%23%20%25%20&amp;%20image.jpg'
        src.write_text(f'<h1>test</h1><a href="{ref}"><img src="{ref}"></a>'
                       '<img src="https://example.invalid/image.jpg"><img src="data:image/png;base64,AA==">', encoding='utf-8')
        out = self.root / 'inline out' / 'result.html'
        code, output = self.call(inline.main, ['--html', str(src), '--out', str(out), '--keep-links'])
        self.assertEqual(code, 0)
        self.assert_markers(output, out.with_suffix('.summary.json'), out)
        result = json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))
        self.assertEqual((result['embedded'], result['missing'], result['external_images']), (1, 0, 1))
        self.assertFalse(result['self_contained'])
        self.assertEqual(result['status'], 'warning')
        self.assertIn('EXTERNAL_IMAGE_REMAINS', [w['code'] for w in result['warnings']])
        self.assertIn(f'href="{review.path_href(crop, out)}"', out.read_text(encoding='utf-8'))
        self.call(inline.main, ['--html', str(src), '--out', str(out), '--drop-links'])
        self.assertNotIn('<a ', out.read_text(encoding='utf-8'))

    def test_inline_missing_keeps_exit_two(self):
        src = self.root / 'missing.html'
        src.write_text('<img src="missing.jpg">', encoding='utf-8')
        code, output = self.call(inline.main, ['--html', str(src)])
        self.assertEqual(code, 2)
        out = src.with_name('missing_inline.html')
        self.assert_markers(output, out.with_suffix('.summary.json'), out)
        result = json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))
        self.assertFalse(result['self_contained'])

    def test_save_failure_has_no_markers(self):
        client, _ = self.fake_client()
        qmodule = SimpleNamespace(QdrantClient=Mock(return_value=client), models=self.models())
        cases = [(review, self.index_args()), (inline, ['--html', str(self.db_html), '--out', str(self.root / 'inline.html')]),
                 (gallery, ['--assignments', str(self.assignments), '--output-dir', str(self.root / 'failed_gallery'), '--inline-images']),
                 (db, ['--out', str(self.root / 'failed_db.html'), '--no-thumbs', '--manifest-dir', str(self.manifest_dir)])]
        for module, args in cases:
            output = io.StringIO()
            with patch.object(module, 'write_json', side_effect=OSError('save failed')), contextlib.redirect_stdout(output), \
                    patch.dict(sys.modules, {'qdrant_client': qmodule}), \
                    patch.object(gallery.QdrantHTTP, 'retrieve_points', return_value=self.payloads):
                with self.assertRaises(OSError):
                    module.main(args)
            self.assertNotIn('RESULT_', output.getvalue())

    def test_index_complete_consistent_inputs(self):
        # Clean fixtures exercise the all-confirmed path, without relying on live artifacts.
        self.assignments.write_text('\n'.join(self.assignments.read_text(encoding='utf-8').splitlines()[:6]) + '\n', encoding='utf-8')
        object_dir = self.cluster / 'object'
        object_dir.mkdir()
        object_assignments = object_dir / 'object_leiden_assignments.jsonl'
        object_assignments.write_bytes(self.assignments.read_bytes())
        object_report = json.loads(json.dumps(self.report))
        object_report['config'].update(target='object', collection='forensic_object', sources=[' prw_image ', 'prw_image'])
        review.write_json(object_dir / 'object_leiden_report.json', object_report)
        review.write_json(self.cluster / 'leiden_pipeline_summary.json', [self.report, object_report])
        self.db_summary['build_ids'].pop('no_manifest')
        self.db_summary['legacy'] = 2
        review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        self.run_gallery()
        args = self.gui_argv('image_gallery_object', {'--assignments': object_assignments, '--output-dir': object_dir / 'gallery'})
        with patch.object(gallery.QdrantHTTP, 'retrieve_points', return_value=self.payloads):
            code, _ = self.call(gallery.main, args)
        self.assertEqual(code, 0)
        _, result = self.run_index()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['warnings'], [])
        self.assertTrue(all(c['state'] == '확인' for c in result['checks']))

    def test_index_unknown_identity_and_summary_absent_target(self):
        self.run_gallery()
        self.db_summary.pop('inputs')
        review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        review.write_json(self.cluster / 'leiden_pipeline_summary.json', [])
        _, result = self.run_index()
        checks = {c['name']: c for c in result['checks']}
        self.assertEqual(checks['DB HTML 식별']['state'], '확인 불가')
        self.assertIn('이번 summary 에 없음', checks['person summary']['detail'])

    def test_db_missing_warning_and_scan_zero(self):
        self.payloads['p0']['crop_path'] = str(self.root / 'no_image.jpg')
        client, _ = self.fake_client(2)
        module = SimpleNamespace(QdrantClient=Mock(return_value=client), models=self.models())
        out = self.root / 'missing_db.html'
        with patch.dict(sys.modules, {'qdrant_client': module}):
            code, text = self.call(db.main, ['--out', str(out), '--manifest-dir', str(self.manifest_dir)])
        result = json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 0)
        self.assertIn('[경고]', text)
        self.assertEqual(result['thumbnails']['missing'], 4)
        self.assertIn('THUMBNAILS_MISSING', [w['code'] for w in result['warnings']])
        client, _ = self.fake_client(0)
        meta = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rows, count = db.scroll_collection(client, self.models(), 'x', 'image', None, 1, 1, meta)
        self.assertEqual((rows, count), ([], 0))
        self.assertEqual(meta['stop_reason'], 'exhausted')

    def test_schema_common_fields_all_producers(self):
        gallery_html, gallery_report = self.run_gallery()
        _, index_report = self.run_index()
        out = self.root / 'self_contained.html'
        self.call(inline.main, ['--html', str(gallery_html), '--out', str(out)])
        inline_report = json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))
        self.assertTrue(inline_report['self_contained'])
        for result in (gallery_report, index_report, inline_report):
            self.assertEqual(result['schema_version'], 1)
            self.assertIsNotNone(review.datetime.fromisoformat(result['generated_at']).tzinfo)
            self.assertTrue(Path(result['out_html']).is_absolute())
            for item in result['inputs']:
                for key in ('role', 'path', 'exists', 'size'):
                    self.assertIn(key, item)
                if item['exists']:
                    self.assertEqual(len(item['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
