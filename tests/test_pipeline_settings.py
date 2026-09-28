"""Bounded settings/CLI tests: temporary YAML, fake clients, no live database."""
import ast
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import report_common as common
from tests import test_image_review_html as existing


def spec(scope, text=False):
    return dict(scope=scope, supports_text=text, dim=768, module='fake', **{'class': 'Fake'})


def fixture():
    return dict(collection_prefix='custom', qdrant={'url': 'http://yaml.invalid:6333'},
                retrievers={'solider': spec('person'), 'dinov2': spec('object'), 'irra': spec('person', True)})


class SettingsTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket', 'socket.create_connection', 'socket.getaddrinfo',
                     'requests.Session.request'):
            guard = patch(name, side_effect=AssertionError('NETWORK FORBIDDEN'))
            guard.start()
            self.addCleanup(guard.stop)
        temp = tempfile.TemporaryDirectory(prefix='pipeline_settings_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'pipeline.yaml'
        self.raw = fixture()

    def load(self, raw=None, bom=False):
        content = json.dumps(self.raw if raw is None else raw).encode('utf-8')
        self.path.write_bytes((b'\xef\xbb\xbf' if bom else b'') + content)
        return common.load_pipeline_settings(self.path)

    def test_defaults_and_inferred(self):
        settings = self.load()
        self.assertEqual(settings.clustering, dict(person=dict(vector='solider', threshold=.97),
                         object=dict(vector='dinov2', threshold=.9), secondary_vector='irra', sources=None))
        self.assertEqual(settings.collection_for('person'), 'custom_person')
        self.assertEqual(settings.collection_for('object'), 'custom_object')
        self.assertEqual(settings.qdrant_url, 'http://yaml.invalid:6333')
        self.assertTrue({'person.vector', 'person.threshold', 'object.vector', 'object.threshold',
                         'secondary_vector'}.issubset(settings.inferred))
        self.assertEqual(settings.summary()['config_path'], str(self.path.resolve()))
        self.assertNotIn('retrievers', settings.summary())
        self.assertEqual(settings.retrievers['solider'], dict(scope='person', dim=768, supports_text=False))

    def test_explicit_clustering_and_sources(self):
        self.raw['clustering'] = dict(person=dict(vector='irra', threshold=0),
            object=dict(vector='dinov2', threshold='0.81'), secondary_vector='solider', sources=' B, A,B, ')
        settings = self.load()
        self.assertEqual(settings.vector_for('person'), 'irra')
        self.assertEqual(settings.threshold_for('person'), 0)
        self.assertEqual(settings.threshold_for('object'), .81)
        self.assertEqual(settings.clustering['sources'], ['B', 'A'])
        self.assertEqual(settings.inferred, {})
        self.raw['clustering']['sources'] = ['A', ' A ', 'B']
        self.assertEqual(self.load().clustering['sources'], ['A', 'B'])

    def test_new_person_retriever_and_no_secondary(self):
        self.raw['retrievers'] = {'newreid': spec('person'), 'dinov2': spec('object')}
        settings = self.load()
        self.assertEqual(settings.vector_for('person'), 'newreid')
        self.assertIsNone(settings.clustering['secondary_vector'])
        self.assertIn('newreid', settings.inferred['person.vector'])

    def test_all_scope_and_fallback_order(self):
        self.raw['retrievers'] = {'universal': spec('all', True)}
        settings = self.load()
        self.assertEqual(settings.vector_for('person'), 'universal')
        self.assertEqual(settings.vector_for('object'), 'universal')
        self.raw['retrievers'].update(text=spec('person', True), newreid=spec('person'), obj=spec('object'))
        settings = self.load()
        self.assertEqual(settings.vector_for('person'), 'newreid')
        self.assertEqual(settings.vector_for('object'), 'obj')
        self.raw['retrievers']['solider'] = spec('object')
        self.raw['retrievers']['dinov2'] = spec('person')
        self.assertEqual(self.load().vector_for('person'), 'newreid')
        self.assertEqual(self.load().vector_for('object'), 'obj')

    def test_bad_clustering_messages(self):
        cases = [({'person': {'vector': 'typo'}}, 'person.vector.*typo.*후보'),
                 ({'person': {'vector': 'dinov2'}}, 'person.vector.*dinov2.*후보'),
                 ({'object': {'vector': 'solider'}}, 'object.vector.*solider.*후보'),
                 ({'secondary_vector': 'dinov2'}, 'secondary_vector.*dinov2.*후보'),
                 ({'secondary_vector': 'typo'}, 'secondary_vector.*typo.*후보'),
                 ({'person': {'threshold': 2.0}}, 'person.threshold.*finite'),
                 ({'object': {'threshold': float('nan')}}, 'object.threshold.*finite'),
                 ({'object': {'threshold': float('inf')}}, 'object.threshold.*finite'),
                 ({'person': {'threshold': True}}, 'person.threshold.*finite'),
                 ({'surprise': 1}, 'clustering.*unknown.*surprise'),
                 ({'person': {'surprise': 1}}, 'clustering.person.*unknown.*surprise'),
                 ({'person': []}, 'clustering.person.*mapping'),
                 ([], 'clustering.*mapping'),
                 ({'sources': []}, 'sources.*empty'),
                 ({'sources': ' , '}, 'sources.*empty'),
                 ({'sources': None}, 'sources.*list'),
                 ({'sources': {}}, 'sources.*list')]
        for section, message in cases:
            with self.subTest(section=section):
                self.raw['clustering'] = section
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_missing_candidate_and_pipeline_validation(self):
        self.raw['retrievers'] = {'newreid': spec('person')}
        with self.assertRaisesRegex(ValueError, 'object.vector.*후보'):
            self.load()
        self.raw = fixture()
        self.raw['retrievers']['solider']['dim'] = 0
        with self.assertRaisesRegex(ValueError, 'dim'):
            self.load()

    def test_missing_invalid_and_permission_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, 'pipeline.yaml'):
            common.load_pipeline_settings(self.path)
        self.assertIsNone(common.load_pipeline_settings(self.path, require=False))
        import yaml
        self.path.write_text('retrievers: [broken', encoding='utf-8')
        with self.assertRaises(yaml.YAMLError):
            common.load_pipeline_settings(self.path, require=False)
        self.path.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mapping'):
            common.load_pipeline_settings(self.path, require=False)
        with patch.object(Path, 'read_bytes', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                common.load_pipeline_settings(self.path, require=False)

    def test_bom_hash_one_read_and_cwd_independent_default(self):
        settings = self.load(bom=True)
        content = self.path.read_bytes()
        self.assertEqual(settings.config_sha256, hashlib.sha256(content).hexdigest())
        self.assertNotEqual(settings.config_sha256, hashlib.sha256(content[3:]).hexdigest())
        with patch.object(Path, 'read_bytes', return_value=content) as read:
            common.load_pipeline_settings(self.path)
        self.assertEqual(read.call_count, 1)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch.object(common, 'DEFAULT_CONFIG_PATH', self.path):
                self.assertEqual(common.load_pipeline_settings().config_path, settings.config_path)
            self.assertEqual(common.DEFAULT_CONFIG_PATH, Path(common.__file__).resolve().parent / 'pipeline.yaml')
        finally:
            os.chdir(previous)

    def test_resolve_and_applied(self):
        self.assertEqual(common.resolve(0, .95, 'threshold'), 0)
        self.assertEqual(common.resolve('', 'fallback', 'name'), 'fallback')
        self.assertEqual(common.resolve(None, 0, 'threshold'), 0)
        with self.assertRaisesRegex(ValueError, 'name: --config 또는 플래그로 지정'):
            common.resolve(None, None, 'name')
        settings = self.load()
        applied = common.applied_config(settings, person_collection='override')
        self.assertEqual(applied['applied']['person_collection'], 'override')
        self.assertEqual(applied['settings']['person_collection'], 'custom_person')
        self.assertEqual(common.applied_config(None, value=0), dict(settings=None, applied={'value': 0}))

    def test_helper_imports_are_standard_library_and_reexports_identical(self):
        tree = ast.parse(Path(common.__file__).read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.assertIn(name.name.split('.')[0], sys.stdlib_module_names)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split('.')[0], sys.stdlib_module_names | {'__future__'})
        for name in ('normalize_sources', 'write_json', 'path_href', 'assignment_stats', 'size_histogram'):
            self.assertIs(getattr(existing.review, name), getattr(common, name))
        self.assertIs(existing.review.os.path, os.path)


class SettingsIntegrationTests(unittest.TestCase):
    # Reuse only fixture helpers, without rerunning/inheriting the original test cases.
    call = existing.OfflineReviewTests.call
    fake_client = existing.OfflineReviewTests.fake_client
    models = staticmethod(existing.OfflineReviewTests.models)
    gui_argv = existing.OfflineReviewTests.gui_argv
    assert_markers = existing.OfflineReviewTests.assert_markers
    run_gallery = existing.OfflineReviewTests.run_gallery
    run_index = existing.OfflineReviewTests.run_index
    index_args = existing.OfflineReviewTests.index_args

    def setUp(self):
        existing.OfflineReviewTests.setUp(self)
        # Import clustering only after the network guards have started.
        self.leiden = importlib.import_module('clustering.cluster_leiden_qdrant')
        self.custom_path = self.root / 'custom.yaml'
        self.custom_raw = fixture()
        self.custom_path.write_text(json.dumps(self.custom_raw), encoding='utf-8')
        self.custom = common.load_pipeline_settings(self.custom_path)

    def db_run(self, extra):
        client, _ = self.fake_client()
        client.scroll = Mock(wraps=client.scroll)
        module = SimpleNamespace(QdrantClient=Mock(return_value=client), models=self.models())
        out = self.root / 'settings_db.html'
        argv = ['--config', str(self.custom_path), '--out', str(out), '--no-thumbs',
                '--manifest-dir', str(self.manifest_dir)] + extra
        with patch.dict(sys.modules, {'qdrant_client': module}):
            code, text = self.call(existing.db.main, argv)
        self.assertEqual(code, 0)
        report = json.loads(out.with_suffix('.summary.json').read_text(encoding='utf-8'))
        self.assert_markers(text, out.with_suffix('.summary.json'), out)
        self.assertEqual(report['inputs'][-1]['role'], 'out_html')
        self.assertEqual(module.QdrantClient.call_args.kwargs['url'], report['config']['applied']['qdrant_url'])
        self.assertEqual({c.kwargs['collection_name'] for c in client.scroll.call_args_list}, set(report['collections']))
        return report

    def test_db_yaml_and_cli_override(self):
        report = self.db_run([])
        self.assertEqual(report['config']['applied'], dict(person_collection='custom_person',
                         object_collection='custom_object', qdrant_url='http://yaml.invalid:6333'))
        self.assertEqual(report['config']['settings'], self.custom.summary())
        report = self.db_run(['--person-collection', 'X'])
        self.assertEqual(report['config']['applied']['person_collection'], 'X')
        self.assertEqual(report['config']['settings']['person_collection'], 'custom_person')
        self.assertIn('custom.yaml', Path(report['out_html']).read_text(encoding='utf-8'))

    def test_db_missing_config_full_cli_only(self):
        missing = str(self.root / 'absent.yaml')
        report = self.db_run(['--config', missing, '--person-collection', 'P', '--object-collection', 'O',
                              '--qdrant-url', 'http://cli.invalid'])
        self.assertIsNone(report['config']['settings'])
        self.assertIn('CONFIG_UNAVAILABLE', [v['code'] for v in report['warnings']])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            existing.db.main(['--config', missing, '--person-collection', 'P'])
        self.assertEqual(error.exception.code, 2)

    def test_argument_errors_precede_yaml_reads(self):
        for module, argv in ((existing.db, ['--samples', '-1']),
                             (existing.gallery, ['--images-per-cluster', '0']),
                             (self.leiden, ['--person-threshold', '2']),
                             (self.leiden, ['--knn', '0'])):
            with self.subTest(module=module.__name__), patch.object(module, 'load_pipeline_settings') as loader, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                module.main(argv)
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()

    def test_cluster_resolve_defaults_overrides_and_purity(self):
        args = self.leiden.parse_args([])
        original = copy.deepcopy(vars(args))
        targets = self.leiden.resolve_targets(args, self.custom)
        self.assertEqual(vars(args), original)
        self.assertEqual(targets, dict(person=dict(collection='custom_person', vector='solider', threshold=.97),
                                       object=dict(collection='custom_object', vector='dinov2', threshold=.9)))
        args = self.leiden.parse_args(['--person-collection', 'P', '--person-vector', 'irra', '--person-threshold', '0'])
        self.assertEqual(self.leiden.resolve_targets(args, self.custom)['person'],
                         dict(collection='P', vector='irra', threshold=0))
        for flag, value in (('--person-vector', 'typo'), ('--object-vector', 'solider'), ('--secondary-vector', 'dinov2')):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, '후보'):
                self.leiden.resolve_targets(self.leiden.parse_args([flag, value]), self.custom)

    def test_cluster_identity_safe_after_merge(self):
        args = self.leiden.parse_args(['--target', 'person', '--identity-safe'])
        self.assertEqual(self.leiden.resolve_targets(args, self.custom)['person']['vector'], 'solider')
        args.person_vector = 'irra'
        with self.assertRaisesRegex(ValueError, 'must differ'):
            self.leiden.resolve_targets(args, self.custom)
        args.secondary_vector = 'solider'
        self.assertEqual(self.leiden.resolve_targets(args, self.custom)['person']['vector'], 'irra')
        settings = copy.deepcopy(self.custom)
        settings.clustering['secondary_vector'] = None
        args.person_vector = args.secondary_vector = None
        with self.assertRaisesRegex(ValueError, 'requires secondary-vector'):
            self.leiden.resolve_targets(args, settings)

    def cluster_run(self, extra):
        captured = []
        def run(args, target):
            captured.append((target, copy.deepcopy(args.resolved_targets[target]), args.qdrant_url,
                             list(args.sources), args.secondary_vector))
            return {'target': target, 'config': self.leiden.config_provenance(args, target)}
        with patch.object(self.leiden, 'run_target', side_effect=run), \
                patch.object(self.leiden, 'Qdrant', side_effect=AssertionError('NO QDRANT')):
            self.leiden.main(['--config', str(self.config_path), '--output-dir', str(self.root / 'cluster_config')] + extra)
        report = json.loads((self.root / 'cluster_config' / 'leiden_pipeline_summary.json').read_text(encoding='utf-8'))
        self.assertIsInstance(report, list)
        return captured, report

    def test_cluster_cli_equivalence_sources_and_missing_config(self):
        explicit = ['--person-collection', 'forensic_person', '--person-vector', 'solider', '--person-threshold', '.97',
                    '--object-collection', 'forensic_object', '--object-vector', 'dinov2', '--object-threshold', '.9',
                    '--qdrant-url', 'http://localhost:6333', '--secondary-vector', 'irra']
        implicit, report = self.cluster_run([])
        full, _ = self.cluster_run(explicit)
        self.assertEqual(full, implicit)
        missing, reports = self.cluster_run(explicit + ['--config', str(self.root / 'missing.yaml')])
        self.assertEqual(missing, implicit)
        self.assertIsNone(reports[0]['config']['config_sha256'])
        self.assertIn('확인 불가', reports[0]['config']['retriever_validation'])
        self.assertEqual(report[0]['config']['config_path'], str(self.config_path))
        self.custom_raw['clustering'] = {'sources': 'yaml_A, yaml_B', 'person': {'threshold': .72}}
        self.custom_path.write_text(json.dumps(self.custom_raw), encoding='utf-8')
        calls, _ = self.cluster_run(['--config', str(self.custom_path)])
        self.assertEqual(calls[0][3], ['yaml_A', 'yaml_B'])
        self.assertEqual(calls[0][1]['threshold'], .72)
        calls, _ = self.cluster_run(['--config', str(self.custom_path), '--sources', 'cli_A'])
        self.assertEqual(calls[0][3], ['cli_A'])

    def test_gui_cluster_optional_thresholds_and_url(self):
        overrides = {'--output-dir': self.root / 'cluster_config'}
        current = self.gui_argv('image_cluster', overrides)
        omitted = self.gui_argv('image_cluster', {
            **overrides, '--person-threshold': '', '--object-threshold': '', '--qdrant-url': ''})
        for flag in ('--person-threshold', '--object-threshold', '--qdrant-url'):
            self.assertNotIn(flag, omitted)
        self.assertEqual(self.cluster_run(current)[0], self.cluster_run(omitted)[0])

    def test_cluster_invalid_vector_before_client(self):
        with patch.object(self.leiden, 'Qdrant') as client, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            self.leiden.main(['--config', str(self.custom_path), '--person-vector', 'typo'])
        self.assertEqual(error.exception.code, 2)
        client.assert_not_called()

    def test_cluster_skipped_provenance(self):
        client = Mock()
        client.get.return_value = {'result': {}}
        client.scroll_ids.return_value = []
        folder = self.root / 'skipped'
        with patch.object(self.leiden, 'Qdrant', return_value=client), contextlib.redirect_stdout(io.StringIO()):
            self.leiden.main(['--config', str(self.custom_path), '--output-dir', str(folder)])
        reports = json.loads((folder / 'leiden_pipeline_summary.json').read_text(encoding='utf-8'))
        for report in reports:
            self.assertTrue(report['skipped'])
            self.assertEqual(report['config_path'], str(self.custom_path))
            self.assertEqual(report['config_sha256'], self.custom.config_sha256)
            self.assertEqual(report['sources'], ['final_db_candidates', 'forensic_image'])
            self.assertEqual(report['collection'], 'custom_' + report['target'])

    def test_cluster_nonempty_report_provenance_without_algorithms(self):
        client = Mock()
        client.get.return_value = {'result': {}}
        client.scroll_ids.return_value = ['p0']
        folder = self.root / 'nonempty'
        with patch.object(self.leiden, 'Qdrant', return_value=client), \
                patch.object(self.leiden, 'build_graph', return_value=([], [], {'points': 1})), \
                patch.object(self.leiden, 'leiden', return_value=([0], {})), \
                patch.object(self.leiden, 'normalize', return_value=({'p0': {'cluster_id': None}}, {})), \
                patch.object(self.leiden, 'write_payloads') as write, contextlib.redirect_stdout(io.StringIO()):
            self.leiden.main(['--config', str(self.custom_path), '--target', 'person', '--dry-run',
                              '--person-threshold', '0', '--output-dir', str(folder)])
        write.assert_not_called()
        report = json.loads((folder / 'person' / 'person_leiden_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['config']['config_path'], str(self.custom_path))
        self.assertEqual(report['config']['config_sha256'], self.custom.config_sha256)
        self.assertEqual(report['config']['settings_inferred'], self.custom.inferred)
        self.assertEqual(report['config']['applied']['threshold'], 0)
        self.assertEqual(report['config']['score_threshold'], 0)
        self.assertEqual(report['config']['applied']['qdrant_url'], self.custom.qdrant_url)

    def gallery_run(self, extra, assignments=None):
        folder = self.root / 'settings_gallery'
        argv = ['--config', str(self.custom_path), '--assignments', str(assignments or self.assignments),
                '--output-dir', str(folder), '--inline-images'] + extra
        with patch.object(existing.gallery, 'QdrantHTTP') as client:
            client.return_value.retrieve_points.return_value = self.payloads
            code, _ = self.call(existing.gallery.main, argv)
        self.assertEqual(code, 0)
        report = json.loads((folder / 'gallery_report.json').read_text(encoding='utf-8'))
        applied = report['config']['applied']
        self.assertEqual(client.call_args.args[0], applied['qdrant_url'])
        self.assertEqual(client.return_value.retrieve_points.call_args.args[0], applied['collection'])
        return report

    def test_gallery_target_inference_override_and_explicit_collection(self):
        report = self.gallery_run([])
        self.assertEqual(report['collection'], 'custom_person')
        self.assertEqual(report['config']['applied']['target'], 'person')
        report = self.gallery_run(['--target', 'object'])
        self.assertEqual(report['collection'], 'custom_object')
        self.assertEqual(report['target'], 'object')
        unknown = self.root / 'assignments.jsonl'
        unknown.write_bytes(self.assignments.read_bytes())
        with patch.object(existing.gallery, 'load_pipeline_settings') as loader, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            existing.gallery.main(['--assignments', str(unknown)])
        self.assertEqual(error.exception.code, 2)
        loader.assert_not_called()
        report = self.gallery_run(['--collection', 'EXPLICIT'], unknown)
        self.assertEqual(report['collection'], 'EXPLICIT')
        self.assertIsNone(report['target'])

    def test_gui_explicit_and_empty_omitted_values_equivalent(self):
        for step, module, flags in (
            ('image_db_html', existing.db, ['--person-collection', '--object-collection', '--qdrant-url']),
            ('image_gallery_person', existing.gallery, ['--collection', '--qdrant-url'])):
            old = self.gui_argv(step)
            new = self.gui_argv(step, dict.fromkeys(flags, ''))
            self.assertTrue(all(flag not in new for flag in flags))
            parser = module.build_parser()
            old_args, new_args = parser.parse_args(old), parser.parse_args(new)
            for flag in flags:
                name = flag[2:].replace('-', '_')
                value = self.settings.person_collection if name == 'collection' else getattr(self.settings, name)
                self.assertEqual(common.resolve(getattr(old_args, name), value, name),
                                 common.resolve(getattr(new_args, name), value, name))
        implicit = self.db_run([])['config']['applied']
        explicit = self.db_run(['--person-collection', 'custom_person', '--object-collection', 'custom_object',
                                '--qdrant-url', 'http://yaml.invalid:6333'])['config']['applied']
        self.assertEqual(implicit, explicit)
        self.assertEqual(self.gallery_run([])['config']['applied'], self.gallery_run(
            ['--collection', 'custom_person', '--qdrant-url', 'http://yaml.invalid:6333'])['config']['applied'])

    def test_index_config_path_and_content_three_states(self):
        self.run_gallery()
        _, result = self.run_index()
        def states(result):
            return {v['name']: v for v in result['checks'] if v['name'].startswith('config ')}
        self.assertEqual({v['state'] for v in states(result).values()}, {'확인'})
        self.db_summary['config']['settings']['config_sha256'] = 'f' * 64
        existing.review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        content, result = self.run_index()
        self.assertEqual(states(result)['config 경로 일치']['state'], '확인')
        self.assertEqual(states(result)['config 내용 일치']['state'], '불일치')
        self.assertIn('ffffffffffff', content)
        self.assertIn('pipeline.yaml', content)
        self.db_summary['config']['settings']['config_sha256'] = self.settings.config_sha256
        self.db_summary['config']['settings']['config_path'] = str(self.root / 'elsewhere.yaml')
        existing.review.write_json(self.db_html.with_suffix('.summary.json'), self.db_summary)
        _, result = self.run_index()
        self.assertEqual(states(result)['config 경로 일치']['state'], '불일치')
        self.assertEqual(states(result)['config 내용 일치']['state'], '확인')
        for path in (self.db_html.with_suffix('.summary.json'), self.report_path,
                     self.cluster / 'person' / 'gallery' / 'gallery_report.json'):
            value = json.loads(path.read_text(encoding='utf-8'))
            if path == self.report_path:
                value['config'].pop('config_path')
                value['config'].pop('config_sha256')
            else:
                value.pop('config')
            existing.review.write_json(path, value)
        _, result = self.run_index()
        self.assertEqual({v['state'] for v in states(result).values()}, {'확인 불가'})
        self.assertTrue(all('구형 산출물' in v['detail'] for v in states(result).values()))

    def test_index_legacy_single_provenance_is_unknown(self):
        # Only one recorded artifact is insufficient to establish agreement.
        self.report['config'].pop('config_path')
        self.report['config'].pop('config_sha256')
        existing.review.write_json(self.report_path, self.report)
        _, result = self.run_index()
        checks = [v for v in result['checks'] if v['name'].startswith('config ')]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(v['state'] == '확인 불가' for v in checks))
        self.assertTrue(all('구형 산출물' in v['detail'] for v in checks))


if __name__ == '__main__':
    unittest.main()
