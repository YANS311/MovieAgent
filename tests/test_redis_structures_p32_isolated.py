"""Command class AST + stdlib/Fake only; never run Django or the command CLI."""
import argparse
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

FORBIDDEN = {'torch', 'transformers', 'sentence_transformers', 'numpy', 'faiss',
             'tensorflow', 'redis', 'django', 'MySQLdb', 'neo4j'}


class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Forbidden import: ' + fullname)


sys.meta_path.insert(0, ImportGuard())
ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = ROOT / 'myapp/management/commands/consume_redis_feedback.py'
spec = importlib.util.spec_from_file_location('p32_p31_fixture', ROOT / 'tests/test_redis_structures_p31_isolated.py')
p31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p31)
analytics = p31.analytics


class CommandError(Exception):
    pass


class BaseCommand:
    """Only the stdout surface: NOT Django's initialization or execute()."""
    def __init__(self):
        self.stdout = io.StringIO()


def command_class(settings, factory):
    source = COMMAND_PATH.read_text()
    if p31.unbound_globals(source):
        raise AssertionError('Production command has unbound globals')
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert any(n.module == 'myapp.services.redis_analytics' and
               [a.name for a in n.names] == ['analytics_from_settings'] for n in imports)
    assert callable(analytics.analytics_from_settings)  # Actual helper, never an invented export.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Command')
    scope = {'BaseCommand': BaseCommand, 'CommandError': CommandError,
             'settings': settings, 'analytics_from_settings': factory,
             'json': json, 'time': __import__('time')}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(COMMAND_PATH), 'exec'), scope)
    return scope['Command']


def empty_report(**changes):
    report = dict.fromkeys(('processed', 'duplicates', 'acked', 'failed', 'prune_failed',
                           'claimed', 'fresh', 'deleted_pending', 'deferred'), 0)
    report.update(reclaim_cursor='0-0', next_lane='fresh', failed_ids=[])
    report.update(changes)
    return report


class FakeRedis(p31.p3.FakeRedis):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class P32Tests(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        self.sleep = Mock(side_effect=lambda delay: self.clock.__setitem__(0, self.clock[0] + delay))
        self.settings = types.SimpleNamespace(REDIS_ANALYTICS_ENABLED=True,
            REDIS_ANALYTICS_URL='redis://localhost:6379/2',
            CACHES={'default': {'LOCATION': 'redis://localhost:6379/0'}})
        self.redis = FakeRedis()
        self.service = analytics.FeedbackAnalytics(self.redis)
        self.factory = Mock(return_value=self.service)

    def run_command(self, service=None, factory=None, **options):
        factory = factory if factory is not None else self.factory
        if service is not None:
            factory.return_value = service
        command = command_class(self.settings, factory)()
        options = {'count': 100, 'max_polls': 4, 'max_seconds': 30, **options}
        with patch.object(analytics.time, 'monotonic', side_effect=lambda: self.clock[0]), \
             patch.object(analytics.time, 'sleep', self.sleep):
            command.handle(**options)
        return command, json.loads(command.stdout.getvalue())

    def fake_consumer(self, report=None):
        return types.SimpleNamespace(client=types.SimpleNamespace(close=Mock()),
            consume_once=Mock(side_effect=lambda **options: dict(report or empty_report())))

    def event(self, event_id):
        return {'event_id': str(event_id), 'movie_id': '9', 'feedback_type': 'click',
                'event_timestamp': '2026-10-14T12:00:00.000000+00:00', 'schema_version': '1'}

    def seed(self, pending_count, fresh_count, poison=True):
        mids = [self.service.publish(self.event(i + 1)) for i in range(pending_count)]
        self.service.ensure_group()
        self.redis.xreadgroup(analytics.GROUP, 'previous', {analytics.STREAM_KEY: '>'}, count=pending_count)
        if poison:
            for mid in mids: self.redis.stream[mid]['feedback_type'] = 'invalid'
        self.redis.now = 61
        fresh = [self.service.publish(self.event(1000 + i)) for i in range(fresh_count)]
        self.redis.calls.clear()
        return mids, fresh

    def test_factory_once_and_four_polls_reuse_one_consumer(self):
        fake = self.fake_consumer(empty_report(fresh=1, processed=1, acked=1))
        _, summary = self.run_command(service=fake, count=1, max_polls=4)
        self.factory.assert_called_once_with(self.settings)
        self.assertEqual(fake.consume_once.call_count, 4)
        self.assertEqual([c.kwargs for c in fake.consume_once.call_args_list],
                         [{'count': 1, 'max_seconds': 10}] * 4)
        self.assertEqual((summary['polls'], summary['acked'], summary['exit_reason']), (4, 4, 'poll_limit'))
        fake.client.close.assert_called_once()
        self.sleep.assert_not_called()

    def test_count_one_poison_pending_fresh_progress_and_cursor_retained(self):
        poison, fresh = self.seed(2, 2)
        with patch.object(analytics.logger, 'error') as logs:
            _, summary = self.run_command(count=1)
        self.assertEqual((summary['polls'], summary['failed'], summary['acked']), (4, 2, 2))
        self.assertEqual([c[1] for c in self.redis.calls if c[0] == 'xack'], fresh)
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), set(poison))
        claims = [c for c in self.redis.calls if c[0] == 'xautoclaim']
        self.assertEqual(claims, [('xautoclaim', 1, '0-0'), ('xautoclaim', 1, '2-0')])
        self.assertEqual([c[0] for c in self.redis.calls if c[0] in ('xautoclaim', 'xreadgroup')],
                         ['xautoclaim', 'xreadgroup', 'xautoclaim', 'xreadgroup'])
        self.assertEqual(logs.call_count, 2)
        self.assertEqual(self.redis.close_calls, 1)
        self.factory.assert_called_once()

    def test_count_two_both_lanes_in_each_poll(self):
        poison, fresh = self.seed(2, 2)
        with patch.object(analytics.logger, 'error'):
            _, summary = self.run_command(count=2, max_polls=2)
        self.assertEqual((summary['claimed'], summary['fresh'], summary['failed'], summary['acked']), (2, 2, 2, 2))
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), set(poison))
        self.assertEqual([c[1] for c in self.redis.calls if c[0] == 'xack'], fresh)
        self.assertEqual(self.factory.call_count, 1)

    def test_count_hundred_full_poison_keeps_fresh_quota(self):
        poison, fresh = self.seed(100, 100)
        with patch.object(analytics.logger, 'error'):
            _, summary = self.run_command(count=100, max_polls=2)
        self.assertEqual((summary['claimed'], summary['fresh'], summary['failed'], summary['acked']), (100, 100, 100, 100))
        self.assertEqual([c[1] for c in self.redis.calls if c[0] == 'xack'], fresh)
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), set(poison))
        self.assertEqual([c[2] for c in self.redis.calls if c[0] == 'xautoclaim'], ['0-0', '51-0'])

    def test_poll_limit_is_exact_and_not_infinite(self):
        fake = self.fake_consumer()
        _, summary = self.run_command(service=fake, max_polls=3)
        self.assertEqual(fake.consume_once.call_count, 3)
        self.assertEqual((summary['polls'], summary['exit_reason']), (3, 'poll_limit'))
        self.assertEqual(self.sleep.call_count, 2)

    def test_time_budget_exits_without_rounding_up_remaining(self):
        fake = self.fake_consumer()
        def consume(**options):
            self.clock[0] += 1.25
            return empty_report(fresh=1, processed=1, acked=1)
        fake.consume_once.side_effect = consume
        _, summary = self.run_command(service=fake, max_polls=10, max_seconds=3)
        self.assertEqual(fake.consume_once.call_count, 2)
        self.assertEqual([c.kwargs['max_seconds'] for c in fake.consume_once.call_args_list], [3, 1.75])
        self.assertEqual((summary['polls'], summary['exit_reason']), (2, 'time_budget'))
        self.assertEqual(self.clock[0], 2.5)

    def test_initialization_time_counts_toward_total_budget(self):
        fake = self.fake_consumer()
        def factory(settings):
            self.clock[0] = 2.5
            return fake
        _, summary = self.run_command(factory=Mock(side_effect=factory), max_seconds=3)
        fake.consume_once.assert_not_called()
        self.assertEqual(summary['exit_reason'], 'time_budget')
        fake.client.close.assert_called_once()

    def test_empty_poll_sleeps_bounded_and_does_not_spin(self):
        fake = self.fake_consumer()
        _, summary = self.run_command(service=fake, max_polls=4)
        self.assertEqual(self.sleep.call_args_list, [unittest.mock.call(0.1)] * 3)
        self.assertEqual(summary['polls'], 4)
        self.assertAlmostEqual(self.clock[0], 0.3)

    def test_empty_sleep_is_capped_by_remaining_time(self):
        fake = self.fake_consumer()
        def consume(**options):
            self.clock[0] = 1.95
            return empty_report()
        fake.consume_once.side_effect = consume
        _, summary = self.run_command(service=fake, max_seconds=2)
        self.assertAlmostEqual(self.sleep.call_args.args[0], 0.05)
        self.assertEqual(summary['polls'], 1)
        self.assertEqual(summary['exit_reason'], 'time_budget')

    def test_disabled_no_factory_or_connection(self):
        self.settings.REDIS_ANALYTICS_ENABLED = False
        with self.assertRaisesRegex(CommandError, 'ENABLED'):
            self.run_command()
        self.factory.assert_not_called()
        self.assertEqual(self.redis.calls, [])

    def test_missing_connection_refused_before_factory(self):
        for url in (None, '', 7):
            self.settings.REDIS_ANALYTICS_URL = url
            with self.assertRaisesRegex(CommandError, 'URL required'):
                self.run_command()
        self.factory.assert_not_called()
        self.assertEqual(self.redis.calls, [])

    def test_real_factory_shared_session_db_is_rejected_without_network(self):
        self.settings.REDIS_ANALYTICS_URL = 'redis://127.0.0.1:6379/0'
        connection = Mock()
        factory = Mock(side_effect=lambda settings: analytics.analytics_from_settings(settings, factory=connection))
        with self.assertRaisesRegex(CommandError, 'configuration'):
            self.run_command(factory=factory)
        connection.assert_not_called()
        self.assertEqual(self.redis.calls, [])

    def test_factory_failure_does_not_expose_credentials(self):
        self.factory.side_effect = ConnectionError('redis://user:secret@host/2 raw-event')
        with self.assertRaises(CommandError) as caught:
            self.run_command()
        self.assertNotIn('secret', str(caught.exception))
        self.assertNotIn('raw-event', str(caught.exception))
        self.factory.assert_called_once()
        self.assertEqual(self.redis.calls, [])

    def test_disabled_after_validation_is_refused(self):
        self.factory.return_value = None
        with self.assertRaisesRegex(CommandError, 'disabled during initialization'):
            self.run_command()
        self.factory.assert_called_once()

    def test_infrastructure_failure_exits_once_with_failure_summary(self):
        fake = self.fake_consumer()
        fake.consume_once.side_effect = ConnectionError('user:secret raw event')
        command = command_class(self.settings, self.factory)()
        self.factory.return_value = fake
        with patch.object(analytics.time, 'monotonic', return_value=0), \
             patch.object(analytics.time, 'sleep', self.sleep), self.assertRaises(CommandError) as caught:
            command.handle(count=1, max_polls=1000, max_seconds=300)
        fake.consume_once.assert_called_once()
        fake.client.close.assert_called_once()
        self.sleep.assert_not_called()
        summary = json.loads(command.stdout.getvalue())
        self.assertEqual((summary['polls'], summary['acked'], summary['exit_reason']), (0, 0, 'infrastructure_error'))
        self.assertNotIn('secret', str(caught.exception) + command.stdout.getvalue())

    def test_message_failures_continue_but_stay_distinct_from_infrastructure(self):
        fake = self.fake_consumer(empty_report(claimed=1, failed=1, failed_ids=['private-id']))
        _, summary = self.run_command(service=fake)
        self.assertEqual((summary['polls'], summary['failed'], summary['acked']), (4, 4, 0))
        self.assertEqual(summary['exit_reason'], 'poll_limit')
        self.assertNotIn('private-id', json.dumps(summary))
        self.sleep.assert_not_called()

    def test_actual_ack_failure_keeps_pending_and_does_not_recount(self):
        mid = self.service.publish(self.event(1))
        self.redis.fail_ack = True
        with patch.object(analytics.logger, 'error'):
            _, summary = self.run_command(count=1, max_polls=2)
        self.assertEqual((summary['failed'], summary['acked'], summary['processed']), (1, 0, 1))
        self.assertIn(mid, self.redis.pending[analytics.GROUP])
        self.assertEqual(self.service.counts(9, '20261014')['click'], 1)
        self.assertEqual(self.redis.close_calls, 1)

    def test_cleanup_failure_is_reported_not_silently_successful(self):
        fake = self.fake_consumer()
        fake.client.close.side_effect = RuntimeError('private credential')
        with self.assertRaisesRegex(CommandError, 'cleanup failed') as caught:
            self.run_command(service=fake, max_polls=1)
        self.assertNotIn('private credential', str(caught.exception))
        fake.consume_once.assert_called_once()
        fake.client.close.assert_called_once()

    def test_original_failure_is_not_hidden_by_cleanup_failure(self):
        fake = self.fake_consumer()
        fake.consume_once.side_effect = ConnectionError('connection')
        fake.client.close.side_effect = RuntimeError('cleanup')
        with self.assertRaisesRegex(CommandError, 'polling failed'):
            self.run_command(service=fake)
        fake.client.close.assert_called_once()

    def test_output_is_fixed_aggregate_schema_not_payload_or_token(self):
        fake = self.fake_consumer(empty_report(fresh=1, acked=1, failed_ids=['private'],
                                               reclaim_cursor='session-token', next_lane='fresh'))
        command, summary = self.run_command(service=fake, max_polls=1)
        self.assertEqual(set(summary), {'polls', 'exit_reason', 'processed', 'duplicates', 'acked',
                                        'failed', 'prune_failed', 'claimed', 'fresh', 'deleted_pending', 'deferred'})
        for word in ('private', 'session-token', 'feedback_type', 'event_timestamp', 'redis://'):
            self.assertNotIn(word, command.stdout.getvalue())

    def test_invalid_options_rejected_before_factory_or_poll(self):
        for changes in ({'count':0}, {'count':101}, {'count':True}, {'max_polls':0},
                        {'max_polls':1001}, {'max_seconds':0}, {'max_seconds':301},
                        {'max_seconds':float('nan')}, {'max_seconds':float('inf')}):
            with self.subTest(changes=changes), self.assertRaises(CommandError):
                self.run_command(**changes)
        self.factory.assert_not_called()
        self.assertEqual(self.redis.calls, [])

    def test_cli_defaults_and_limits_are_finite(self):
        command = command_class(self.settings, self.factory)()
        parser = argparse.ArgumentParser()
        command.add_arguments(parser)
        self.assertEqual(vars(parser.parse_args([])), {'count':100, 'max_polls':10, 'max_seconds':30})
        self.assertEqual(vars(parser.parse_args(['--count','1','--max-polls','4','--max-seconds','20'])),
                         {'count':1, 'max_polls':4, 'max_seconds':20})
        self.assertEqual(command.requires_system_checks, [])
        self.assertFalse(command.requires_migrations_checks)

    def test_actual_production_import_bindings_and_no_model_or_service_calls(self):
        source = COMMAND_PATH.read_text()
        self.assertEqual(p31.unbound_globals(source), set())
        missing = source.replace('from myapp.services.redis_analytics import analytics_from_settings', '')
        self.assertIn('analytics_from_settings', p31.unbound_globals(missing))
        tree = ast.parse(source)
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(modules, {'django.conf', 'django.core.management.base', 'myapp.services.redis_analytics'})
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('analytics_from_settings'), 1)
        for name in ('django.setup', 'subprocess.run', 'cache.clear', 'client.flushdb', 'service.publish'):
            self.assertNotIn(name, calls)
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        self.assertFalse(any(isinstance(n, ast.While) for n in loops))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id=='analytics_from_settings'
                             for loop in loops for n in ast.walk(loop)))
        # No startup/automatic scheduling integration added.
        self.assertNotIn('consume_redis_feedback', (ROOT / 'myapp/apps.py').read_text())


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) or not sys.flags.no_site:
        raise RuntimeError('CPU-only -S environment required')
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError('Unsafe module imported')
    unittest.main(verbosity=2)
