"""P3.1 scheduling contracts; stdlib + Fake only, no real Redis/Lua/Django."""
import ast
import builtins
import importlib.util
import os
from pathlib import Path
import symtable
import sys
import unittest
from unittest.mock import patch

FORBIDDEN = {'torch', 'transformers', 'sentence_transformers', 'numpy', 'faiss',
             'tensorflow', 'redis', 'django', 'MySQLdb', 'neo4j'}


class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Forbidden import: ' + fullname)


# Install before loading any fixture/helper, not just before unittest.main().
sys.meta_path.insert(0, ImportGuard())
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p31_fixture', ROOT / 'tests/test_redis_structures_p3_isolated.py')
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
analytics = p3.analytics


def unbound_globals(source):
    """Inspect the actual module bindings, without injecting production symbols."""
    tree = symtable.symtable(source, 'redis_analytics.py', 'exec')
    bound = {s.get_name() for s in tree.get_symbols() if s.is_assigned() or s.is_imported()}
    bound.update(dir(builtins))
    missing = set()
    def visit(table):
        for symbol in table.get_symbols():
            # Inlined comprehension symbols can be both local and global in
            # newer CPython symtables; parameters/loop targets remain local.
            if (symbol.is_referenced() and symbol.is_global() and not symbol.is_local()
                    and symbol.get_name() not in bound):
                missing.add(symbol.get_name())
        for child in table.get_children():
            visit(child)
    visit(tree)
    return missing


class P31Tests(unittest.TestCase):
    def setUp(self):
        self.redis = p3.FakeRedis()
        self.service = analytics.FeedbackAnalytics(self.redis)

    def event(self, event_id, movie_id=9):
        return {'event_id': str(event_id), 'movie_id': str(movie_id), 'feedback_type': 'click',
                'event_timestamp': '2026-10-13T12:00:00.000000+00:00', 'schema_version': '1'}

    def pending(self, size, poison=True):
        ids = [self.service.publish(self.event(i + 1)) for i in range(size)]
        self.service.ensure_group()
        self.redis.xreadgroup(analytics.GROUP, 'old', {analytics.STREAM_KEY: '>'}, count=size)
        if poison:
            for mid in ids:
                self.redis.stream[mid]['feedback_type'] = 'invalid'
        self.redis.now = 61
        self.redis.calls.clear()
        return ids

    def poll(self, **kwargs):
        # Toxic messages deliberately emit errors; collect logs without silencing failures.
        with patch.object(analytics.logger, 'error') as logs:
            report = self.service.consume_once(**kwargs)
        if report['failed']:
            self.assertGreaterEqual(logs.call_count, report['failed'])
        return report

    def reads(self):
        return [call for call in self.redis.calls if call[0] in ('xautoclaim', 'xreadgroup')]

    def ack_ids(self):
        return [call[1] for call in self.redis.calls if call[0] == 'xack']

    def test_count_one_rotates_with_persistent_poison_and_new_arrivals(self):
        poison = self.pending(1)[0]
        results = []
        fresh_ids = []
        for i in range(4):
            fresh_ids.append(self.service.publish(self.event(10 + i)))
            self.redis.now = 61 * (i + 1)
            results.append(self.poll(count=1))
        self.assertEqual([r['acked'] for r in results], [0, 1, 0, 1])
        self.assertEqual([r['failed'] for r in results], [1, 0, 1, 0])
        self.assertEqual(self.ack_ids(), fresh_ids[:2])
        self.assertIn(poison, self.redis.pending[analytics.GROUP])
        self.assertEqual([c[0] for c in self.reads()], ['xautoclaim', 'xreadgroup'] * 2)

    def test_count_two_reserves_both_lanes_each_poll(self):
        poison = self.pending(2)
        fresh = [self.service.publish(self.event(10 + i)) for i in range(2)]
        first = self.poll(count=2)
        self.redis.now = 122
        second = self.poll(count=2)
        self.assertEqual([(r['claimed'], r['fresh'], r['acked'], r['failed']) for r in (first, second)],
                         [(1, 1, 1, 1), (1, 1, 1, 1)])
        self.assertEqual(self.ack_ids(), fresh)
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), set(poison))
        self.assertEqual([c[0] for c in self.reads()], ['xautoclaim', 'xreadgroup', 'xreadgroup', 'xautoclaim'])

    def test_count_hundred_full_poison_cannot_take_fresh_quota(self):
        self.pending(100)
        fresh = [self.service.publish(self.event(1000 + i)) for i in range(100)]
        report = self.poll(count=100)
        self.assertEqual((report['claimed'], report['fresh'], report['failed'], report['acked']), (50, 50, 50, 50))
        self.assertEqual(self.ack_ids(), fresh[:50])
        self.assertEqual(self.reads(), [('xautoclaim', 50, '0-0'), ('xreadgroup', 50)])
        self.assertEqual(report['reclaim_cursor'], '51-0')

    def test_without_pending_fresh_uses_whole_budget(self):
        for i in range(6): self.service.publish(self.event(i + 1))
        report = self.poll(count=6)
        self.assertEqual((report['fresh'], report['claimed'], report['acked']), (6, 0, 6))
        self.assertEqual(self.reads(), [('xautoclaim', 3, '0-0'), ('xreadgroup', 6)])

    def test_without_fresh_pending_refills_using_returned_cursor(self):
        mids = self.pending(4, poison=False)
        report = self.poll(count=4)
        self.assertEqual((report['claimed'], report['fresh'], report['acked']), (4, 0, 4))
        self.assertEqual(self.reads(), [('xautoclaim', 2, '0-0'), ('xreadgroup', 2), ('xautoclaim', 2, '3-0')])
        self.assertEqual(self.ack_ids(), mids)

    def test_fresh_first_no_fresh_pending_gets_all_budget(self):
        mids = self.pending(5, poison=False)
        report = self.poll(count=5, first_lane='fresh')
        self.assertEqual(report['acked'], 5)
        self.assertEqual(self.reads(), [('xreadgroup', 3), ('xautoclaim', 5, '0-0')])
        self.assertEqual(self.ack_ids(), mids)

    def test_pending_first_fallback_for_count_one(self):
        mid = self.service.publish(self.event(1))
        report = self.poll(count=1)
        self.assertEqual((report['fresh'], report['acked']), (1, 1))
        self.assertEqual(self.reads(), [('xautoclaim', 1, '0-0'), ('xreadgroup', 1)])
        self.assertEqual(self.ack_ids(), [mid])

    def test_fresh_first_fallback_for_count_one(self):
        mids = self.pending(1, poison=False)
        report = self.poll(count=1, first_lane='fresh')
        self.assertEqual((report['claimed'], report['acked']), (1, 1))
        self.assertEqual(self.reads(), [('xreadgroup', 1), ('xautoclaim', 1, '0-0')])
        self.assertEqual(self.ack_ids(), mids)

    def test_failed_pending_no_ack_and_healthy_same_batch_continues(self):
        mids = self.pending(2, poison=False)
        self.redis.stream[mids[0]]['feedback_type'] = 'invalid'
        report = self.poll(count=2)
        self.assertEqual((report['failed'], report['acked'], report['failed_ids']), (1, 1, [mids[0]]))
        self.assertEqual(self.ack_ids(), mids[1:])
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), {mids[0]})

    def test_processing_interleaves_before_rest_of_poison_batch(self):
        mids = self.pending(4)
        fresh = [self.service.publish(self.event(20 + i)) for i in range(2)]
        order = []
        original = self.service.process
        def process(event):
            order.append(event['event_id'])
            return original(event)
        with patch.object(self.service, 'process', side_effect=process): report = self.poll(count=4)
        self.assertEqual(order, ['1', '20', '2', '21'])
        self.assertEqual(report['acked'], 2)
        self.assertEqual(self.ack_ids(), fresh)
        self.assertTrue(all(mid in self.redis.pending[analytics.GROUP] for mid in mids))

    def test_duplicate_delivery_across_lanes_counts_once(self):
        mids = self.pending(1, poison=False)
        self.service.process(self.event(1))  # Statistics applied, previous ACK was lost.
        duplicate = self.service.publish(self.event(1))
        report = self.poll(count=2)
        self.assertEqual((report['duplicates'], report['processed'], report['acked']), (2, 0, 2))
        self.assertEqual(self.ack_ids(), mids + [duplicate])
        self.assertEqual(self.service.counts(9, '20261013')['click'], 1)

    def test_cursor_advances_numeric_ids_then_wraps_and_is_reused(self):
        self.pending(12)
        fresh = [self.service.publish(self.event(100 + i)) for i in range(3)]
        reports = [self.poll(count=8)]
        self.redis.now = 122
        reports.append(self.poll(count=8))
        self.redis.now = 183
        reports.append(self.poll(count=8))
        claims = [c for c in self.reads() if c[0] == 'xautoclaim']
        self.assertEqual(claims[:3], [('xautoclaim', 4, '0-0'), ('xautoclaim', 1, '5-0'), ('xautoclaim', 8, '6-0')])
        self.assertEqual(reports[1]['reclaim_cursor'], '0-0')
        self.assertEqual(claims[3][2], '0-0')
        self.assertEqual(self.ack_ids(), fresh)

    def test_cursor_scan_cap_advances_even_when_no_eligible_messages(self):
        self.pending(12)
        for item in self.redis.pending[analytics.GROUP].values(): item['at'] = self.redis.now
        report = self.poll(count=1)
        self.assertEqual(report['reclaim_cursor'], '0-0')  # Two bounded scans: 10 entries then EOF.
        self.assertEqual(self.reads(), [('xautoclaim', 1, '0-0'), ('xreadgroup', 1), ('xautoclaim', 1, '11-0')])
        self.assertEqual(report['acked'], 0)

    def test_explicit_cursor_and_lane_resume_on_new_instances(self):
        self.pending(3)
        fresh = [self.service.publish(self.event(10 + i)) for i in range(2)]
        state = {}
        reports = []
        for i in range(4):
            self.service = analytics.FeedbackAnalytics(self.redis)
            self.redis.now = 61 * (i + 1)
            report = self.poll(count=1, **state)
            state = {'reclaim_cursor': report['reclaim_cursor'], 'first_lane': report['next_lane']}
            reports.append(report)
        self.assertEqual([r['acked'] for r in reports], [0, 1, 0, 1])
        self.assertEqual(self.ack_ids(), fresh)
        self.assertEqual([c[2] for c in self.reads() if c[0] == 'xautoclaim'], ['0-0', '2-0'])

    def test_restart_without_checkpoint_explicitly_has_no_count_one_guarantee(self):
        self.pending(1)
        mid = self.service.publish(self.event(10))
        for i in range(3):
            self.service = analytics.FeedbackAnalytics(self.redis)
            self.redis.now = 61 * (i + 1)
            report = self.poll(count=1)
            self.assertEqual((report['failed'], report['fresh'], report['next_lane']), (1, 0, 'fresh'))
        self.assertEqual(self.ack_ids(), [])
        self.assertIn(mid, self.redis.stream)
        self.assertNotIn(mid, self.redis.pending[analytics.GROUP])

    def test_restart_count_two_still_reads_both_lanes(self):
        self.pending(1)
        fresh = [self.service.publish(self.event(10 + i)) for i in range(3)]
        for i in range(3):
            self.service = analytics.FeedbackAnalytics(self.redis)
            self.redis.now = 61 * (i + 1)
            self.assertEqual(self.poll(count=2)['acked'], 1)
        self.assertEqual(self.ack_ids(), fresh)

    def test_empty_poll_is_bounded_and_does_not_ack(self):
        report = self.poll(count=100)
        self.assertEqual(len(self.reads()), 3)
        self.assertEqual((report['acked'], report['processed'], report['failed'], report['deferred']), (0, 0, 0, 0))
        self.assertEqual(self.ack_ids(), [])

    def test_budget_no_more_than_count_fetched_or_attempted(self):
        for count in (1, 2, 3, 100):
            with self.subTest(count=count):
                self.setUp()
                self.pending(110)
                for i in range(110): self.service.publish(self.event(1000 + i))
                report = self.poll(count=count)
                self.assertLessEqual(report['claimed'] + report['fresh'], count)
                self.assertLessEqual(report['failed'] + report['acked'] + report['deferred'], count)
                self.assertLessEqual(len(self.reads()), 3)

    def test_no_block_argument_in_real_call(self):
        self.service.publish(self.event(1))
        with patch.object(self.redis, 'xreadgroup', wraps=self.redis.xreadgroup) as read:
            self.poll(count=2)
        self.assertTrue(read.called)
        self.assertTrue(all('block' not in call.kwargs for call in read.call_args_list))

    def test_claim_error_propagates_without_ack_or_success_report(self):
        self.pending(1)
        with patch.object(self.redis, 'xautoclaim', side_effect=ConnectionError('fake')):
            with self.assertRaises(ConnectionError): self.service.consume_once(count=2)
        self.assertEqual(self.ack_ids(), [])
        self.assertEqual(self.redis.hashes, {})
        self.assertEqual(self.service._next_lane, 'fresh')

    def test_read_error_leaves_claimed_pending_and_keeps_cursor(self):
        mids = self.pending(2, poison=False)
        with patch.object(self.redis, 'xreadgroup', side_effect=ConnectionError('fake')):
            with self.assertRaises(ConnectionError): self.service.consume_once(count=2)
        self.assertEqual(self.ack_ids(), [])
        self.assertEqual(set(self.redis.pending[analytics.GROUP]), set(mids))
        self.assertEqual(self.service._reclaim_cursor, '2-0')
        self.assertEqual(self.redis.hashes, {})

    def test_ack_failure_does_not_prevent_other_lane_success(self):
        mids = self.pending(1, poison=False)
        fresh = self.service.publish(self.event(10))
        original = self.redis.xack
        def ack(group_key, group, mid):
            if mid == mids[0]: raise ConnectionError('fake ACK failure')
            return original(group_key, group, mid)
        with patch.object(self.redis, 'xack', side_effect=ack): report = self.poll(count=2)
        self.assertEqual((report['failed'], report['acked']), (1, 1))
        self.assertEqual(self.ack_ids(), [fresh])
        self.assertIn(mids[0], self.redis.pending[analytics.GROUP])

    def test_time_budget_interleaving_and_next_poll_priority(self):
        self.pending(2)
        fresh = self.service.publish(self.event(10))
        clock = [0]
        original = self.service.process
        def process(event):
            try: return original(event)
            finally: clock[0] = 2
        with patch.object(analytics.time, 'monotonic', side_effect=lambda: clock[0]),patch.object(self.service, 'process', side_effect=process):
            report = self.poll(count=2, max_seconds=1)
        self.assertEqual((report['failed'], report['acked'], report['deferred']), (1, 0, 1))
        self.assertEqual(report['next_lane'], 'fresh')
        self.redis.now = 122
        next_report = self.poll(count=2)
        # The deferred fresh entry is now Pending; recovery, not XREADGROUP, delivers it.
        self.assertEqual(next_report['acked'], 1)
        self.assertIn(fresh, self.ack_ids())

    def test_invalid_state_or_malformed_cursor_cannot_ack(self):
        for options in ({'first_lane': 'wrong'}, {'reclaim_cursor': 3}, {'reclaim_cursor': 'secret'}):
            with self.assertRaises(analytics.AnalyticsError): self.service.consume_once(**options)
        self.assertEqual(self.redis.calls, [])
        with patch.object(self.redis, 'xautoclaim', return_value=['invalid', [], []]):
            with self.assertRaises(analytics.AnalyticsError): self.service.consume_once()
        self.assertEqual(self.ack_ids(), [])

    def test_production_symbol_bindings_not_supplied_by_fixture(self):
        source = (ROOT / 'myapp/services/redis_analytics.py').read_text()
        self.assertEqual(unbound_globals(source), set())
        self.assertIn('time', unbound_globals(source.replace('import time\n', '')))
        p3.p01.p0.assert_production_cache_bindings('myapp/agent_views.py')
        ast.parse(source)

    def test_original_p3_has_all_41_test_methods_and_original_assertions(self):
        source = ast.parse((ROOT / 'tests/test_redis_structures_p3_isolated.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'P3Tests')
        tests = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        self.assertEqual(len(tests), 41)
        budget = next(n for n in tests if n.name == 'test_time_budget_leaves_fetched_unprocessed_message_pending')
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'assertEqual'
                             for n in ast.walk(budget)), 3)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) or not sys.flags.no_site:
        raise RuntimeError('CPU-only -S environment required')
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError('Unsafe module imported')
    unittest.main(verbosity=2)
