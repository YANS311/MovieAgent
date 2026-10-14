"""Stdlib/Fake contracts; does NOT execute Lua or real Django/Redis."""
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRedis:
    """Explicit Python model of the scripts' intended contract, not Lua proof."""
    def __init__(self):
        self.hashes, self.zsets, self.strings, self.ttl = {}, {}, {}, {}
        self.stream, self.pending, self.groups = {}, {}, {}
        self.sequence = 0
        self.now = 0
        self.calls = []
        self.fail_update = False
        self.fail_ack = False
        self.fail_append = False
        self.fail_mid_update = False
        self.fail_prune = False

    def sweep(self):
        for key, end in list(self.ttl.items()):
            if end <= self.now:
                for store in (self.hashes, self.zsets, self.strings): store.pop(key, None)
                del self.ttl[key]

    def eval(self, script, numkeys, *values):
        self.calls.append(('eval', script, numkeys, values))
        self.sweep()
        if script == analytics.APPEND_LUA:
            _, capacity, event_id, movie_id, field, stamp = values
            if self.fail_append: raise ConnectionError('synthetic connection error')
            if len(self.stream) >= capacity: raise RuntimeError('ANALYTICS_BACKLOG_FULL')
            self.sequence += 1
            message_id = str(self.sequence) + '-0'
            self.stream[message_id] = {'event_id': event_id, 'movie_id': movie_id,
                                      'feedback_type': field, 'event_timestamp': stamp, 'schema_version': '1'}
            return message_id
        if script == analytics.PRUNE_LUA:
            _, group, message_id = values
            if self.fail_prune: raise ConnectionError('synthetic prune error')
            if list(self.groups) != [group] or message_id in self.pending[group]: return 0
            return int(self.stream.pop(message_id, None) is not None)
        if script != analytics.UPDATE_LUA: raise AssertionError('Unknown script')
        hash_key, hot_key, marker_key, field, weight, movie_id, stats_ttl, dedupe_ttl, fingerprint = values
        if self.fail_update: raise ConnectionError('synthetic update error')
        marker = self.strings.get(marker_key)
        if marker == 'done:' + fingerprint: return 0
        if marker: raise RuntimeError('ANALYTICS_UNCERTAIN_UPDATE_OR_ID_CONFLICT')
        count = self.hashes.get(hash_key, {}).get(field, 0)
        if type(count) is not int or not 0 <= count <= 1000000000000:
            raise RuntimeError('ANALYTICS_BAD_COUNT')
        # Atomicity assumption is modeled by one method invocation; only real
        # Redis integration can establish isolation of the actual Lua script.
        self.strings[marker_key] = 'pending:' + fingerprint
        self.ttl[marker_key] = self.now + dedupe_ttl
        self.hashes.setdefault(hash_key, {})[field] = count + 1
        if self.fail_mid_update: raise RuntimeError('synthetic mid-script error')
        self.zsets.setdefault(hot_key, {})[movie_id] = self.zsets.get(hot_key, {}).get(movie_id, 0) + weight
        self.ttl[hash_key] = self.ttl[hot_key] = self.now + stats_ttl
        self.strings[marker_key] = 'done:' + fingerprint
        return 1

    def hgetall(self, key):
        self.sweep()
        return dict(self.hashes.get(key, {}))

    def zrevrange(self, key, start, stop, withscores):
        self.sweep()
        return sorted(self.zsets.get(key, {}).items(), key=lambda pair: (pair[1], pair[0]), reverse=True)[start:stop+1]

    def xgroup_create(self, key, group, id, mkstream):
        self.calls.append(('xgroup_create', key, group, id, mkstream))
        if group in self.groups: raise RuntimeError('BUSYGROUP already exists')
        self.groups[group] = 0
        self.pending[group] = {}

    def xinfo_groups(self, key):
        return [{'name': group} for group in self.groups]

    def xreadgroup(self, group, consumer, streams, count):
        self.calls.append(('xreadgroup', count))
        fresh = [(mid, event) for mid, event in self.stream.items() if int(mid.split('-')[0]) > self.groups[group]][:count]
        for mid, event in fresh:
            self.pending[group][mid] = {'consumer': consumer, 'at': self.now}
            self.groups[group] = int(mid.split('-')[0])
        return [(analytics.STREAM_KEY, fresh)] if fresh else []

    def xautoclaim(self, key, group, consumer, min_idle, cursor, count):
        self.calls.append(('xautoclaim', count, cursor))
        reclaimed, deleted = [], []
        # Bounded approximation of Redis's numeric ID / COUNT*10 scan semantics.
        # Still not execution of Redis or Lua, nor a concurrency/ACL proof.
        stream_id = lambda mid: tuple(map(int, mid.split('-')))
        entries = sorted((mid for mid in self.pending[group] if stream_id(mid) >= stream_id(cursor)), key=stream_id)
        scanned = 0
        for mid in entries:
            if scanned >= count * 10 or len(reclaimed) + len(deleted) >= count: break
            scanned += 1
            pending = self.pending[group][mid]
            if mid not in self.stream:
                deleted.append(mid); del self.pending[group][mid]; continue
            if self.now - pending['at'] < min_idle / 1000: continue
            pending.update(consumer=consumer, at=self.now)
            reclaimed.append((mid, self.stream[mid]))
        next_cursor = entries[scanned] if scanned < len(entries) else '0-0'
        return [next_cursor, reclaimed, deleted]

    def xack(self, key, group, message_id):
        self.calls.append(('xack', message_id))
        if self.fail_ack: raise ConnectionError('synthetic ACK error')
        return int(self.pending[group].pop(message_id, None) is not None)


analytics = load_file('p3_analytics', 'myapp/services/redis_analytics.py')
p01 = load_file('p3_p01_fixture', 'tests/test_redis_cache_p01_isolated.py')


class P3Tests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.service = analytics.FeedbackAnalytics(self.redis)
        self.settings = types.SimpleNamespace(
            REDIS_ANALYTICS_ENABLED=True, REDIS_ANALYTICS_URL='redis://localhost:6379/2',
            CACHES={'default': {'LOCATION': 'redis://127.0.0.1:6379/0'}})

    def event(self, event_id=1, movie_id=9, field='click'):
        return {'event_id': str(event_id), 'movie_id': str(movie_id), 'feedback_type': field,
                'event_timestamp': '2026-10-13T12:00:00.000000+00:00', 'schema_version': '1'}

    def test_each_hash_field_counts_processed_events(self):
        for i, field in enumerate(analytics.FIELDS): self.service.process(self.event(i+1, field=field))
        self.assertEqual(self.service.counts(9, '20261013'), dict.fromkeys(analytics.FIELDS, 1))

    def test_hash_and_zset_ttl_eight_days_marker_nine_days(self):
        self.service.process(self.event())
        self.assertEqual(self.redis.ttl[analytics.PREFIX+':movie:9:day:20261013'], 8*86400)
        self.assertEqual(self.redis.ttl[analytics.PREFIX+':hot:20261013'], 8*86400)
        self.assertEqual(self.redis.ttl[analytics.PREFIX+':dedupe:event:1'], 9*86400)

    def test_weights_are_auxiliary_and_sum_expected_score(self):
        for i, field in enumerate(analytics.FIELDS): self.service.process(self.event(i+1, field=field))
        self.assertEqual(self.service.top('20261013'), [{'movie_id': 9, 'analytics_score': 11.0}])

    def test_top_k_and_equal_score_reverse_lexical_order(self):
        for i, mid in enumerate((2, 9, 10)): self.service.process(self.event(i+1, mid))
        self.assertEqual([r['movie_id'] for r in self.service.top('20261013', 2)], [9, 2])

    def test_adjustable_weights_do_not_change_other_caches(self):
        self.settings.REDIS_ANALYTICS_WEIGHTS = {'like': 7}
        service = analytics.analytics_from_settings(self.settings, factory=lambda *a, **k: self.redis)
        service.process(self.event(field='like'))
        self.assertEqual(service.top('20261013')[0]['analytics_score'], 7.0)

    def test_publish_group_consume_ack_and_safe_retention(self):
        message_id = self.service.publish(self.event())
        report = self.service.consume_once()
        self.assertEqual((report['processed'], report['acked']), (1, 1))
        self.assertNotIn(message_id, self.redis.pending[analytics.GROUP])
        self.assertNotIn(message_id, self.redis.stream)

    def test_publish_is_atomic_capacity_admission_not_pending_trim(self):
        self.service.capacity = 1
        first = self.service.publish(self.event())
        with self.assertRaisesRegex(RuntimeError, 'BACKLOG_FULL'): self.service.publish(self.event(2))
        self.assertIn(first, self.redis.stream)
        self.assertEqual(len(self.redis.stream), 1)

    def test_processing_failure_does_not_ack(self):
        mid = self.service.publish(self.event()); self.redis.fail_update = True
        with self.assertLogs('movie_agent.analytics', level='ERROR'): report = self.service.consume_once()
        self.assertEqual((report['failed'], report['acked']), (1, 0))
        self.assertIn(mid, self.redis.pending[analytics.GROUP])
        self.assertFalse(any(call[0]=='xack' for call in self.redis.calls))

    def test_stale_pending_is_reclaimed_and_processed(self):
        self.service.publish(self.event()); self.redis.fail_update=True
        with self.assertLogs('movie_agent.analytics', level='ERROR'): self.service.consume_once()
        self.redis.fail_update=False;self.redis.now=61
        report=self.service.consume_once(consumer='recovery')
        self.assertEqual((report['claimed'],report['acked']), (1,1))

    def test_ack_failure_then_redelivery_does_not_double_count(self):
        self.service.publish(self.event()); self.redis.fail_ack=True
        with self.assertLogs('movie_agent.analytics', level='ERROR'): self.service.consume_once()
        self.redis.now=61;self.redis.fail_ack=False
        report=self.service.consume_once()
        self.assertEqual((report['duplicates'],report['acked']), (1,1))
        self.assertEqual(self.service.counts(9,'20261013')['click'],1)

    def test_duplicate_xadd_delivery_counts_once(self):
        self.service.publish(self.event());self.service.publish(self.event())
        report=self.service.consume_once()
        self.assertEqual((report['processed'],report['duplicates'],report['acked']),(1,1,2))

    def test_event_id_conflict_is_not_acknowledged(self):
        self.service.publish(self.event());self.service.publish(self.event(movie_id=10))
        with self.assertLogs('movie_agent.analytics', level='ERROR'): report=self.service.consume_once()
        self.assertEqual((report['acked'],report['failed']),(1,1))

    def test_atomic_contract_one_eval_updates_three_scoped_keys(self):
        self.service.process(self.event())
        call=self.redis.calls[-1]
        self.assertEqual((call[0],call[2]),('eval',3))
        self.assertTrue(all(key.startswith(analytics.PREFIX+':') for key in call[3][:3]))
        for command in ('HINCRBY','ZINCRBY','EXPIRE','SET','GET'):
            self.assertIn("'"+command+"'",analytics.UPDATE_LUA)
        self.assertLess(analytics.UPDATE_LUA.index('redis.acl_check_cmd'),analytics.UPDATE_LUA.index("redis.call('HINCRBY'"))

    def test_preflight_bad_count_has_no_partial_updates(self):
        key=analytics.PREFIX+':movie:9:day:20261013';self.redis.hashes[key]={'click':'bad'}
        before=copy.deepcopy((self.redis.hashes,self.redis.zsets,self.redis.strings))
        with self.assertRaises(RuntimeError):self.service.process(self.event())
        self.assertEqual(before,(self.redis.hashes,self.redis.zsets,self.redis.strings))

    def test_unexpected_mid_script_failure_is_reserved_not_recounted(self):
        self.redis.fail_mid_update=True
        with self.assertRaises(RuntimeError):self.service.process(self.event())
        self.redis.fail_mid_update=False
        with self.assertRaises(RuntimeError):self.service.process(self.event())
        self.assertEqual(self.service.counts(9,'20261013')['click'],1)
        self.assertEqual(self.redis.zsets,{})
        self.assertTrue(next(iter(self.redis.strings.values())).startswith('pending:'))

    def test_dedupe_expiry_explicitly_allows_recount(self):
        self.service.process(self.event());self.redis.now=9*86400+1
        self.assertEqual(self.service.process(self.event()),1)

    def test_second_group_rejected_and_pruning_keeps_shared_entries(self):
        mid=self.service.publish(self.event());self.service.ensure_group()
        self.redis.groups['other']=0;self.redis.pending['other']={}
        with self.assertRaises(analytics.AnalyticsError):self.service.consume_once()
        self.assertEqual(self.redis.eval(analytics.PRUNE_LUA,1,analytics.STREAM_KEY,analytics.GROUP,mid),0)
        self.assertIn(mid,self.redis.stream)

    def test_missing_pending_payload_is_reported_not_success(self):
        mid=self.service.publish(self.event());self.service.ensure_group()
        self.redis.xreadgroup(analytics.GROUP,'old',{analytics.STREAM_KEY:'>'},count=1)
        del self.redis.stream[mid];self.redis.now=61
        with self.assertLogs('movie_agent.analytics', level='ERROR'):report=self.service.consume_once()
        self.assertEqual(report['deleted_pending'],1)
        self.assertEqual(report['acked'],0)

    def test_prune_failure_reports_retention_without_recount(self):
        mid=self.service.publish(self.event());self.redis.fail_prune=True
        with self.assertLogs('movie_agent.analytics', level='ERROR'):report=self.service.consume_once()
        self.assertEqual((report['acked'],report['prune_failed']),(1,1));self.assertIn(mid,self.redis.stream)

    def test_bounded_consumer_and_no_blocking_reads(self):
        for i in range(3):self.service.publish(self.event(i+1))
        self.assertEqual(self.service.consume_once(count=2)['acked'],2)
        self.assertEqual(len(self.redis.stream),1)
        for options in ({'count':101},{'min_idle_ms':1},{'max_seconds':31},{'consumer':'unsafe:token'}):
            with self.assertRaises(analytics.AnalyticsError):self.service.consume_once(**options)

    def test_disabled_never_reads_url_connects_or_registers(self):
        settings=types.SimpleNamespace(REDIS_ANALYTICS_ENABLED=False)
        factory=Mock();tx=Mock()
        self.assertIsNone(analytics.analytics_from_settings(settings,factory=factory))
        self.assertFalse(analytics.schedule_feedback_analytics(tx,object(),settings))
        factory.assert_not_called();tx.on_commit.assert_not_called()

    def test_missing_url_and_shared_db_refused_before_connection(self):
        for url in (None,'redis://localhost:6379/0','redis://user:pass@127.0.0.1:6379/00',
                    'redis://localhost:6379','redis://localhost:6379/2?db=0'):
            self.settings.REDIS_ANALYTICS_URL=url;factory=Mock()
            with self.assertRaises(analytics.AnalyticsError):analytics.analytics_from_settings(self.settings,factory)
            factory.assert_not_called()

    def test_explicit_connection_timeouts_and_capacity(self):
        factory=Mock(return_value=self.redis)
        self.assertIsNotNone(analytics.analytics_from_settings(self.settings,factory))
        self.assertEqual(factory.call_args.kwargs['max_connections'],2)
        self.assertEqual(factory.call_args.kwargs['socket_timeout'],.5)

    def test_event_privacy_and_utc_day(self):
        feedback=types.SimpleNamespace(pk=1,movie_id=9,feedback_type='like',
            created_at=datetime.fromisoformat('2026-10-14T00:30:00+08:00'),user='private',source='chat')
        event=analytics.event_from_feedback(feedback)
        self.assertEqual(set(event),{'event_id','movie_id','feedback_type','event_timestamp','schema_version'})
        self.service.process(event);self.assertTrue(self.redis.hashes.get(analytics.PREFIX+':movie:9:day:20261013'))
        with self.assertRaises(analytics.AnalyticsError):analytics.validate_event({**event,'session_id':'secret'})

    def test_invalid_events_have_no_stats_or_stream_mutation(self):
        for change in ({'schema_version':'old'},{'feedback_type':'unknown'},{'movie_id':'0'},
                       {'event_timestamp':'2026-10-13T12:00:00'}):
            with self.assertRaises((analytics.AnalyticsError,ValueError)):self.service.publish({**self.event(),**change})
        self.assertEqual(self.redis.calls,[])

    def feedback_fixture(self, enabled=True):
        old=p01.P01Tests();old.setUp();fn,request,db,tx,modules=old.feedback('recommend_feedback_view')
        fn.__globals__['settings']=self.settings if enabled else types.SimpleNamespace(REDIS_ANALYTICS_ENABLED=False)
        fn.__globals__['schedule_feedback_analytics']=analytics.schedule_feedback_analytics
        original=db.create
        def create(**kwargs):
            original(**kwargs)
            return types.SimpleNamespace(pk=10,movie_id=kwargs['movie'].id,feedback_type=kwargs['feedback_type'],
                                         created_at=datetime(2026,10,13,tzinfo=timezone.utc))
        db.create=create
        return fn,request,db,tx,modules

    def test_db_commit_delivers_then_keeps_original_invalidation(self):
        fn,req,db,tx,modules=self.feedback_fixture()
        invalidation=Mock();fn.__globals__['invalidate_recommendation_cache']=invalidation
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):
            response=fn(req)
        self.assertEqual(response['status'],'success');self.assertEqual(len(self.redis.stream),1)
        invalidation.assert_called_once();self.assertEqual(len(db.rows),2)

    def test_outer_rollback_does_not_publish_or_invalidate(self):
        fn,req,db,tx,modules=self.feedback_fixture();invalidation=Mock();fn.__globals__['invalidate_recommendation_cache']=invalidation
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):
            with self.assertRaises(RuntimeError):
                with tx.atomic():fn(req);raise RuntimeError('rollback')
        self.assertEqual(self.redis.stream,{});invalidation.assert_not_called();self.assertEqual(len(db.rows),1)

    def test_analytics_failure_does_not_repeat_db_or_block_invalidation(self):
        fn,req,db,tx,modules=self.feedback_fixture();self.redis.fail_append=True
        invalidation=Mock();fn.__globals__['invalidate_recommendation_cache']=invalidation
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service),self.assertLogs('movie_agent.analytics',level='ERROR'):
            response=fn(req)
        self.assertEqual(response['status'],'success');self.assertEqual(len(db.rows),2);invalidation.assert_called_once()

    def test_cache_failure_does_not_prevent_already_committed_event_delivery(self):
        fn,req,db,tx,modules=self.feedback_fixture()
        fn.__globals__['invalidate_recommendation_cache']=Mock(side_effect=p01.keys.RecommendationCacheInvalidationError('cache',user_id=7))
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):response=fn(req)
        self.assertEqual(response['status'],'partial_success');self.assertEqual(len(self.redis.stream),1)
        self.assertEqual(len(db.rows),2)

    def test_db_create_failure_never_publishes(self):
        fn,req,db,tx,modules=self.feedback_fixture();db.fail_create=True
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):response=fn(req)
        self.assertEqual(response['status'],'error');self.assertEqual(self.redis.stream,{})

    def test_disabled_view_leaves_original_behavior_without_analytics_calls(self):
        fn,req,db,tx,modules=self.feedback_fixture(enabled=False)
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings') as factory:response=fn(req)
        self.assertEqual(response['status'],'success');factory.assert_not_called()

    def test_nested_commit_waits_for_outer_commit(self):
        fn,req,db,tx,modules=self.feedback_fixture()
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):
            with tx.atomic():
                fn(req)
                self.assertEqual(self.redis.stream,{})
            self.assertEqual(len(self.redis.stream),1)

    def test_savepoint_rollback_discards_only_rolled_back_event(self):
        fn,req,db,tx,modules=self.feedback_fixture()
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service):
            with tx.atomic():
                try:
                    with tx.atomic():fn(req);raise RuntimeError('savepoint')
                except RuntimeError:pass
                fn(req)
        self.assertEqual(len(self.redis.stream),1);self.assertEqual(len(db.rows),2)

    def test_time_budget_leaves_fetched_unprocessed_message_pending(self):
        for i in range(2):self.service.publish(self.event(i+1))
        # Advance the clock after the first actual update, independent of the
        # number of scheduler clock checks. Keep all original outcome assertions.
        clock = [0]
        original = self.service.process
        def process(event):
            result = original(event)
            clock[0] = 2
            return result
        with patch.object(analytics.time,'monotonic',side_effect=lambda: clock[0]),patch.object(self.service,'process',side_effect=process):
            report=self.service.consume_once(count=2,max_seconds=1)
        self.assertEqual(report['acked'],1);self.assertEqual(report['deferred'],1)
        self.assertEqual(len(self.redis.pending[analytics.GROUP]),1)

    def test_logger_failure_does_not_break_delivery_callback_or_invalidation(self):
        fn,req,db,tx,modules=self.feedback_fixture();self.redis.fail_append=True
        invalidation=Mock();fn.__globals__['invalidate_recommendation_cache']=invalidation
        with patch.dict(sys.modules,modules),patch.object(analytics,'analytics_from_settings',return_value=self.service),patch.object(analytics.logger,'error',side_effect=RuntimeError('handler')):
            response=fn(req)
        self.assertEqual(response['status'],'success');invalidation.assert_called_once()

    def test_invalid_weight_or_capacity_refused_before_factory(self):
        for weights in ({'click':True},{'like':float('nan')},{'click':-1},{'extra':1}):
            self.settings.REDIS_ANALYTICS_WEIGHTS=weights;factory=Mock()
            with self.assertRaises(analytics.AnalyticsError):analytics.analytics_from_settings(self.settings,factory)
            factory.assert_not_called()

    def test_failure_report_contains_stream_ids_for_pending_recovery(self):
        mid=self.service.publish(self.event());self.redis.fail_update=True
        with self.assertLogs('movie_agent.analytics',level='ERROR'):report=self.service.consume_once()
        self.assertEqual(report['failed_ids'],[mid])

    def test_delivery_log_uses_fixed_reason_without_raw_secret(self):
        with self.assertLogs('movie_agent.analytics',level='ERROR') as captured:
            analytics._failure('delivery',RuntimeError('ANALYTICS_BACKLOG_FULL secret-url-password'),event_id='10')
        self.assertIn('event_id=10',captured.output[0]);self.assertIn('reason=ANALYTICS_BACKLOG_FULL',captured.output[0])
        self.assertNotIn('secret-url-password',captured.output[0])

    def test_production_import_binding_and_callback_order(self):
        p01.p0.assert_production_cache_bindings('myapp/agent_views.py')
        source=(ROOT/'myapp/agent_views.py').read_text()
        with self.assertRaises(AssertionError):p01.p0.assert_production_cache_bindings('myapp/agent_views.py',source.replace('from myapp.services.redis_analytics import schedule_feedback_analytics',''))
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='recommend_feedback_view')
        body=ast.unparse(node)
        self.assertLess(body.index('schedule_feedback_analytics('),body.index('transaction.on_commit('))

    def test_cache_keys_algorithms_models_and_p2_data_unchanged(self):
        paths=['myapp/views.py','myapp/utils/cache_keys.py','myapp/utils/cache_access.py',
               'myapp/services/cache_mutations.py','myapp/agent/movie_agent.py','myapp/recommender/recall.py',
               'myapp/recommender/rerank.py','myapp/models_upgrade.py',
               'benchmarks/redis_cache_microbench_validation.json']
        for path in paths:self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','f4037f90acd1132ae68652cf0e3d7b6dec58bf3a:'+path],cwd=ROOT))

    def test_no_global_clean_or_stream_expiry_and_no_startup_hook(self):
        for forbidden in ('FLUSHDB','FLUSHALL','XTRIM','MAXLEN'):
            self.assertNotIn("'"+forbidden+"'",analytics.APPEND_LUA+analytics.UPDATE_LUA+analytics.PRUNE_LUA)
        self.assertNotIn(analytics.STREAM_KEY,self.redis.ttl)
        self.assertNotIn('redis_analytics',(ROOT/'myapp/apps.py').read_text())


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) or not sys.flags.no_site:
        raise RuntimeError('CPU-only -S environment required')
    forbidden={'torch','transformers','sentence_transformers','numpy','faiss','tensorflow','redis','django','MySQLdb','neo4j'}
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in forbidden:raise RuntimeError('Forbidden import: '+fullname)
    sys.meta_path.insert(0,Guard())
    if any(name.split('.')[0] in forbidden for name in sys.modules):raise RuntimeError('Unsafe module imported')
    unittest.main(verbosity=2)
