"""Stdlib-only standalone PoC tests; never initialize Django or real services."""
import ast
import builtins
import asyncio
import contextlib
import contextvars
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import symtable
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / 'experiments/asgi_streaming'
BASELINE = 'e60eb5a96662f9e448df571e5e0e495bd43ba737'
PRODUCTION_FILES = (
    'myapp/agent_views.py',
    'myapp/services/agent_chat_service.py',
    'myapp/services/agent_contracts.py',
)
FROZEN_MANIFEST_SHA256 = '5e97348f2b4c586f39f83f36cb796db9b442855aa1795d526c7d9b98c7fa3c99'
# Explicit R0 test-only revision; never substitute a production or document hash.
AUTHORIZED_TEST_REPAIRS = {
    'tests/test_agent_sse_lifecycle_isolated.py': (
        'f6172bf0be0cbec9b58240cd2e4c14f0ccef71245cddc4c7d7e2d50892f0164b',
        '1ec028e1868a75188944d0c126796f33e779148649818784ea5a9facfb8c550b',
    ),
}


def frozen_production_diff(root):
    return subprocess.check_output([
        'git', 'diff', '--no-ext-diff', '--no-textconv', '--binary',
        BASELINE, '--', *PRODUCTION_FILES,
    ], cwd=root, timeout=15)


def assert_protected_p4b_files(root):
    path = root / 'docs/agent_backend/p4b1/P4B_FROZEN_SHA256.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise AssertionError('Original frozen manifest changed')
    manifest = json.loads(raw)
    if manifest['baseline_commit'] != BASELINE:
        raise AssertionError('Frozen baseline changed')
    if not set(PRODUCTION_FILES) <= manifest['protected_files_sha256'].keys():
        raise AssertionError('Missing production SHA protection')
    for name, original_digest in manifest['protected_files_sha256'].items():
        digest = original_digest
        if name in AUTHORIZED_TEST_REPAIRS:
            original, revision = AUTHORIZED_TEST_REPAIRS[name]
            if original_digest != original:
                raise AssertionError('Original test freeze evidence changed')
            digest = revision
        file = root / name
        if file.is_symlink() or not file.is_file():
            raise AssertionError(f'Frozen path missing or not regular: {name}')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Frozen file content changed: {name}')
    if hashlib.sha256(frozen_production_diff(root)).hexdigest() != manifest['p4b_diff_sha256']:
        raise AssertionError('Fixed-baseline production diff changed')
    # Neither the index nor HEAD may hide an unsafe blob/mode under good
    # worktree bytes. New documents may be absent before staging/commit.
    allowed = {}
    for name, approved in manifest['protected_files_sha256'].items():
        allowed[name] = {approved}
        if name in AUTHORIZED_TEST_REPAIRS:
            allowed[name].update(AUTHORIZED_TEST_REPAIRS[name])
        if name in PRODUCTION_FILES:
            original = subprocess.check_output(['git', 'show', BASELINE + ':' + name],
                                               cwd=root, timeout=15)
            allowed[name].add(hashlib.sha256(original).hexdigest())
    paths = sorted(allowed)
    for revision, command in (
        (':', ['git', 'ls-files', '--stage', '-z', '--', *paths]),
        ('HEAD:', ['git', 'ls-tree', '-z', 'HEAD', '--', *paths]),
    ):
        for record in subprocess.check_output(command, cwd=root, timeout=15).split(b'\0'):
            if not record:
                continue
            metadata, raw_name = record.split(b'\t', 1)
            mode, kind_or_oid, stage_or_oid = metadata.split()
            if mode != b'100644' or (revision == ':' and stage_or_oid != b'0'):
                raise AssertionError('Frozen blob mode/stage changed')
            name = raw_name.decode('utf-8')
            data = subprocess.check_output(['git', 'show', revision + name], cwd=root, timeout=15)
            if hashlib.sha256(data).hexdigest() not in allowed[name]:
                raise AssertionError(f'Unfrozen stored blob: {revision}{name}')



class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'django', 'myapp', 'movie', 'torch', 'transformers',
                'sentence_transformers', 'redis', 'numpy', 'neo4j', 'pymysql'}:
            raise AssertionError('Disallowed application/model/service import')
        return None


sys.meta_path.insert(0, ImportGuard())


def deny_network(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise AssertionError('Real network access forbidden')


sys.addaudithook(deny_network)


def load(name):
    spec = importlib.util.spec_from_file_location('p4b1_test_' + name, POC / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge, fake, probe_module = load('async_bridge_poc'), load('fake_agent'), load('lifecycle_probe')


def decode(blob):
    return json.loads(blob.decode()[6:])


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = bridge.BridgePool()
        self.probe = probe_module.LifecycleProbe()
        self.gates, self.tasks = [], []
        self.scenario = fake.FakeScenario(self.probe)

    def gate(self):
        gate = threading.Event()
        self.gates.append(gate)
        return gate

    def stream(self, source=None, **kwargs):
        return self.pool.open(source or self.scenario.source, probe=self.probe, **kwargs)

    def task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def collect(self, stream):
        return [decode(blob) async for blob in stream]

    async def asyncTearDown(self):
        for gate in self.gates:
            gate.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        state = await self.pool.aclose(grace=2)
        self.assertTrue(state['drained'])
        self.assertTrue(state['joined'])
        self.assertEqual(state['workers_alive'], 0)
        self.assertEqual(state['admitted'], 0)

    async def test_order_and_single_done(self):
        events = await self.collect(self.stream())
        self.assertEqual([e['type'] for e in events],
            ['intent', 'thought', 'action', 'observation', 'chunk', 'chunk', 'chunk', 'chunk', 'done'])
        self.assertEqual(''.join(e['content'] for e in events if e['type'] == 'chunk'), 'synthetic answer')
        self.assertEqual(len(self.scenario.persistence.rows), 3)
        self.assertTrue(self.scenario.closed.is_set())

    async def test_first_event_before_agent_completion(self):
        agent = fake.FakeAgent(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        stream = self.stream()
        self.assertEqual(decode(await anext(stream))['type'], 'intent')
        await probe_module.wait_until(agent.entered.is_set)
        self.assertFalse(agent.returned.is_set())
        self.assertEqual(self.scenario.persistence.rows, [])
        stream.close()

    async def test_async_contract(self):
        stream = self.stream()
        self.assertIs(aiter(stream), stream)
        self.assertFalse(hasattr(stream, '__iter__'))
        await self.collect(stream)
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_prestart_close_never_runs_factory(self):
        stream = self.stream()
        stream.close()
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertEqual(self.scenario.agent.calls, 0)
        self.assertEqual(self.pool.snapshot()['workers_alive'], 0)

    async def test_cancel_before_agent_checkpoint(self):
        self.scenario = fake.FakeScenario(self.probe, pause_stage='before_agent', gate=self.gate())
        task = self.task(self.collect(self.stream()))
        await probe_module.wait_until(self.scenario.paused.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.gates[0].set()
        await probe_module.wait_until(self.scenario.closed.is_set)
        self.assertEqual(self.scenario.agent.calls, 0)

    async def test_cancel_running_agent_does_not_kill_thread(self):
        agent = fake.FakeAgent(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        stream = self.stream()
        task = self.task(self.collect(stream))
        await probe_module.wait_until(agent.entered.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(agent.returned.is_set())
        self.assertFalse(stream.snapshot()['producer_finished'])
        self.assertEqual(self.pool.snapshot()['running_jobs'], 1)
        with self.assertRaises(bridge.CapacityExceeded):
            self.stream()
        self.gates[0].set()
        await probe_module.wait_until(stream._producer_done.is_set)
        self.assertEqual(self.scenario.persistence.calls, 0)

    async def cancel_at_stage(self, stage, expected_db_calls):
        self.scenario = fake.FakeScenario(self.probe, pause_stage=stage, gate=self.gate())
        stream = self.stream()
        task = self.task(self.collect(stream))
        await probe_module.wait_until(self.scenario.paused.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.gates[0].set()
        await probe_module.wait_until(self.scenario.closed.is_set)
        self.assertEqual(self.scenario.persistence.calls, expected_db_calls)
        self.assertFalse(stream.snapshot()['terminal_delivered'])

    async def test_cancel_trace(self):
        await self.cancel_at_stage('trace_action', 0)

    async def test_cancel_chunk(self):
        await self.cancel_at_stage('chunk', 0)

    async def test_cancel_before_persist(self):
        await self.cancel_at_stage('before_persist', 0)

    async def test_cancel_after_commit_preserves_rows(self):
        await self.cancel_at_stage('after_persist', 1)
        self.assertEqual(len(self.scenario.persistence.rows), 3)

    async def test_cancel_during_transaction_is_not_rollback(self):
        db = fake.FakePersistence(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, persistence=db)
        task = self.task(self.collect(self.stream()))
        await probe_module.wait_until(db.entered.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(db.finished.is_set())
        self.gates[0].set()
        await probe_module.wait_until(db.finished.is_set)
        self.assertEqual(len(db.rows), 3)
        self.assertEqual(db.calls, 1)

    async def test_three_atomic_failure_points(self):
        for point in (1, 2, 3):
            with self.subTest(point=point):
                db = fake.FakePersistence(self.probe, fail_at=point)
                scenario = fake.FakeScenario(self.probe, persistence=db)
                events = await self.collect(self.stream(scenario.source))
                await probe_module.wait_until(lambda: self.pool.snapshot()['admitted'] == 0)
                self.assertEqual(events[-1]['type'], 'error')
                self.assertEqual(events[-1]['code'], 'AGENT_PERSISTENCE_FAILED')
                self.assertNotIn('done', [e['type'] for e in events])
                self.assertEqual(db.rows, [])
                self.assertEqual(db.calls, 1)

    async def test_commit_ack_loss_not_success_or_retry(self):
        db = fake.FakePersistence(self.probe, commit_ack_loss=True)
        self.scenario = fake.FakeScenario(self.probe, persistence=db)
        events = await self.collect(self.stream())
        self.assertEqual(events[-1]['type'], 'error')
        self.assertEqual(len(db.rows), 3)
        self.assertEqual(db.calls, 1)

    async def test_agent_failure_no_persistence(self):
        agent = fake.FakeAgent(self.probe, fail=True)
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        events = await self.collect(self.stream())
        self.assertEqual(events[-1]['code'], 'AGENT_EXECUTION_FAILED')
        self.assertEqual(self.scenario.persistence.calls, 0)
        self.assertEqual(agent.calls, 1)

    async def test_raw_error_stops_before_done(self):
        pulled = []
        def source(stop):
            yield {'type': 'error', 'content': 'secret traceback'}
            pulled.append('after_error')
            yield {'type': 'done'}
        events = await self.collect(self.stream(source))
        self.assertEqual([e['type'] for e in events], ['error'])
        self.assertNotIn('secret', json.dumps(events))
        self.assertEqual(pulled, [])

    async def test_duplicate_done_becomes_error(self):
        def source(stop):
            yield {'type': 'done'}
            yield {'type': 'done'}
        events = await self.collect(self.stream(source))
        self.assertEqual([e['type'] for e in events], ['error'])

    async def test_done_then_failure_never_success(self):
        def source(stop):
            yield {'type': 'done'}
            raise RuntimeError('private details')
        events = await self.collect(self.stream(source))
        self.assertEqual([e['type'] for e in events], ['error'])
        self.assertNotIn('private', json.dumps(events))

    async def test_eof_without_done_is_not_success(self):
        events = await self.collect(self.stream(lambda stop: iter([{'type': 'chunk', 'content': 'x'}])))
        self.assertEqual([e['type'] for e in events], ['chunk'])

    async def test_empty_eof_not_success(self):
        self.assertEqual(await self.collect(self.stream(lambda stop: iter(()))), [])

    async def test_full_queue_cancel_terminates_producer(self):
        stream = self.stream(capacity=1)
        await anext(stream)
        await probe_module.wait_until(lambda: stream.full_waits > 0)
        self.assertEqual(stream.snapshot()['queue_size'], 1)
        await stream.aclose(grace=.1)
        await probe_module.wait_until(stream._producer_done.is_set)
        self.assertLessEqual(stream.max_queue_observed, 1)
        self.assertEqual(stream.snapshot()['queue_size'], 0)
        self.assertEqual(self.scenario.persistence.calls, 0)

    async def test_slow_consumer_bounded_queue(self):
        stream = self.stream(capacity=2)
        events = []
        async for blob in stream:
            events.append(decode(blob)['type'])
            await asyncio.sleep(.003)
        self.assertEqual(events[-1], 'done')
        self.assertLessEqual(stream.max_queue_observed, 2)
        self.assertEqual(stream.snapshot()['queued_bytes_upper_bound'], 8192)

    async def test_oversize_event_sanitized(self):
        events = await self.collect(self.stream(lambda stop: iter([{'type': 'chunk', 'content': 'x' * 600}]),
                                                max_event_bytes=512))
        self.assertEqual([e['type'] for e in events], ['error'])

    async def test_bad_payload_sanitized(self):
        events = await self.collect(self.stream(lambda stop: iter([object()])))
        self.assertEqual([e['type'] for e in events], ['error'])

    async def test_nonfinite_json_sanitized(self):
        events = await self.collect(self.stream(lambda stop: iter([{'type': 'chunk', 'value': float('nan')}])))
        self.assertEqual([e['type'] for e in events], ['error'])

    async def test_cleanup_error_overrides_done(self):
        def source(stop):
            try:
                yield {'type': 'done'}
                yield {'type': 'error'}
            finally:
                raise RuntimeError('private cleanup')
        stream = self.stream(source)
        events = await self.collect(stream)
        self.assertEqual([e['type'] for e in events], ['error'])
        self.assertEqual(stream.cleanup_errors, 1)

    async def test_baseexception_observed_not_false_done(self):
        class SyntheticAbort(BaseException):
            pass
        def source(stop):
            yield {'type': 'done'}
            raise SyntheticAbort()
        events = await self.collect(self.stream(source))
        self.assertEqual([e['type'] for e in events], ['error'])
        await probe_module.wait_until(lambda: self.pool.unexpected_aborts == 1)

    async def test_cleanup_baseexception_overrides_done(self):
        class Source:
            def __init__(self):
                self.first = True
            def __iter__(self):
                return self
            def __next__(self):
                if self.first:
                    self.first = False
                    return {'type': 'done'}
                raise StopIteration
            def close(self):
                raise BaseException('synthetic abort')
        events = await self.collect(self.stream(lambda stop: Source()))
        self.assertEqual([e['type'] for e in events], ['error'])
        await probe_module.wait_until(lambda: self.pool.unexpected_aborts == 1)

    async def test_context_and_thread_affinity(self):
        marker = contextvars.ContextVar('synthetic_marker', default='unset')
        token = marker.set('local')
        seen = []
        def source(stop):
            seen.append((marker.get(), threading.get_ident()))
            yield from self.scenario.source(stop)
        try:
            await self.collect(self.stream(source))
        finally:
            marker.reset(token)
        ids = self.scenario.thread_ids + self.scenario.agent.thread_ids + self.scenario.persistence.thread_ids
        self.assertEqual(len(set(ids)), 1)
        self.assertNotEqual(ids[0], threading.get_ident())
        self.assertEqual(seen, [('local', ids[0])])

    async def test_finished_unread_stream_keeps_admission(self):
        stream = self.stream(lambda stop: iter([{'type': 'intent'}, {'type': 'done'}]))
        await anext(stream)
        await probe_module.wait_until(stream._producer_done.is_set)
        with self.assertRaises(bridge.CapacityExceeded):
            self.stream()
        await self.collect(stream)
        await probe_module.wait_until(lambda: self.pool.snapshot()['admitted'] == 0)

    async def test_repeated_close_is_idempotent(self):
        stream = self.stream()
        stream.close()
        stream.close()
        await stream.aclose()
        await stream.aclose()
        self.assertEqual([e[0] for e in self.probe.snapshot()].count('consumer_close'), 1)

    async def test_async_generator_finally(self):
        stream = self.stream()
        generator = bridge.encoded_events(stream)
        await anext(generator)
        await generator.aclose()
        self.assertTrue(stream.snapshot()['consumer_finished'])
        await probe_module.wait_until(stream._producer_done.is_set)

    async def test_send_side_cancel_needs_aclosing(self):
        stream = self.stream()
        entered = asyncio.Event()
        async def sender():
            async with contextlib.aclosing(bridge.encoded_events(stream)) as generator:
                async for blob in generator:
                    entered.set()
                    await asyncio.Event().wait()
        task = self.task(sender())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.snapshot()['consumer_finished'])

    async def test_consumer_deadline_reports_residual(self):
        agent = fake.FakeAgent(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        stream = self.stream(max_seconds=.05)
        await anext(stream)
        with self.assertRaises(bridge.PollDeadlineExceeded):
            await anext(stream)
        self.assertFalse(stream.snapshot()['producer_finished'])
        self.assertEqual(self.scenario.persistence.calls, 0)

    async def test_pool_close_reports_then_drains_residual(self):
        agent = fake.FakeAgent(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        stream = self.stream()
        await anext(stream)
        await probe_module.wait_until(agent.entered.is_set)
        state = await self.pool.aclose(grace=0)
        self.assertFalse(state['drained'])
        self.assertEqual(state['running_jobs'], 1)
        self.gates[0].set()
        state = await self.pool.aclose()
        self.assertTrue(state['drained'])
        self.assertEqual(state['workers_alive'], 0)

    async def test_probe_failure_does_not_break_stream(self):
        class Broken:
            def record(self, stage):
                raise RuntimeError('synthetic')
        stream = self.pool.open(self.scenario.source, probe=Broken())
        self.assertEqual((await self.collect(stream))[-1]['type'], 'done')

    async def test_no_unowned_async_tasks(self):
        before = set(asyncio.all_tasks())
        await self.collect(self.stream())
        await self.pool.aclose()
        await asyncio.sleep(0)
        self.assertEqual(set(asyncio.all_tasks()) - before, set())

    async def test_capacity_and_limits_reject_unbounded_config(self):
        for kwargs in ({'capacity': 0}, {'capacity': 9}, {'max_seconds': 0},
                       {'max_seconds': 6}, {'max_event_bytes': 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.stream(**kwargs)
        with self.assertRaises(ValueError):
            bridge.BridgePool(max_workers=3)

    async def test_missing_notifications_have_bounded_timer_fallback(self):
        stream = self.stream()
        stream._notify = lambda complete=False: None
        self.assertEqual((await self.collect(stream))[-1]['type'], 'done')

    async def test_concurrent_pulls_rejected(self):
        agent = fake.FakeAgent(self.probe, gate=self.gate())
        self.scenario = fake.FakeScenario(self.probe, agent=agent)
        stream = self.stream()
        await anext(stream)
        task = self.task(anext(stream))
        await probe_module.wait_until(lambda: stream._pulling)
        with self.assertRaises(RuntimeError):
            await anext(stream)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_two_worker_admission_is_strict(self):
        pool = bridge.BridgePool(max_workers=2)
        try:
            a = pool.open(lambda stop: iter([{'type': 'done'}]))
            b = pool.open(lambda stop: iter([{'type': 'done'}]))
            with self.assertRaises(bridge.CapacityExceeded):
                pool.open(lambda stop: iter(()))
            results = await asyncio.gather(self.collect(a), self.collect(b))
            self.assertEqual([r[-1]['type'] for r in results], ['done', 'done'])
            self.assertEqual(pool.max_admitted, 2)
        finally:
            state = await pool.aclose()
            self.assertTrue(state['joined'])
            self.assertEqual(state['workers_alive'], 0)

    async def test_client_done_delivery_not_asserted(self):
        stream = self.stream()
        self.assertEqual((await self.collect(stream))[-1]['type'], 'done')
        self.assertIsNone(stream.snapshot()['client_received_done'])


class StaticBoundaryTests(unittest.TestCase):
    def test_protected_p4b_files_byte_identical(self):
        assert_protected_p4b_files(ROOT)

    def test_poc_imports_stdlib_only(self):
        allowed = set(sys.stdlib_module_names)
        for path in POC.glob('*.py'):
            tree = ast.parse(path.read_text(), filename=path.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    modules = [node.module.split('.')[0]]
                else:
                    continue
                self.assertTrue(set(modules) <= allowed, (path.name, modules))

    def test_actual_module_global_bindings(self):
        paths = [*POC.glob('*.py'), ROOT / 'myapp/agent_views.py',
                 ROOT / 'myapp/services/agent_chat_service.py',
                 ROOT / 'myapp/services/agent_contracts.py']
        for path in paths:
            table = symtable.symtable(path.read_text(), path.name, 'exec')
            bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
            bound.update(dir(builtins))
            bound.update({'__name__', '__file__'})
            missing = set()
            def visit(scope):
                for symbol in scope.get_symbols():
                    if (symbol.is_referenced() and symbol.is_global() and not symbol.is_local()
                            and symbol.get_name() not in bound):
                        missing.add(symbol.get_name())
                for child in scope.get_children():
                    visit(child)
            visit(table)
            self.assertEqual(missing, set(), path.name)

    def test_no_application_imports_or_hidden_startup(self):
        for path in POC.glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self.assertNotIn(ast.unparse(node.func), {'django.setup', 'os.system', 'subprocess.run',
                                                             'eval', 'exec', 'socket.create_connection'})
        self.assertNotIn('experiments.asgi_streaming', (ROOT / 'myapp/agent_views.py').read_text())

    @unittest.skip('No audited Django installation for active Python; no framework initialization permitted')
    def test_real_django_streaming_response_is_async(self):
        """Deferred real framework layer; no Fake substitutes for this assertion."""


if __name__ == '__main__':
    unittest.main(verbosity=2)
