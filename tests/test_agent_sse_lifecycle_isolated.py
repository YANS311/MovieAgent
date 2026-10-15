"""P4-B stdlib/AST/FakeDB contracts, NOT HTTP disconnect integration tests."""
import ast
import asyncio
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
import weakref
from unittest.mock import Mock, patch

FORBIDDEN = {'django', 'redis', 'torch', 'transformers', 'sentence_transformers',
             'numpy', 'faiss', 'tensorflow', 'MySQLdb', 'pymysql', 'neo4j',
             'requests', 'ollama', 'langchain'}


class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Forbidden lifecycle test import')


sys.meta_path.insert(0, Guard())
ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'e60eb5a96662f9e448df571e5e0e495bd43ba737'
PRODUCTION_FILES = frozenset({
    'myapp/agent_views.py',
    'myapp/services/agent_chat_service.py',
    'myapp/services/agent_contracts.py',
})
REQUIRED_ADDITIONS = frozenset({
    'tests/test_agent_sse_lifecycle_isolated.py',
    'tests/test_asgi_streaming_poc_isolated.py',
    'docs/agent_backend/P4B_LIFECYCLE_AUDIT.md',
    'docs/agent_backend/P4B_CANCELLATION_SEMANTICS.md',
    'docs/agent_backend/P4B_SSE_CONTRACT.md',
    'docs/agent_backend/P4B_IMPLEMENTATION.md',
    'docs/agent_backend/P4B_TEST_REPORT.md',
    'docs/agent_backend/P4B_INTEGRATION_TEST_PLAN.md',
    'docs/agent_backend/P4B_GO_NO_GO.md',
    'docs/agent_backend/p4b1/DATABASE_AND_CANCELLATION.md',
    'docs/agent_backend/p4b1/INTEGRATION_TEST_PLAN.md',
    'docs/agent_backend/p4b1/ASGI_STREAMING_AUDIT.md',
    'docs/agent_backend/p4b1/POC_DESIGN.md',
    'docs/agent_backend/p4b1/ASYNC_ADAPTER_COMPARISON.md',
    'docs/agent_backend/p4b1/P4B_FROZEN_SHA256.json',
    'docs/agent_backend/p4b1/POC_TEST_REPORT.md',
    'docs/agent_backend/p4b1/P4B1_GO_NO_GO.md',
    'experiments/asgi_streaming/lifecycle_probe.py',
    'experiments/asgi_streaming/poc_evidence.json',
    'experiments/asgi_streaming/README.md',
    'experiments/asgi_streaming/fake_agent.py',
    'experiments/asgi_streaming/async_bridge_poc.py',
})
OPTIONAL_REPAIR_ADDITIONS = frozenset({
    'tests/test_git_freeze_states_isolated.py',
    'docs/agent_backend/p4b2/FREEZE_BLOCKER_FIX_REPORT.md',
})


def git_scope_records(root, *comparison):
    """NUL paths and disabled rename detection make deletes/renames explicit."""
    data = subprocess.check_output([
        'git', 'diff', '--no-ext-diff', '--no-textconv', '--no-renames',
        '--name-status', '-z', *comparison, '--',
    ], cwd=root, timeout=15)
    fields = data.split(b'\0')
    if fields[-1] == b'':
        fields.pop()
    if len(fields) % 2:
        raise AssertionError('Malformed Git name-status output')
    return [(fields[i].decode('ascii'), os.fsdecode(fields[i + 1]))
            for i in range(0, len(fields), 2)]


def assert_authorized_change_scope(root):
    """Check worktree/index/HEAD against the fixed baseline, plus all untracked paths."""
    additions = REQUIRED_ADDITIONS | OPTIONAL_REPAIR_ADDITIONS
    working = git_scope_records(root, BASELINE)
    staged = git_scope_records(root, '--cached', BASELINE)
    committed = git_scope_records(root, BASELINE, 'HEAD')
    unstaged = git_scope_records(root)
    for relation, records in [('baseline/worktree', working),
                              ('baseline/index', staged), ('baseline/HEAD', committed),
                              ('index/worktree', unstaged)]:
        for status, name in records:
            permitted = ({'M'} if name in PRODUCTION_FILES else
                         ({'A', 'M'} if relation == 'index/worktree' else {'A'})
                         if name in additions else set())
            if status not in permitted:
                raise AssertionError(f'Unauthorized Git change: {relation} {status} {name}')
    untracked = {os.fsdecode(name) for name in subprocess.check_output([
        'git', 'ls-files', '--others', '-z',
    ], cwd=root, timeout=15).split(b'\0') if name}
    if untracked - additions:
        raise AssertionError(f'Unauthorized untracked paths: {sorted(untracked - additions)}')
    if {name for _, name in working if name in PRODUCTION_FILES} != PRODUCTION_FILES:
        raise AssertionError('The three frozen production changes must remain present')
    present = {name for _, name in working} | untracked
    if REQUIRED_ADDITIONS - present:
        raise AssertionError(f'Missing stage files: {sorted(REQUIRED_ADDITIONS - present)}')
    for name in PRODUCTION_FILES | REQUIRED_ADDITIONS | (OPTIONAL_REPAIR_ADDITIONS & present):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise AssertionError(f'Frozen path must remain a regular file: {name}')


spec = importlib.util.spec_from_file_location('p4b_p4a_fixture', ROOT / 'tests/test_agent_backend_p4a_isolated.py')
p4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4a)
contracts = p4a.contracts


def decode(wire):
    return json.loads(wire.removeprefix('data: ').strip())


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        # Reuse the exact accepted FakeDB/service AST fixture, no Django imports.
        self.env = p4a.P4ATests()
        self.env.setUp()
        module_patch = patch.dict(sys.modules, self.env.modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.response = self.env.views['agent_stream_view'](self.env.request)
        self.stream = self.response.streaming_content
        self.life = self.response.sse_lifecycle
        self.addCleanup(self.stream.close)

    def drain(self):
        return [decode(wire) for wire in self.stream]

    def until(self, event_type):
        seen = []
        for _ in range(100):
            event = decode(next(self.stream))
            seen.append(event)
            if event['type'] == event_type:
                return seen
        self.fail('Bounded event search exceeded')

    def producer(self, factory):
        service = types.SimpleNamespace(stream_chat=factory)
        self.env.provider.side_effect = lambda *args: service
        return service

    def assert_no_writes(self):
        self.assertEqual(self.env.db.attempts, 0)
        self.assertEqual(self.env.db.rows, {'history': [], 'trace': []})

    def test_response_is_lazy_and_session_ready(self):
        self.assertEqual(self.life.state, 'VALIDATED')
        self.env.provider.assert_not_called()
        self.assertEqual(self.env.agent.calls, 0)
        self.assert_no_writes()
        self.assertEqual(self.env.request.session.session_key, 'synthetic-private-session')

    def test_first_event_precedes_agent_execution(self):
        self.assertEqual(decode(next(self.stream))['type'], 'intent')
        self.assertEqual(self.env.agent.calls, 0)
        self.assert_no_writes()

    def test_success_exactly_one_done_and_three_owned_writes(self):
        events = self.drain()
        self.assertEqual(sum(e['type'] == 'done' for e in events), 1)
        self.assertNotIn('error', [e['type'] for e in events])
        self.assertEqual(self.env.agent.calls, 1)
        self.assertEqual(self.env.db.attempts, 3)
        self.assertEqual(len(self.env.db.rows['history']), 2)
        self.assertEqual(len(self.env.db.rows['trace']), 1)
        self.assertTrue(self.life.persistence_confirmed)
        self.assertTrue(self.life.iteration_exhausted)
        self.assertEqual(self.life.state, 'COMPLETED')

    def test_success_handoff_is_not_client_acknowledgment(self):
        self.until('done')
        snapshot = self.life.snapshot()
        self.assertTrue(snapshot['agent_completed'])
        self.assertTrue(snapshot['persistence_confirmed'])
        self.assertTrue(snapshot['done_handed_off'])
        self.assertFalse(snapshot['iteration_exhausted'])
        self.assertIsNone(snapshot['client_received_done'])
        self.assertIsNone(snapshot['http_disconnect_observed'])

    def test_progress_tracks_running_replay_and_persistence(self):
        observed = []
        run = self.env.agent.run
        create = self.env.db.create
        def running(prompt):
            observed.append(self.life.state)
            return run(prompt)
        def write(*args, **kwargs):
            observed.append(self.life.state)
            return create(*args, **kwargs)
        self.env.agent.run = running
        self.env.db.create = write
        self.until('thought')
        self.assertEqual(self.life.state, 'REPLAYING')
        self.drain()
        self.assertEqual(observed, ['RUNNING', 'PERSISTING', 'PERSISTING', 'PERSISTING'])
        self.assertEqual(self.life.state, 'COMPLETED')

    def test_request_id_matches_service_and_error(self):
        self.env.agent.error = RuntimeError('private details')
        events = self.drain()
        request_id = self.response.headers['X-Request-ID']
        self.assertEqual(events[-1]['request_id'], request_id)
        self.assertEqual(self.env.instances[-1].request_id, request_id)
        self.assertEqual(self.life.trace.request_id, request_id)

    def test_agent_failure_is_failed_without_done_or_persistence(self):
        self.env.agent.error = RuntimeError('private internal details')
        events = self.drain()
        self.assertEqual([e['type'] for e in events], ['intent', 'error'])
        self.assertEqual(events[-1]['code'], 'AGENT_EXECUTION_FAILED')
        self.assertFalse(events[-1]['retryable'])
        self.assertNotIn('private', json.dumps(events))
        self.assertEqual(self.life.state, 'FAILED')
        self.assertFalse(self.life.agent_completed)
        self.assert_no_writes()

    def test_explicit_dependency_failure_keeps_classification(self):
        self.env.agent.error = contracts.AgentDependencyUnavailable('private backend')
        events = self.drain()
        self.assertEqual(events[-1]['code'], 'AGENT_DEPENDENCY_UNAVAILABLE')
        self.assertEqual(self.env.agent.calls, 1)
        self.assert_no_writes()

    def write_failure(self, index):
        self.env.db.fail_write = index
        events = self.drain()
        self.assertEqual(events[-1]['code'], 'AGENT_PERSISTENCE_FAILED')
        self.assertIn('chunk', [e['type'] for e in events])
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertEqual(self.env.db.attempts, index)
        self.assertEqual(self.env.agent.calls, 1)
        self.assertEqual(self.env.db.rows, {'history': [], 'trace': []})
        self.assertIn(('rollback', 'default'), self.env.db.events)
        self.assertFalse(self.life.persistence_confirmed)
        self.assertEqual(self.life.state, 'FAILED')

    def test_first_history_write_failure(self):
        self.write_failure(1)

    def test_second_history_write_failure(self):
        self.write_failure(2)

    def test_trace_write_failure(self):
        self.write_failure(3)

    def test_commit_ack_loss_not_reported_as_confirmed_rollback_or_done(self):
        self.env.db.commit_fault = 'after'
        events = self.drain()
        self.assertEqual(events[-1]['code'], 'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(len(self.env.db.rows['trace']), 1)
        self.assertFalse(self.life.persistence_confirmed)
        self.assertFalse(self.life.done_handed_off)
        self.assertEqual(self.env.agent.calls, 1)

    def test_prestart_close_is_observable_and_does_not_construct_service(self):
        self.stream.close()
        self.assertEqual(list(self.stream), [])
        self.env.provider.assert_not_called()
        self.assertEqual(self.life.state, 'CANCELLED')
        self.assertTrue(self.life.closed)
        self.assertFalse(self.life.iteration_exhausted)
        self.assert_no_writes()

    def test_local_cancel_before_first_yield(self):
        self.life.cancel()
        self.assertEqual(self.drain(), [])
        self.env.provider.assert_not_called()
        self.assert_no_writes()

    def test_close_after_intent_prevents_agent(self):
        next(self.stream)
        self.stream.close()
        self.assertEqual(self.drain(), [])
        self.assertEqual(self.env.agent.calls, 0)
        self.assert_no_writes()

    def test_cancel_during_agent_waits_for_return_without_persistence(self):
        original = self.env.agent.run
        def run(prompt):
            self.life.cancel()
            result = original(prompt)  # Synchronous work still executes/returns.
            self.assertEqual(self.life.state, 'CANCELLED')
            return result
        self.env.agent.run = run
        self.assertEqual([e['type'] for e in self.drain()], ['intent'])
        self.assertEqual(self.env.agent.calls, 1)
        self.assertTrue(self.life.agent_completed)
        self.assert_no_writes()

    def test_close_inside_active_next_defers_generator_close(self):
        original = self.env.agent.run
        def run(prompt):
            self.stream.close()
            self.assertFalse(self.life.closed)  # next owns cleanup after return.
            return original(prompt)
        self.env.agent.run = run
        self.assertEqual([e['type'] for e in self.drain()], ['intent'])
        self.assertTrue(self.life.closed)
        self.assert_no_writes()

    def test_cancel_during_trace_replay_stops_before_other_frames(self):
        self.until('thought')
        self.life.cancel()
        self.assertEqual(self.drain(), [])
        self.assert_no_writes()
        self.assertFalse(self.life.done_handed_off)

    def test_close_mid_answer_does_not_persist(self):
        self.until('chunk')
        self.stream.close()
        self.assertEqual(self.drain(), [])
        self.assert_no_writes()

    def test_cancel_just_before_persistence(self):
        self.env.agent.get_react_trace = lambda result: self.life.cancel() or 'display'
        events = self.drain()
        self.assertIn('chunk', [e['type'] for e in events])
        self.assertFalse(self.life.persistence_started)
        self.assert_no_writes()

    def test_cancel_during_transaction_does_not_force_rollback(self):
        original = self.env.db.create
        def write(*args, **kwargs):
            self.assertEqual(self.env.db.depth, 1)
            self.life.cancel()
            return original(*args, **kwargs)
        self.env.db.create = write
        events = self.drain()
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertNotIn('error', [e['type'] for e in events])
        self.assertEqual(len(self.env.db.rows['history']), 2)
        self.assertEqual(len(self.env.db.rows['trace']), 1)
        self.assertTrue(self.life.persistence_confirmed)
        self.assertEqual(self.life.state, 'CANCELLED')
        self.assertEqual(self.env.agent.calls, 1)

    def test_close_during_transaction_does_not_reenter_generator(self):
        original = self.env.db.create
        def write(*args, **kwargs):
            self.stream.close()
            return original(*args, **kwargs)
        self.env.db.create = write
        events = self.drain()
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertEqual(self.env.db.attempts, 3)
        self.assertTrue(self.life.closed)
        self.assertTrue(self.life.persistence_confirmed)

    def test_cancel_and_database_failure_suppress_unwritable_error_frame(self):
        original = self.env.db.create
        self.env.db.fail_write = 2
        def write(*args, **kwargs):
            self.life.cancel()
            return original(*args, **kwargs)
        self.env.db.create = write
        events = self.drain()
        self.assertNotIn('error', [e['type'] for e in events])
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertEqual(self.env.db.rows, {'history': [], 'trace': []})
        self.assertEqual(self.life.failure_code, 'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(self.life.state, 'CANCELLED')

    def test_cancel_after_commit_before_done_preserves_committed_rows(self):
        original = self.life.persistence_succeeded
        def committed():
            original()
            self.life.cancel()
        self.life.persistence_succeeded = committed
        events = self.drain()
        self.assertTrue(self.life.persistence_confirmed)
        self.assertEqual(len(self.env.db.rows['trace']), 1)
        self.assertNotIn('done', [e['type'] for e in events])

    def test_close_after_done_before_iterator_exhaustion_is_not_delivery_success(self):
        self.until('done')
        self.stream.close()
        self.assertEqual(self.life.state, 'CANCELLED')
        self.assertTrue(self.life.done_handed_off)
        self.assertTrue(self.life.persistence_confirmed)
        self.assertFalse(self.life.iteration_exhausted)
        self.assertIsNone(self.life.snapshot()['client_received_done'])
        self.assertEqual(self.env.db.attempts, 3)

    def test_normal_response_close_does_not_turn_completed_into_cancelled(self):
        self.drain()
        self.stream.close()
        self.stream.close()
        self.assertEqual(self.life.state, 'COMPLETED')
        self.assertFalse(self.life.snapshot()['cancel_requested'])

    def test_repeated_cancel_and_close_are_idempotent(self):
        with self.assertLogs('movie_agent.request', level='INFO') as records:
            self.life.cancel()
            self.life.cancel()
            self.stream.close()
            self.stream.close()
            self.drain()
        finished = [r for r in records.records if r.stage == 'request_finished']
        self.assertEqual(len(finished), 1)
        self.assertEqual(self.life.cancel_source, 'local_cancel')

    def test_retained_inner_generator_is_explicitly_closed_once(self):
        closed = []
        def generate(*args):
            try:
                yield {'type': 'intent', 'content': 'synthetic'}
                yield {'type': 'chunk', 'content': 'synthetic'}
            finally:
                closed.append('inner-finally')
        retained = generate()
        self.producer(lambda *args: retained)
        next(self.stream)
        self.stream.close()
        self.stream.close()
        self.assertEqual(closed, ['inner-finally'])
        self.assertEqual(list(retained), [])

    def test_close_never_resumes_code_after_yield(self):
        resumed = []
        def generate(*args):
            yield {'type': 'chunk', 'content': 'synthetic'}
            resumed.append('must-not-run')
        self.producer(generate)
        next(self.stream)
        self.stream.close()
        self.assertEqual(resumed, [])
        self.assertEqual(self.drain(), [])

    def test_generator_exit_propagates_without_error_or_done(self):
        def generate(*args):
            yield {'type': 'intent', 'content': 'synthetic'}
            raise GeneratorExit
        self.producer(generate)
        next(self.stream)
        with self.assertRaises(GeneratorExit):
            next(self.stream)
        self.assertEqual(self.life.state, 'CANCELLED')
        self.assertEqual(self.life.cancel_source, 'generator_exit')
        self.assertTrue(self.life.closed)
        self.assertFalse(self.life.error_handed_off)

    def test_base_exception_is_not_swallowed_or_sent_as_business_error(self):
        def generate(*args):
            raise KeyboardInterrupt
            yield  # Only define a generator; no threads/process signals are used.
        self.producer(generate)
        with self.assertRaises(KeyboardInterrupt):
            next(self.stream)
        self.assertEqual(self.life.state, 'FAILED')
        self.assertEqual(self.life.failure_code, 'SSE_ABORTED')
        self.assertTrue(self.life.closed)

    def test_async_cancel_injected_into_sync_code_is_not_claimed_as_disconnect(self):
        def generate(*args):
            raise asyncio.CancelledError
            yield
        self.producer(generate)
        with self.assertRaises(asyncio.CancelledError):
            next(self.stream)
        self.assertIsNone(self.life.snapshot()['http_disconnect_observed'])
        self.assertFalse(self.life.done_handed_off)

    def test_cleanup_error_is_recorded_without_private_details_or_repeated_close(self):
        close = Mock(side_effect=RuntimeError('synthetic-private-cleanup-credentials'))
        class Inner:
            def __iter__(self): return self
            def __next__(self): return {'type': 'intent', 'content': 'synthetic'}
        inner = Inner()
        inner.close = close
        self.producer(lambda *args: inner)
        next(self.stream)
        with self.assertLogs('movie_agent.sse', level='INFO') as records:
            self.stream.close()
            self.stream.close()
        close.assert_called_once()
        self.assertEqual(self.life.cleanup_errors, 1)
        self.assertEqual(self.life.failure_code, 'SSE_CLEANUP_FAILED')
        self.assertNotIn('synthetic-private', '\n'.join(records.output))

    def test_cleanup_base_exception_propagates_and_finish_still_executes(self):
        class Inner:
            def __iter__(self): return self
            def __next__(self): return {'type': 'intent', 'content': 'synthetic'}
            def close(self): raise SystemExit('synthetic')
        self.producer(lambda *args: Inner())
        next(self.stream)
        with self.assertRaises(SystemExit): self.stream.close()
        self.assertTrue(self.life.closed)
        self.assertFalse(self.life.done_handed_off)

    def test_broken_log_handlers_do_not_break_success_or_cleanup(self):
        with (patch.object(contracts.logger, 'info', side_effect=RuntimeError('log')),
              patch.object(contracts.sse_logger, 'info', side_effect=RuntimeError('log'))):
            events = self.drain()
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(self.life.state, 'COMPLETED')

    def test_broken_trace_stage_does_not_break_prestart_close(self):
        self.life.trace.stage = Mock(side_effect=RuntimeError('log'))
        self.stream.close()
        self.assertTrue(self.life.closed)

    def test_logs_and_snapshot_have_no_payload_user_or_session(self):
        with self.assertLogs('movie_agent.sse', level='INFO') as records:
            self.drain()
        output = '\n'.join(records.output) + repr(self.life.snapshot())
        for private in ('synthetic-private-prompt', 'synthetic private answer',
                        'synthetic-private-user', 'synthetic-private-session'):
            self.assertNotIn(private, output)
        self.assertIn(self.response.headers['X-Request-ID'], output)

    def test_unknown_exception_has_one_safe_error_and_no_done(self):
        self.env.provider.side_effect = RuntimeError('private SQL redis://credentials')
        events = self.drain()
        self.assertEqual([e['type'] for e in events], ['error'])
        self.assertEqual(events[0]['code'], 'INTERNAL_ERROR')
        self.assertNotIn('credentials', json.dumps(events))
        self.assert_no_writes()

    def test_error_frame_closes_source_without_pulling_later_done(self):
        pulled = []
        def generate(*args):
            yield {'type': 'error', 'content': 'safe error'}
            pulled.append('after-error')
            yield {'type': 'done', 'content': ''}
        self.producer(generate)
        self.assertEqual([e['type'] for e in self.drain()], ['error'])
        self.assertEqual(pulled, [])
        self.assertFalse(self.life.done_handed_off)

    def test_unexpected_eof_is_incomplete_not_success(self):
        self.producer(lambda *args: iter([{'type': 'chunk', 'content': 'synthetic'}]))
        events = self.drain()
        self.assertEqual([e['type'] for e in events], ['chunk'])
        self.assertEqual(self.life.state, 'INCOMPLETE')
        self.assertFalse(self.life.done_handed_off)

    def test_empty_source_is_incomplete(self):
        self.producer(lambda *args: iter(()))
        self.assertEqual(self.drain(), [])
        self.assertEqual(self.life.state, 'INCOMPLETE')

    def test_post_done_failure_is_error_and_frontend_can_reject_success(self):
        def generate(*args):
            yield {'type': 'done', 'content': ''}
            raise RuntimeError('private post-done failure')
        self.producer(generate)
        self.assertEqual([e['type'] for e in self.drain()], ['done', 'error'])
        self.assertEqual(self.life.state, 'FAILED')

    def test_invalid_event_gets_safe_error(self):
        self.producer(lambda *args: iter([None]))
        events = self.drain()
        self.assertEqual([e['type'] for e in events], ['error'])
        self.assertEqual(events[0]['code'], 'INTERNAL_ERROR')

    def test_encoding_failure_is_mapped_without_replaying_producer(self):
        self.producer(lambda *args: iter([{'type': 'chunk', 'content': object()}]))
        events = self.drain()
        self.assertEqual([e['type'] for e in events], ['error'])
        self.env.provider.assert_called_once()
        self.assertFalse(self.life.done_handed_off)

    def test_shared_resources_are_not_closed(self):
        shared = Mock()
        self.env.agent.close = shared.agent_close
        self.env.db.close = shared.db_close
        next(self.stream)
        self.stream.close()
        self.assertEqual(shared.mock_calls, [])

    def test_factory_reference_released_even_before_first_next(self):
        factory = Mock(return_value=iter(()))
        reference = weakref.ref(factory)
        stream = contracts.AgentRequestTrace().sse_lifecycle().wrap(factory, lambda e: e)
        del factory
        self.assertIsNotNone(reference())
        stream.close()
        self.assertIsNone(reference())

    def test_reentrant_next_is_rejected_without_duplicate_execution(self):
        original = self.env.agent.run
        def run(prompt):
            with self.assertRaisesRegex(RuntimeError, 'Concurrent'):
                next(self.stream)
            return original(prompt)
        self.env.agent.run = run
        self.drain()
        self.assertEqual(self.env.agent.calls, 1)
        self.assertEqual(self.env.db.attempts, 3)

    def test_response_construction_failure_closes_before_agent_creation(self):
        with patch.dict(self.env.views['agent_stream_view'].__globals__,
                        {'StreamingHttpResponse': Mock(side_effect=RuntimeError('response'))}):
            with self.assertRaisesRegex(RuntimeError, 'response'):
                self.env.views['agent_stream_view'](self.env.request)
        self.env.provider.assert_not_called()
        self.assert_no_writes()

    def test_pre_response_sanitizer_failure_finishes_without_hiding_exception(self):
        self.env.sanitize.side_effect = ValueError('private sanitizer details')
        with self.assertLogs('movie_agent.request', level='INFO') as records:
            with self.assertRaises(ValueError):
                self.env.views['agent_stream_view'](self.env.request)
        self.assertEqual(sum(r.stage == 'request_finished' for r in records.records), 1)
        self.env.provider.assert_not_called()
        self.assertNotIn('private sanitizer', '\n'.join(records.output))

    def test_pre_response_session_failure_still_propagates(self):
        self.env.request.session.session_key = None
        self.env.request.session.create = Mock(side_effect=ConnectionError('private session'))
        with self.assertLogs('movie_agent.request', level='INFO') as records:
            with self.assertRaises(ConnectionError):
                self.env.views['agent_stream_view'](self.env.request)
        self.assertEqual(sum(r.stage == 'request_finished' for r in records.records), 1)
        self.env.provider.assert_not_called()

    def test_response_close_registration_shape_without_running_django(self):
        # Model only Django's documented iterable/closer API, not middleware/ASGI.
        def response(iterator, **kwargs):
            return types.SimpleNamespace(streaming_content=map(str, iter(iterator)),
                                         close=iterator.close, headers=kwargs['headers'])
        with patch.dict(self.env.views['agent_stream_view'].__globals__,
                        {'StreamingHttpResponse': response}):
            response = self.env.views['agent_stream_view'](self.env.request)
        response.close()
        self.assertEqual(list(response.streaming_content), [])
        self.assertEqual(response.sse_lifecycle.state, 'CANCELLED')
        self.env.provider.assert_not_called()

    def test_cleanup_failure_after_done_does_not_claim_all_resources_released(self):
        class Source:
            def __init__(self): self.events = iter([{'type': 'done', 'content': ''}])
            def __iter__(self): return self
            def __next__(self): return next(self.events)
            def close(self): raise RuntimeError('private close details')
        self.producer(lambda *args: Source())
        self.assertEqual([e['type'] for e in self.drain()], ['done'])
        self.assertEqual(self.life.state, 'FAILED')
        self.assertEqual(self.life.cleanup_errors, 1)
        self.assertTrue(self.life.done_handed_off)
        self.assertIsNone(self.life.snapshot()['client_received_done'])

    def test_failure_cause_remains_chained_in_original_persistence_coordinator(self):
        service = self.env.new_service()
        self.env.db.fail_write = 1
        with self.assertRaises(contracts.AgentPersistenceError) as caught:
            service._persist_result('synthetic', copy.deepcopy(self.env.result))
        self.assertIsInstance(caught.exception.__cause__, p4a.FakeDBError)

    def test_empty_greeting_retains_done_without_business_writes(self):
        self.env.request.POST = {'msg': ''}
        response = self.env.views['agent_stream_view'](self.env.request)
        self.assertEqual([decode(x)['type'] for x in response.streaming_content], ['done'])
        self.assertEqual(response.sse_lifecycle.state, 'COMPLETED')
        self.assertFalse(response.sse_lifecycle.persistence_confirmed)
        self.env.provider.assert_not_called()

    def test_security_block_retains_legacy_error_and_no_service(self):
        self.env.sanitize.side_effect = lambda text: 'MALICIOUS_INJECTION_DETECTED'
        response = self.env.views['agent_stream_view'](self.env.request)
        events = [decode(x) for x in response.streaming_content]
        self.assertEqual(events, [{'type': 'error', 'content': '🛡️ 系统拦截：检测到不安全的输入内容。'}])
        self.assertEqual(response.sse_lifecycle.state, 'FAILED')
        self.env.provider.assert_not_called()

    def test_plain_json_api_has_no_lifecycle_object_or_new_return_fields(self):
        response = self.env.api()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('sse_lifecycle', response)
        self.assertFalse(hasattr(self.env.instances[-1], '_sse_lifecycle'))

    def test_only_authorized_production_files_changed(self):
        assert_authorized_change_scope(ROOT)

    def test_inference_json_and_transaction_methods_are_byte_frozen(self):
        path = 'myapp/services/agent_chat_service.py'
        baseline = subprocess.check_output(['git', 'show', BASELINE + ':' + path], text=True, cwd=ROOT)
        current = (ROOT / path).read_text()
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
            return {n.name: ast.get_source_segment(source, n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(baseline), methods(current)
        for name in before.keys() - {'stream_chat'}:
            self.assertEqual(after[name], before[name], name)

    def test_new_contracts_have_real_symbol_bindings_and_stdlib_only(self):
        for path in ('myapp/services/agent_contracts.py', 'myapp/services/agent_chat_service.py',
                     'myapp/agent_views.py'):
            self.assertEqual(p4a.p31.unbound_globals((ROOT / path).read_text()) - {'__file__'}, set())
        for path in ('myapp/services/agent_contracts.py', 'tests/test_agent_sse_lifecycle_isolated.py'):
            for node in ast.walk(ast.parse((ROOT / path).read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertIn(name.split('.')[0], sys.stdlib_module_names)

    def test_no_asgi_disconnect_or_hard_timeout_guarantee_is_invented(self):
        self.drain()
        facts = self.life.snapshot()
        self.assertIsNone(facts['http_disconnect_observed'])
        self.assertIsNone(facts['client_received_done'])
        source = (ROOT / 'myapp/services/agent_contracts.py').read_text()
        self.assertNotIn('.start(', source)  # No background thread/timer task.


if __name__ == '__main__':
    if not sys.flags.no_site or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(name) != '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise RuntimeError('CPU-only python3 -S required')
    unittest.main(verbosity=2)
