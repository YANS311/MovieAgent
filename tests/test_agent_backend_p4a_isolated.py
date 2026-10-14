"""CPU-only AST + FakeDB contracts. No Django setup, SQL, Redis or real agent."""
import ast
import copy
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch
from typing import Any, Dict, Generator, Optional
import uuid

FORBIDDEN = {'django', 'redis', 'torch', 'transformers', 'sentence_transformers', 'numpy',
             'faiss', 'tensorflow', 'MySQLdb', 'neo4j', 'requests', 'ollama', 'langchain'}


class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Forbidden import: ' + fullname)


sys.meta_path.insert(0, Guard())
ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'e33fdd4f63f6f50778e50d13bbed18433b02b15f'


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p0 = load_file('p4a_p0_fixture', 'tests/test_redis_cache_p0_isolated.py')
contracts = p0.contracts
p31 = load_file('p4a_symbol_checker', 'tests/test_redis_structures_p31_isolated.py')


class FakeDBError(Exception):
    pass


class FakeDB:
    """Committed rows are separate from transaction-local rows; not a MySQL proof."""
    def __init__(self):
        self.rows = {'history': [], 'trace': []}
        self.frames = []
        self.attempts = 0
        self.fail_write = None
        self.commit_fault = None
        self.events = []
        self.aliases = {'ChatHistory': 'default', 'AgentTrace': 'default'}

    @property
    def depth(self):
        return len(self.frames)

    def db_for_write(self, model, **hints):
        return self.aliases[model.__name__]

    def get_connection(self, using=None):
        return types.SimpleNamespace(in_atomic_block=bool(self.frames))

    @contextmanager
    def atomic(self, using=None):
        assert using == 'default'
        frame = copy.deepcopy(self.frames[-1] if self.frames else self.rows)
        self.frames.append(frame)
        self.events.append(('begin', using))
        try:
            yield
        except BaseException:
            self.frames.pop()
            self.events.append(('rollback', using))
            raise
        else:
            self.frames.pop()
            if self.commit_fault == 'before':
                self.events.append(('commit-not-confirmed', using))
                raise FakeDBError('synthetic SQL/password/model-path')
            if self.frames:
                self.frames[-1] = frame
            else:
                self.rows = frame
            self.events.append(('commit', using))
            if self.commit_fault == 'after':
                raise FakeDBError('synthetic commit acknowledgement lost')

    def create(self, table, alias='default', **kwargs):
        self.attempts += 1
        self.events.append(('write', table, alias, self.depth))
        if self.attempts == self.fail_write:
            raise FakeDBError('synthetic SQL/password/model-path')
        target = self.frames[-1] if self.frames else self.rows
        # Store only role/fields for assertions; the user fixture is not deep-copied.
        row = {k: copy.deepcopy(v) for k, v in kwargs.items() if k != 'user'}
        row['pk'] = len(target[table]) + 1
        target[table].append(row)
        return types.SimpleNamespace(**row)

    def models(self):
        db = self
        class Manager:
            _db = None
            def __init__(self, table, alias='default'):
                self.table, self.alias = table, alias
            def using(self, alias):
                return Manager(self.table, alias)
            def create(self, **kwargs):
                return db.create(self.table, self.alias, **kwargs)
        history = type('ChatHistory', (), {'objects': Manager('history')})
        trace = type('AgentTrace', (), {'objects': Manager('trace')})
        return history, trace


class Response(dict):
    def __init__(self, data, status=200, **kwargs):
        super().__init__(data)
        self.status_code = status
        self.content = json.dumps(data, ensure_ascii=False).encode()


class StreamResponse:
    def __init__(self, iterator, content_type=None, headers=None, status=200):
        self.streaming_content = iterator
        self.headers = headers or {}
        self.status_code = status


class FakeAgent:
    def __init__(self, db, result):
        self.db, self.result = db, result
        self.calls = 0
        self.error = None
    def run(self, prompt):
        assert self.db.depth == 0, 'Inference must never be inside atomic'
        self.calls += 1
        self.db.events.append(('agent', self.db.depth))
        if self.error: raise self.error
        return copy.deepcopy(self.result)
    def get_react_trace(self, result):
        assert self.db.depth == 0
        return 'display trace'


class P4ATests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        self.user = types.SimpleNamespace(id=7, username='synthetic-private-user')
        self.result = {'final_answer': 'synthetic private answer', 'thought': 'synthetic thought',
                       'actions': [{'tool': 'existing'}], 'observations': [{'tool': 'existing', 'count': 2}],
                       'recommended_ids': [9, 2, 3], 'explanations': {9: 'reason'}, 'intent': 'TEST',
                       'latency_ms': 12, 'need_clarification': False, 'clarification_options': [],
                       'trace_steps': [{'type': 'thought', 'content': 'thinking', 'step': 1},
                                       {'type': 'action', 'content': 'tool', 'tool': 'existing', 'input': 'x'},
                                       {'type': 'observation', 'content': 'found', 'count': 2},
                                       {'type': 'clarification', 'content': 'choose'},
                                       {'type': 'final_answer', 'content': 'ignored here'}]}
        self.agent = FakeAgent(self.db, self.result)
        self.history, self.trace_model = self.db.models()
        self.modules = {'myapp': p0.fake_module('myapp'),
                        'myapp.services': p0.fake_module('myapp.services'),
                        'myapp.models': p0.fake_module('myapp.models', ChatHistory=self.history),
                        'myapp.models_upgrade': p0.fake_module('myapp.models_upgrade', AgentTrace=self.trace_model),
                        'myapp.utils': p0.fake_module('myapp.utils'),
                        'myapp.utils.xai_explainer': p0.fake_module('myapp.utils.xai_explainer',
                                                                  translate_react_to_human=lambda data: [])}
        self.time = types.SimpleNamespace(time=lambda: 0, perf_counter=lambda: 0, sleep=Mock())
        ns = {'Optional': Optional, 'Dict': Dict, 'Any': Any, 'Generator': Generator,
              'time': self.time, 'logger': logging.getLogger('isolated.agent'),
              'transaction': self.db, 'router': self.db, 'DEFAULT_DB_ALIAS': 'default'}
        self.Service = p0.extract('myapp/services/agent_chat_service.py', {'AgentChatService'}, ns)['AgentChatService']
        self.instances = []
        self.provider = Mock(side_effect=self.new_service)
        self.modules['myapp.services.agent_chat_service'] = p0.fake_module(
            'myapp.services.agent_chat_service', AgentChatService=self.provider)
        self.sanitize = Mock(side_effect=lambda text: text)
        view_ns = {'JsonResponse': Response, 'StreamingHttpResponse': StreamResponse,
                   'time': self.time, 'json': json, '_sanitize_input': self.sanitize,
                   'logger': logging.getLogger('isolated.agent')}
        self.views = p0.extract('myapp/agent_views.py',
            {'agent_api_view', 'agent_stream_view', '_get_session_key', '_sse_data'}, view_ns)
        self.request = types.SimpleNamespace(method='POST', POST={'msg': 'synthetic-private-prompt'},
            user=self.user, headers={'X-Request-ID': 'untrusted-client-token'},
            session=types.SimpleNamespace(session_key='synthetic-private-session'))

    def new_service(self, user=None, session_key=None, request_id=None, legacy=False):
        cls = self.Service
        if legacy:
            text = subprocess.check_output(['git', 'show', BASELINE+':myapp/services/agent_chat_service.py'],
                                           cwd=ROOT, text=True)
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name=='AgentChatService')
            ns = {'Optional':Optional, 'Dict':Dict, 'Any':Any, 'Generator':Generator,
                  'time':self.time, 'logger':logging.getLogger('isolated.legacy'), 'transaction':self.db}
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy-chat-service', 'exec'), ns)
            cls = ns['AgentChatService']
        service = cls(user or self.user, session_key or 'synthetic-private-session')
        service.request_id = request_id
        service._get_agent = lambda: self.agent
        def excluded():
            self.assertEqual(self.db.depth, 0)
            return {2}
        service._get_excluded_ids = excluded
        def movies(ids):
            self.assertEqual(self.db.depth, 0)
            return {'9': {'title': 'synthetic movie', 'poster': '/media/posters/synthetic.jpg'}}
        service._format_movie_details = movies
        self.instances.append(service)
        return service

    def api(self):
        with patch.dict(sys.modules, self.modules):
            return self.views['agent_api_view'](self.request)

    def stream(self):
        with patch.dict(sys.modules, self.modules):
            response = self.views['agent_stream_view'](self.request)
            events = [json.loads(line.removeprefix('data: ').strip()) for line in response.streaming_content]
        return response, events

    def assert_uuid(self, value):
        self.assertEqual(str(uuid.UUID(value)), value)
        self.assertNotEqual(value, 'untrusted-client-token')

    def test_three_writes_commit_together_and_one_agent_execution(self):
        response = self.api()
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row['role'] for row in self.db.rows['history']], ['user', 'ai'])
        self.assertEqual(len(self.db.rows['trace']), 1)
        self.assertEqual(self.db.attempts, 3)
        self.assertEqual(self.agent.calls, 1)
        self.assertTrue(all(e[-1]==1 for e in self.db.events if e[0]=='write'))
        self.assertEqual([e[0] for e in self.db.events], ['agent','begin','write','write','write','commit'])

    def test_first_history_failure_rolls_back_visible_rows(self):
        self.db.fail_write = 1
        response = self.api()
        self.assertEqual((response.status_code,response['code']), (500,'AGENT_PERSISTENCE_FAILED'))
        self.assertEqual(self.db.rows, {'history':[], 'trace':[]})
        self.assertEqual(self.db.attempts, 1)
        self.assertIn(('rollback','default'), self.db.events)
        self.assertEqual(self.agent.calls, 1)

    def test_second_history_failure_rolls_back_first_message(self):
        self.db.fail_write = 2
        response = self.api()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.db.rows, {'history':[], 'trace':[]})
        self.assertEqual(self.db.attempts, 2)
        self.assertEqual(self.agent.calls, 1)

    def test_trace_failure_rolls_back_both_history_rows(self):
        self.db.fail_write = 3
        response = self.api()
        self.assertEqual(response['code'], 'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(self.db.rows, {'history':[], 'trace':[]})
        self.assertEqual(self.db.attempts, 3)
        self.assertEqual(self.agent.calls, 1)

    def test_rollback_preserves_preexisting_committed_records(self):
        self.db.rows={'history':[{'role':'user','message':'prior'}], 'trace':[{'pk':42}]}
        before=copy.deepcopy(self.db.rows)
        self.db.fail_write=3
        self.api()
        self.assertEqual(self.db.rows,before)

    def test_agent_and_external_formatting_precede_atomic(self):
        self.api()
        self.assertEqual(self.db.events[0], ('agent',0))
        self.assertEqual(sum(e[0]=='begin' for e in self.db.events),1)
        self.assertEqual(self.db.depth,0)

    def test_persistence_error_preserves_original_chain_no_retry(self):
        self.db.fail_write=2
        with patch.dict(sys.modules,self.modules):
            with self.assertRaises(contracts.AgentPersistenceError) as caught:
                self.new_service().process_chat('synthetic-private-prompt')
        self.assertIsInstance(caught.exception.__cause__,FakeDBError)
        self.assertEqual(self.agent.calls,1)
        self.assertEqual(self.db.attempts,2)
        self.assertEqual(self.db.rows,{'history':[],'trace':[]})

    def test_commit_failure_before_commit_has_no_visible_new_rows(self):
        self.db.commit_fault='before'
        response=self.api()
        self.assertEqual(response['code'],'AGENT_PERSISTENCE_FAILED')
        self.assertFalse(response['retryable'])
        self.assertEqual(self.db.rows,{'history':[],'trace':[]})
        self.assertEqual((self.agent.calls,self.db.attempts),(1,3))

    def test_lost_commit_acknowledgement_never_promises_rollback_or_safe_retry(self):
        self.db.commit_fault='after'
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(500,'AGENT_PERSISTENCE_FAILED'))
        self.assertFalse(response['retryable'])
        self.assertEqual((len(self.db.rows['history']),len(self.db.rows['trace'])),(2,1))
        self.assertNotIn('已回滚',response['message'])
        self.assertEqual((self.agent.calls,self.db.attempts),(1,3))

    def test_cross_database_alias_mismatch_rejected_before_any_write(self):
        self.db.aliases['AgentTrace']='other'
        response=self.api()
        self.assertEqual(response['code'],'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(self.db.attempts,0)
        self.assertEqual(self.db.rows,{'history':[],'trace':[]})

    def test_explicit_manager_database_is_respected_and_mismatch_rejected(self):
        self.trace_model.objects._db='other'
        response=self.api()
        self.assertEqual(response['code'],'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(self.db.attempts,0)

    def test_outer_transaction_prevents_agent_execution(self):
        with self.db.atomic(using='default'):
            response=self.api()
            self.assertEqual(response['code'],'AGENT_EXECUTION_FAILED')
            self.assertEqual(self.agent.calls,0)
        self.assertEqual(self.db.rows,{'history':[],'trace':[]})

    def test_outer_transaction_cannot_produce_false_persistence_success(self):
        service=self.new_service()
        with patch.dict(sys.modules,self.modules),self.db.atomic(using='default'):
            with self.assertRaises(contracts.AgentPersistenceError):service._persist_result('x',self.result)
            self.assertEqual(self.db.attempts,0)

    def test_empty_input_400_preserves_greeting_fields(self):
        self.request.POST={'msg':'  '}
        response=self.api()
        self.assertEqual(response.status_code,400)
        self.assertEqual(response['code'],'INVALID_REQUEST')
        self.assertIn('您好',response['response'])
        self.assertIsNone(response['react_trace'])
        self.assert_uuid(response['request_id'])
        self.provider.assert_not_called();self.sanitize.assert_not_called()

    def test_non_string_parameters_are_400_not_internal_errors(self):
        for raw in (None,7,['not text']):
            self.request.POST={'msg':raw}
            response=self.api()
            self.assertEqual((response.status_code,response['code']),(400,'INVALID_REQUEST'))
            self.assertFalse(response['retryable'])
        self.provider.assert_not_called()

    def test_unsafe_input_400_keeps_original_warning(self):
        self.sanitize.side_effect=lambda text:'MALICIOUS_INJECTION_DETECTED'
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(400,'UNSAFE_INPUT'))
        self.assertEqual(response['response'],'🛡️ 系统拦截：检测到不安全的输入内容。')
        self.assertIsNone(response['react_trace'])
        self.assert_uuid(response['request_id'])
        self.provider.assert_not_called()

    def test_agent_execution_failure_is_500_and_not_persisted(self):
        self.agent.error=RuntimeError('synthetic password SQL model-path')
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(500,'AGENT_EXECUTION_FAILED'))
        self.assertFalse(response['retryable'])
        self.assertEqual(self.db.attempts,0)
        self.assertEqual(self.agent.calls,1)

    def test_explicit_dependency_failure_is_503_and_not_auto_retryable(self):
        self.agent.error=contracts.AgentDependencyUnavailable('private Redis URL')
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(503,'AGENT_DEPENDENCY_UNAVAILABLE'))
        self.assertFalse(response['retryable'])
        self.assertEqual((self.agent.calls,self.db.attempts),(1,0))

    def test_typed_memory_dependency_failure_is_503(self):
        self.agent.error=p0.access.MemoryCacheUnavailable('private preference cache')
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(503,'AGENT_DEPENDENCY_UNAVAILABLE'))
        self.assertFalse(response['retryable'])

    def test_unclassified_internal_error_is_500(self):
        self.sanitize.side_effect=RuntimeError('synthetic SQL password')
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(500,'INTERNAL_ERROR'))
        self.assertEqual(self.agent.calls,0)
        self.assertFalse(response['retryable'])

    def test_generic_connection_and_timeout_not_inferred_as_503_or_504(self):
        for error in (ConnectionError('ambiguous'),TimeoutError('ambiguous')):
            self.agent.error=error
            response=self.api()
            self.assertEqual((response.status_code,response['code']),(500,'AGENT_EXECUTION_FAILED'))
            self.assertFalse(response['retryable'])

    def test_postcommit_presentation_failure_returns_safe_internal_error(self):
        self.modules['myapp.utils.xai_explainer'].translate_react_to_human=Mock(side_effect=RuntimeError('private'))
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(500,'INTERNAL_ERROR'))
        self.assertFalse(response['retryable'])
        self.assertEqual((len(self.db.rows['history']),len(self.db.rows['trace'])),(2,1))
        self.assertEqual(self.agent.calls,1)

    def test_error_payload_never_exposes_original_exception(self):
        self.agent.error=RuntimeError('synthetic-secret SQL redis://private/path model.weights')
        response=self.api()
        for secret in ('synthetic-secret','SQL','redis://','model.weights','Traceback'):
            self.assertNotIn(secret,response.content.decode())
        self.assert_uuid(response['request_id'])
        self.assertEqual(response['error'],'AGENT_EXECUTION_FAILED')

    def test_server_uuid_is_new_per_request_and_shared_with_service(self):
        first=self.api();first_service=self.instances[-1]
        second=self.api();second_service=self.instances[-1]
        self.assert_uuid(first['request_id']);self.assert_uuid(second['request_id'])
        self.assertNotEqual(first['request_id'],second['request_id'])
        self.assertEqual(first_service.request_id,first['request_id'])
        self.assertEqual(second_service.request_id,second['request_id'])

    def test_same_request_id_all_six_stages_and_trace_pk_correlation(self):
        with self.assertLogs('movie_agent.request',level='INFO') as captured:response=self.api()
        self.assertEqual([r.stage for r in captured.records],['request_received','agent_execution_started',
            'agent_execution_finished','persistence_started','persistence_succeeded','request_finished'])
        self.assertEqual({r.request_id for r in captured.records},{response['request_id']})
        self.assertTrue(all(r.elapsed_ms>=0 for r in captured.records))
        self.assertIn('trace_id=1',captured.output[-2])

    def test_failure_logs_stages_without_raw_prompt_answer_user_or_session(self):
        self.db.fail_write=3
        with self.assertLogs('movie_agent.request',level='INFO') as captured:response=self.api()
        output='\n'.join(captured.output)
        self.assertIn('persistence_failed',output);self.assertNotIn('persistence_succeeded',output)
        self.assertEqual({r.request_id for r in captured.records},{response['request_id']})
        for secret in ('synthetic-private-prompt','synthetic private answer','synthetic-private-user',
                       'synthetic-private-session','SQL/password','model-path','untrusted-client-token'):
            self.assertNotIn(secret,output)

    def test_execution_failure_is_logged_and_request_finishes(self):
        self.agent.error=RuntimeError('private')
        with self.assertLogs('movie_agent.request',level='INFO') as captured:self.api()
        self.assertEqual([r.stage for r in captured.records],['request_received','agent_execution_started',
                                                           'agent_execution_finished','request_finished'])

    def test_logging_handler_failure_does_not_change_business_commit(self):
        with patch.object(contracts.logger,'info',side_effect=RuntimeError('logger')):response=self.api()
        self.assertEqual(response.status_code,200)
        self.assertEqual((len(self.db.rows['history']),len(self.db.rows['trace'])),(2,1))

    def test_request_trace_elapsed_never_negative(self):
        values=iter([5,4])
        trace=contracts.AgentRequestTrace(clock=lambda:next(values))
        with self.assertLogs('movie_agent.request',level='INFO') as captured:trace.stage('request_received')
        self.assertEqual(captured.records[0].elapsed_ms,0)

    def test_service_constructor_and_invalid_id_remain_backward_compatible(self):
        service=self.new_service()
        self.assertEqual(service.session_id,'user_7_synthetic-private-session')
        self.assertIsNone(service.request_id)
        with patch.dict(sys.modules,self.modules):service.process_chat('synthetic')
        self.assert_uuid(service.request_id)
        trace=contracts.AgentRequestTrace('synthetic-private-session')
        self.assert_uuid(trace.request_id)

    def test_success_business_fields_match_baseline(self):
        with patch.dict(sys.modules,self.modules):
            baseline=self.new_service(legacy=True).process_chat('synthetic')
            actual=self.new_service().process_chat('synthetic')
        self.assertEqual(actual,baseline)
        response=self.api()
        self.assertEqual({k:response[k] for k in baseline},baseline)
        self.assertIn('user_friendly_trace',response)
        self.assertIn('request_id',response)
        self.assertEqual(response['recommended_ids'],[9,3])

    def test_trace_business_fields_retained_and_request_id_not_stuffed_into_content(self):
        self.api()
        row=self.db.rows['trace'][0]
        self.assertEqual(row['thought'],self.result['thought'])
        self.assertEqual(row['actions'],self.result['actions'])
        self.assertEqual(row['recommended_movies'],[9,3])
        self.assertEqual(row['observations'],[{'tool':'existing','count':2}])
        self.assertNotIn('request_id',row)

    def test_normal_sse_event_sequence_and_structure_match_baseline(self):
        with patch.dict(sys.modules,self.modules):
            baseline=list(self.new_service(legacy=True).stream_chat('synthetic'))
        response,actual=self.stream()
        self.assertEqual(actual,baseline)
        self.assertEqual(response.status_code,200)
        self.assertEqual(actual[-1]['type'],'done')
        self.assert_uuid(response.headers['X-Request-ID'])
        self.assertEqual(self.instances[-1].request_id,response.headers['X-Request-ID'])

    def test_stream_trace_failure_after_chunks_emits_error_no_done_and_rolls_back(self):
        self.db.fail_write=3
        response,events=self.stream()
        self.assertEqual(response.status_code,200)  # Headers already sent; use SSE error.
        self.assertIn('chunk',[e['type'] for e in events])
        self.assertNotIn('done',[e['type'] for e in events])
        self.assertEqual(events[-1]['type'],'error')
        self.assertEqual(events[-1]['code'],'AGENT_PERSISTENCE_FAILED')
        self.assertEqual(events[-1]['request_id'],response.headers['X-Request-ID'])
        self.assertFalse(events[-1]['retryable'])
        self.assertEqual(self.db.rows,{'history':[],'trace':[]})
        self.assertEqual(self.agent.calls,1)

    def test_stream_each_write_failure_rolls_back_without_false_done(self):
        for index in (1,2):
            self.db.fail_write=self.db.attempts+index
            response,events=self.stream()
            self.assertEqual(events[-1]['code'],'AGENT_PERSISTENCE_FAILED')
            self.assertNotIn('done',[e['type'] for e in events])
            self.assertEqual(self.db.rows,{'history':[],'trace':[]})
        self.assertEqual(self.agent.calls,2)

    def test_stream_execution_failure_uses_existing_error_event(self):
        self.agent.error=RuntimeError('synthetic-secret')
        _,events=self.stream()
        self.assertEqual([e['type'] for e in events],['intent','error'])
        self.assertEqual(events[-1]['code'],'AGENT_EXECUTION_FAILED')
        self.assertNotIn('synthetic-secret',json.dumps(events))
        self.assertEqual(self.db.attempts,0)

    def test_stream_lost_commit_ack_emits_error_without_replaying_agent(self):
        self.db.commit_fault='after'
        response,events=self.stream()
        self.assertEqual(response.status_code,200)
        self.assertEqual(events[-1]['code'],'AGENT_PERSISTENCE_FAILED')
        self.assertFalse(events[-1]['retryable'])
        self.assertNotIn('done',[e['type'] for e in events])
        self.assertEqual((len(self.db.rows['history']),len(self.db.rows['trace'])),(2,1))
        self.assertEqual((self.agent.calls,self.db.attempts),(1,3))

    def test_stream_empty_and_blocked_preserve_legacy_event_bodies(self):
        self.request.POST={'msg':''}
        _,events=self.stream()
        self.assertEqual(events,[{'type':'done','content':'您好！我是智能推荐助手，请问今天想看什么类型的电影？'}])
        self.request.POST={'msg':'synthetic'}
        self.sanitize.side_effect=lambda text:'MALICIOUS_INJECTION_DETECTED'
        _,events=self.stream()
        self.assertEqual(events,[{'type':'error','content':'🛡️ 系统拦截：检测到不安全的输入内容。'}])
        self.assertEqual(self.agent.calls,0)

    def test_failed_session_does_not_run_agent_or_fake_auth_recovery(self):
        self.request.session=types.SimpleNamespace(session_key=None,create=Mock(side_effect=ConnectionError('private')))
        response=self.api()
        self.assertEqual((response.status_code,response['code']),(500,'INTERNAL_ERROR'))
        self.assertEqual(self.agent.calls,0)
        self.provider.assert_not_called()
        with self.assertRaises(ConnectionError):self.stream()
        self.assertEqual(self.agent.calls,0)

    def test_existing_feedback_partial_success_and_framework_auth_contracts_unchanged(self):
        baseline=ast.parse(subprocess.check_output(['git','show',BASELINE+':myapp/agent_views.py'],text=True,cwd=ROOT))
        current=ast.parse((ROOT/'myapp/agent_views.py').read_text())
        for name in ('recommend_feedback_view','retry_cache_invalidation','ajax_clear_chat'):
            before=next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name==name)
            after=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        for name in ('agent_api_view','agent_stream_view'):
            fn=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual([ast.unparse(d) for d in fn.decorator_list],['login_required','require_POST'])

    def test_production_bindings_not_masked_by_ast_fixture_injection(self):
        for name in ('myapp/services/agent_chat_service.py','myapp/agent_views.py','myapp/services/agent_contracts.py'):
            source=(ROOT/name).read_text()
            # __file__ is supplied by Python's module loader, not an import.
            self.assertEqual(p31.unbound_globals(source)-{'__file__'},set(),name)
            tree=ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module=='myapp.services.agent_contracts':
                    for alias in node.names:self.assertTrue(hasattr(contracts,alias.name),alias.name)
        source=(ROOT/'myapp/agent_views.py').read_text().replace('from myapp.services.agent_contracts import AgentRequestTrace, agent_error_payload','')
        self.assertIn('AgentRequestTrace',p31.unbound_globals(source))
        source=(ROOT/'myapp/services/agent_chat_service.py').read_text().replace('from django.db import transaction, router, DEFAULT_DB_ALIAS','')
        self.assertTrue({'transaction','router','DEFAULT_DB_ALIAS'}<=p31.unbound_globals(source))

    def test_frozen_algorithms_cache_models_frontend_and_metrics_unchanged(self):
        paths=['myapp/agent/movie_agent.py','myapp/agent/react_agent.py','myapp/views.py','myapp/recommender/recall.py',
               'myapp/recommender/rerank.py','myapp/services/redis_analytics.py','myapp/services/cache_mutations.py',
               'myapp/utils/cache_keys.py','myapp/utils/cache_access.py','myapp/utils/cache_metrics.py','myapp/middleware.py',
               'myapp/models.py','myapp/models_upgrade.py','movie/settings.py','myapp/urls.py',
               'templates/agent_chat.html','templates/chat.html']
        for name in paths:
            current=(ROOT/name).read_bytes()
            baseline=subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT)
            if name == 'templates/agent_chat.html':
                # P4-A.1 explicitly authorizes only sendMessage's SSE termination.
                # Keep the rest of the template, debug branch and success renderer frozen.
                before=current.decode();old=baseline.decode()
                send='function sendMessage() {'
                sse='    // SSE 流式请求（fetch + ReadableStream）'
                tail='// ═══ 快捷消息 ═══'
                success='    function onStreamDone(data, existingDiv, text) {'
                self.assertEqual(before.split(send,1)[0],old.split(send,1)[0])
                self.assertEqual(before.split(send,1)[1].split(sse,1)[0],old.split(send,1)[1].split(sse,1)[0])
                self.assertEqual(before.split(tail,1)[1],old.split(tail,1)[1])
                self.assertEqual(before.split(success,1)[1],old.split(success,1)[1])
                self.assertEqual(before.split("            case 'thought':",1)[1].split("            case 'chunk':",1)[0],
                                 old.split("            case 'thought':",1)[1].split("            case 'chunk':",1)[0])
                self.assertEqual(before.split("            case 'chunk':",1)[1].split("            case 'metadata':",1)[0],
                                 old.split("            case 'chunk':",1)[1].split("            case 'metadata':",1)[0])
            else:
                self.assertEqual(current,baseline,name)


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) or not sys.flags.no_site:
        raise RuntimeError('CPU-only -S environment required')
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):raise RuntimeError('Unsafe module imported')
    unittest.main(verbosity=2)
