"""Request/error contracts; stdlib only, without Django or model initialization."""
import logging
import threading
import time
import uuid

logger = logging.getLogger('movie_agent.request')
sse_logger = logging.getLogger('movie_agent.sse')


class AgentExecutionError(RuntimeError):
    """Agent execution failed; automatic replay is not safe by default."""


class AgentDependencyUnavailable(RuntimeError):
    """Explicitly classified dependency failure, never inferred from error text."""


class AgentPersistenceError(RuntimeError):
    """Persistence not confirmed; commit acknowledgement can be uncertain."""


class AgentRequestTrace:
    def __init__(self, request_id=None, clock=None):
        try:
            self.request_id = str(uuid.UUID(request_id)) if request_id else str(uuid.uuid4())
        except (ValueError, TypeError, AttributeError):
            self.request_id = str(uuid.uuid4())
        self.clock = clock or time.perf_counter
        self.started = self.clock()

    def sse_lifecycle(self):
        return SSERequestLifecycle(self)

    def stage(self, name, started=None, outcome='success', trace_id=None):
        """Fixed, bounded fields only; broken log handlers cannot fail a request."""
        try:
            if name not in ('request_received', 'agent_execution_started', 'agent_execution_finished',
                            'persistence_started', 'persistence_succeeded', 'persistence_failed',
                            'request_finished'):
                return
            elapsed_ms = max(0.0, (self.clock() - (self.started if started is None else started)) * 1000)
            logger.info('agent_request stage=%s request_id=%s elapsed_ms=%.3f outcome=%s trace_id=%s',
                        name, self.request_id, elapsed_ms, outcome, trace_id,
                        extra={'request_id': self.request_id, 'stage': name, 'elapsed_ms': elapsed_ms})
        except Exception:
            pass


class SSERequestLifecycle:
    """Local synchronous stream state, NOT an ASGI disconnect detector.

    A cancellation request is cooperative: it cannot interrupt Agent/DB calls.
    Commit confirmation, iterator handoff and client receipt are separate facts.
    Only the owning stream is closed; no shared clients/models are managed here.
    """
    _PHASES = ('RECEIVED', 'VALIDATED', 'RUNNING', 'REPLAYING', 'PERSISTING')
    _TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'INCOMPLETE')

    def __init__(self, trace):
        self.trace = trace
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self.state = 'RECEIVED'
        self.cancel_source = None
        self.agent_completed = False
        self.persistence_started = False
        self.persistence_confirmed = False
        self.done_handed_off = False
        self.error_handed_off = False
        self.iteration_exhausted = False
        self.closed = False
        self.cleanup_errors = 0
        self.failure_code = None
        self._log()

    def snapshot(self):
        with self._lock:
            return {name: getattr(self, name) for name in (
                'state', 'cancel_source', 'agent_completed', 'persistence_started',
                'persistence_confirmed', 'done_handed_off', 'error_handed_off',
                'iteration_exhausted', 'closed', 'cleanup_errors', 'failure_code'
            )} | {'cancel_requested': self._cancelled.is_set(),
                 'client_received_done': None, 'http_disconnect_observed': None}

    def _log(self):
        try:
            sse_logger.info('agent_sse request_id=%s facts=%s', self.trace.request_id, self.snapshot())
        except Exception:
            pass  # Observability must never replace a business/close exception.

    def checkpoint(self, phase=None):
        changed = False
        with self._lock:
            if self._cancelled.is_set() or self.closed:
                return False
            if phase is not None:
                if phase not in self._PHASES or self.state in self._TERMINAL:
                    raise RuntimeError('Invalid SSE phase transition')
                if self._PHASES.index(phase) < self._PHASES.index(self.state):
                    raise RuntimeError('SSE phase cannot move backwards')
                changed = self.state != phase
                self.state = phase
                if phase == 'PERSISTING':
                    self.persistence_started = True
        if changed:
            self._log()
        return True

    def cancel(self, source='local_cancel'):
        with self._lock:
            if self.closed or self.state == 'COMPLETED' or self._cancelled.is_set():
                return
            self._cancelled.set()
            self.cancel_source = source if source in (
                'local_cancel', 'iterator_close', 'generator_exit') else 'local_cancel'
            if self.state != 'FAILED':
                self.state = 'CANCELLED'
        self._log()

    def agent_succeeded(self):
        with self._lock:
            self.agent_completed = True

    def persistence_succeeded(self):
        # Even cancellation during the synchronous transaction cannot undo it.
        with self._lock:
            self.persistence_confirmed = True
        self._log()

    def fail(self, code):
        with self._lock:
            safe_code = code if code in (
                'AGENT_EXECUTION_FAILED', 'AGENT_DEPENDENCY_UNAVAILABLE',
                'AGENT_PERSISTENCE_FAILED', 'INTERNAL_ERROR', 'SSE_ABORTED',
                'SSE_CLEANUP_FAILED', 'SSE_ERROR') else 'INTERNAL_ERROR'
            if self.failure_code is None:
                self.failure_code = safe_code
            if not self._cancelled.is_set():
                self.state = 'FAILED'
        self._log()

    def handoff(self, kind):
        """Encoded event is leaving our iterator, not acknowledged by a client."""
        with self._lock:
            if self._cancelled.is_set() or self.closed or self.error_handed_off:
                return False
            if self.done_handed_off and kind != 'error':
                raise RuntimeError('Unexpected SSE event after done')
            if kind == 'done':
                self.done_handed_off = True
            elif kind == 'error':
                self.error_handed_off = True
                self.state = 'FAILED'
                if self.failure_code is None:
                    self.failure_code = 'SSE_ERROR'
        if kind in ('done', 'error'):
            self._log()
        return True

    def exhausted(self):
        with self._lock:
            self.iteration_exhausted = True
            if self._cancelled.is_set() or self.state == 'FAILED':
                return
            self.state = 'COMPLETED' if self.done_handed_off else 'INCOMPLETE'

    def close_owned(self, iterator):
        """Best effort close of an owned iterator; never close shared resources."""
        try:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        except Exception:
            with self._lock:
                self.cleanup_errors += 1
            self.fail('SSE_CLEANUP_FAILED')

    def finish(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self.state not in self._TERMINAL:
                self.state = 'CANCELLED' if self._cancelled.is_set() else 'INCOMPLETE'
            outcome = 'sse_iterator_' + self.state.lower()
        self._log()
        try:
            self.trace.stage('request_finished', outcome=outcome)
        except Exception:
            pass

    def wrap(self, factory, encoder):
        return _OwnedSSEIterator(self, factory, encoder)


class _OwnedSSEIterator:
    """Lazy sync iterator with an idempotent closer registered by Django.

    close() while next() is active only signals cancellation. The iterator is
    disposed by that same next() call after control returns, never concurrently
    closed while a synchronous generator is executing. No worker is spawned.
    """
    def __init__(self, lifecycle, factory, encoder):
        self.lifecycle = lifecycle
        self._factory = factory
        self._encoder = encoder
        self._iterator = None
        self._consume_lock = threading.Lock()
        self._disposed = False

    def __iter__(self):
        return self

    def _dispose(self):
        if self._disposed:
            return
        self._disposed = True
        owned = self._iterator
        self._iterator = self._factory = self._encoder = None
        try:
            self.lifecycle.close_owned(owned)
        finally:
            self.lifecycle.finish()

    def close(self):
        if self._disposed:
            return
        self.lifecycle.cancel('iterator_close')
        if self._consume_lock.acquire(blocking=False):
            try:
                self._dispose()
            finally:
                self._consume_lock.release()

    def __next__(self):
        if not self._consume_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent SSE iteration is unsupported')
        try:
            if self._disposed or not self.lifecycle.checkpoint():
                raise StopIteration
            if self._iterator is None:
                self._iterator = iter(self._factory())
                self._factory = None
            if not self.lifecycle.checkpoint():
                raise StopIteration
            try:
                event = next(self._iterator)
            except StopIteration:
                self.lifecycle.exhausted()
                raise
            if not self.lifecycle.checkpoint():
                raise StopIteration
            if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                raise RuntimeError('Invalid SSE event')
            encoded = self._encoder(event)
            if not self.lifecycle.handoff(event['type']):
                raise StopIteration
            if event['type'] == 'error':
                self._dispose()
            return encoded
        except StopIteration:
            self._dispose()
            raise
        except GeneratorExit:
            self.lifecycle.cancel('generator_exit')
            self._dispose()
            raise
        except Exception as exc:
            payload, _ = agent_error_payload(exc, self.lifecycle.trace.request_id)
            self.lifecycle.fail(payload['code'])
            try:
                if not self.lifecycle.checkpoint() or self.lifecycle.error_handed_off:
                    raise StopIteration
                event = {'type': 'error', 'content': payload['message'], 'code': payload['code'],
                         'request_id': self.lifecycle.trace.request_id, 'retryable': False}
                encoded = self._encoder(event)
                if not self.lifecycle.handoff('error'):
                    raise StopIteration
                return encoded
            finally:
                self._dispose()
        except BaseException:
            self.lifecycle.fail('SSE_ABORTED')
            self._dispose()
            raise
        finally:
            # Includes cancellation signalled from within a blocking fake/call.
            # No attempt to close a running generator on another thread.
            try:
                if self.lifecycle.snapshot()['cancel_requested']:
                    self._dispose()
            finally:
                self._consume_lock.release()


def agent_error_payload(error, request_id):
    """No raw exception text, SQL, URL, traceback, prompt or retry promise."""
    if isinstance(error, AgentPersistenceError):
        status, code = 500, 'AGENT_PERSISTENCE_FAILED'
        message = '回复已生成，但保存未确认完成，请勿自动重试。'
    elif isinstance(error, AgentDependencyUnavailable):
        status, code = 503, 'AGENT_DEPENDENCY_UNAVAILABLE'
        message = '智能助手依赖暂时不可用，请稍后再试。'
    elif isinstance(error, AgentExecutionError):
        status, code = 500, 'AGENT_EXECUTION_FAILED'
        message = '智能助手暂时不可用，请稍后再试。'
    else:
        status, code = 500, 'INTERNAL_ERROR'
        message = '智能助手暂时遇到了问题，请稍后重试。'
    return {'code': code, 'message': message, 'request_id': request_id, 'retryable': False,
            'response': message, 'react_trace': None, 'error': code}, status
