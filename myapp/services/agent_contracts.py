"""Request/error contracts; stdlib only, without Django or model initialization."""
import logging
import time
import uuid

logger = logging.getLogger('movie_agent.request')


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
