"""Finite synthetic work only. No application, ORM, model or network imports."""
import threading
import time


class FakeFailure(RuntimeError):
    def __init__(self, code):
        super().__init__('synthetic failure')
        self.code = code


class FakeAgent:
    def __init__(self, probe, *, delay=.005, gate=None, fail=False):
        if not 0 <= delay <= .1:
            raise ValueError('Finite Fake delay required')
        self.probe, self.delay, self.gate, self.fail = probe, delay, gate, fail
        self.calls = 0
        self.entered, self.returned = threading.Event(), threading.Event()
        self.thread_ids = []

    def run(self):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        self.probe.record('agent_start')
        self.entered.set()
        try:
            if self.gate is not None and not self.gate.wait(.8):
                raise FakeFailure('AGENT_DEPENDENCY_UNAVAILABLE')
            time.sleep(self.delay)
            if self.fail:
                raise FakeFailure('AGENT_EXECUTION_FAILED')
            return 'synthetic answer'
        finally:
            self.returned.set()
            self.probe.record('agent_return')


class FakePersistence:
    """Staged synthetic atomic write; NOT Django transaction verification."""
    def __init__(self, probe, *, fail_at=0, gate=None, commit_ack_loss=False):
        self.probe, self.fail_at, self.gate = probe, fail_at, gate
        self.commit_ack_loss = commit_ack_loss
        self.rows, self.calls, self.thread_ids = [], 0, []
        self.entered, self.finished = threading.Event(), threading.Event()

    def persist(self):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        self.entered.set()
        self.probe.record('transaction_start')
        staged = []
        try:
            if self.gate is not None and not self.gate.wait(.8):
                raise FakeFailure('AGENT_PERSISTENCE_FAILED')
            for index, kind in enumerate(('user_history', 'assistant_history', 'trace'), 1):
                staged.append(kind)
                if self.fail_at == index:
                    raise FakeFailure('AGENT_PERSISTENCE_FAILED')
            self.rows.extend(staged)
            self.probe.record('transaction_commit')
            if self.commit_ack_loss:
                raise FakeFailure('AGENT_PERSISTENCE_FAILED')
        finally:
            self.finished.set()
            self.probe.record('transaction_return')


class FakeScenario:
    def __init__(self, probe, *, agent=None, persistence=None, pause_stage=None, gate=None):
        self.probe = probe
        self.agent = agent or FakeAgent(probe)
        self.persistence = persistence or FakePersistence(probe)
        self.pause_stage, self.gate = pause_stage, gate
        self.paused, self.closed = threading.Event(), threading.Event()
        self.thread_ids = []

    def checkpoint(self, stage, stop):
        self.probe.record(stage)
        if stage == self.pause_stage:
            self.paused.set()
            if self.gate is not None and not self.gate.wait(.8):
                raise FakeFailure('AGENT_DEPENDENCY_UNAVAILABLE')
        return not stop.is_set()

    def source(self, stop):
        self.thread_ids.append(threading.get_ident())
        try:
            if not self.checkpoint('before_agent', stop):
                return
            yield {'type': 'intent', 'content': 'synthetic'}
            answer = self.agent.run()
            if stop.is_set():
                return
            for kind in ('thought', 'action', 'observation'):
                if not self.checkpoint('trace_' + kind, stop):
                    return
                yield {'type': kind, 'content': 'synthetic'}
            for offset in range(0, len(answer), 5):
                if not self.checkpoint('chunk', stop):
                    return
                yield {'type': 'chunk', 'content': answer[offset:offset + 5]}
            if not self.checkpoint('before_persist', stop):
                return
            # Never inspect stop halfway through this synthetic atomic operation.
            self.persistence.persist()
            if not self.checkpoint('after_persist', stop):
                return
            yield {'type': 'done', 'content': 'synthetic answer', 'recommendations': []}
        finally:
            self.thread_ids.append(threading.get_ident())
            self.closed.set()
            self.probe.record('source_close')
