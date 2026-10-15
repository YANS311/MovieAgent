"""Bounded local evidence. No payloads, session IDs or raw exception messages."""
import asyncio
from collections import deque
import importlib.util
import json
from pathlib import Path
import threading
import time


class LifecycleProbe:
    def __init__(self, capacity=256):
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError('Bounded probe required')
        self._events = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, stage):
        allowed = {'producer_start', 'producer_done', 'queue_full', 'consumer_close',
                   'consumer_eof', 'async_cancel', 'event_handoff', 'agent_start', 'agent_return',
                   'transaction_start', 'transaction_commit', 'transaction_return',
                   'before_agent', 'trace_thought', 'trace_action', 'trace_observation',
                   'chunk', 'before_persist', 'after_persist', 'source_close'}
        if stage not in allowed:
            raise ValueError('Fixed stage label required')
        with self._lock:
            self._events.append((stage, time.perf_counter_ns(), threading.get_ident()))

    def snapshot(self):
        with self._lock:
            return list(self._events)


async def wait_until(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Finite synthetic checkpoint not reached')
        await asyncio.sleep(.001)


def load_peer(name):
    spec = importlib.util.spec_from_file_location('p4b1_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def evidence():
    bridge, fake = load_peer('async_bridge_poc'), load_peer('fake_agent')
    output = {'kind': 'stdlib synthetic lifecycle evidence; not ASGI or database performance',
              'scenarios': [], 'queue_capacity': 2, 'max_workers': 1}
    for mode in ('normal', 'cancel_agent', 'cancel_transaction', 'full_queue'):
        probe, gate = LifecycleProbe(), threading.Event()
        agent = fake.FakeAgent(probe, gate=gate if mode == 'cancel_agent' else None)
        db = fake.FakePersistence(probe, gate=gate if mode == 'cancel_transaction' else None)
        scenario = fake.FakeScenario(probe, agent=agent, persistence=db)
        pool = bridge.BridgePool()
        stream = pool.open(scenario.source, probe=probe)
        initial_tasks = set(asyncio.all_tasks())
        started = time.perf_counter_ns()
        first = await anext(stream)
        first_ns = time.perf_counter_ns() - started
        types = [json.loads(first.decode()[6:])['type']]
        residual = False
        try:
            if mode == 'normal':
                async for blob in stream:
                    types.append(json.loads(blob.decode()[6:])['type'])
            elif mode == 'cancel_agent':
                await wait_until(agent.entered.is_set)
                residual = not (await stream.aclose(grace=0))['producer_finished']
            elif mode == 'cancel_transaction':
                async def pump():
                    async for blob in stream:
                        types.append(json.loads(blob.decode()[6:])['type'])
                task = asyncio.create_task(pump())
                try:
                    await wait_until(db.entered.is_set)
                    residual = not (await stream.aclose(grace=0))['producer_finished']
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            else:
                await wait_until(lambda: stream.full_waits > 0)
                await stream.aclose()
        finally:
            gate.set()
            shutdown = await pool.aclose()
        await asyncio.sleep(0)
        output['scenarios'].append({'mode': mode, 'first_handoff_ns': first_ns,
            'observed_event_types': types, 'residual_at_cancel': residual,
            'synthetic_rows_committed': len(db.rows), 'agent_calls': agent.calls,
            'stream': stream.snapshot(), 'pool_after_shutdown': shutdown,
            'extra_pending_tasks': len(set(asyncio.all_tasks()) - initial_tasks),
            'stage_sequence': [event[0] for event in probe.snapshot()]})
    return output


if __name__ == '__main__':
    print(json.dumps(asyncio.run(evidence()), ensure_ascii=False, indent=2))
