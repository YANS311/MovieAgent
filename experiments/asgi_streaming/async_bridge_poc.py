"""Standalone stdlib PoC; never imported by MovieAgent production code.

One fixed executor, fail-fast admission, bounded encoded-byte queues. Only use
audited finite Fake producers. A cancelled coroutine cannot kill sync work.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import json
import queue
import threading
import time
import uuid


class CapacityExceeded(RuntimeError):
    pass


class PollDeadlineExceeded(RuntimeError):
    pass


class BridgePool:
    """A lease lasts until BOTH producer exit and consumer close/EOF.

    Thus even finished producers with unread queues occupy admission slots.
    At most max_workers jobs/streams can be admitted; executor's internal work
    queue cannot accumulate arbitrary per-request submissions.
    """
    def __init__(self, max_workers=1):
        if type(max_workers) is not int or not 1 <= max_workers <= 2:
            raise ValueError('PoC max_workers must be 1..2')
        self.max_workers = max_workers
        self._prefix = 'movieagent-asgi-poc-' + uuid.uuid4().hex[:8]
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=self._prefix)
        self._lock = threading.RLock()
        self._streams = set()
        self._loop = None
        self._accepting = True
        self._joined = False
        self.max_admitted = 0
        self.unexpected_aborts = 0

    def open(self, factory, *, capacity=2, max_seconds=2, max_event_bytes=4096, probe=None):
        if (type(capacity) is not int or not 1 <= capacity <= 8
                or type(max_seconds) not in (int, float) or not .05 <= max_seconds <= 5
                or type(max_event_bytes) is not int or not 512 <= max_event_bytes <= 65536
                or not callable(factory)):
            raise ValueError('Finite PoC queue/event/time limits required')
        loop = asyncio.get_running_loop()
        with self._lock:
            if not self._accepting:
                raise RuntimeError('PoC pool is closing')
            if self._loop is not None and self._loop is not loop:
                raise RuntimeError('One event loop per PoC pool required')
            if len(self._streams) >= self.max_workers:
                raise CapacityExceeded('PoC admission capacity exhausted')
            self._loop = loop
            stream = BridgeStream(self, loop, factory, capacity, max_seconds, max_event_bytes, probe)
            self._streams.add(stream)
            self.max_admitted = max(self.max_admitted, len(self._streams))
            return stream

    def _start(self, stream):
        with self._lock:
            if stream._started or stream._closed:
                return
            if not self._accepting or stream not in self._streams:
                raise RuntimeError('PoC stream no longer admitted')
            stream._started = True
            stream._deadline = time.monotonic() + stream.max_seconds
            try:
                stream._future = self._executor.submit(stream._context.run, stream._produce)
            except BaseException:
                stream._producer_done.set()
                stream._completed.set()
                stream.close()
                raise
            stream._future.add_done_callback(lambda future: self._finished(stream, future))

    def _finished(self, stream, future):
        # Observe exceptional futures without logging raw exception contents.
        if future.exception() is not None:
            with self._lock:
                self.unexpected_aborts += 1
        self._release_if_done(stream)

    def _release_if_done(self, stream):
        with self._lock:
            future_done = stream._future is None or stream._future.done()
            if stream._consumer_finished and future_done:
                self._streams.discard(stream)

    def snapshot(self):
        with self._lock:
            return {'admitted': len(self._streams), 'max_admitted': self.max_admitted,
                    'running_jobs': sum(s._future is not None and not s._future.done() for s in self._streams),
                    'workers_alive': sum(t.name.startswith(self._prefix) and t.is_alive()
                                         for t in threading.enumerate()),
                    'max_workers': self.max_workers, 'unexpected_aborts': self.unexpected_aborts,
                    'joined': self._joined}

    async def aclose(self, grace=1):
        if type(grace) not in (int, float) or not 0 <= grace <= 2:
            raise ValueError('PoC close grace must be 0..2 seconds')
        with self._lock:
            self._accepting = False
            streams = tuple(self._streams)
        for stream in streams:
            stream.close()
        deadline = time.monotonic() + grace
        while self.snapshot()['running_jobs'] and time.monotonic() < deadline:
            await asyncio.sleep(.001)
        drained = self.snapshot()['running_jobs'] == 0
        # Join only after observed work completion; never block the loop on an
        # Agent/transaction. Residual work is reported, not force-terminated.
        self._executor.shutdown(wait=drained, cancel_futures=False)
        self._joined = drained
        return self.snapshot() | {'drained': drained}


class BridgeStream:
    def __init__(self, pool, loop, factory, capacity, max_seconds, max_event_bytes, probe):
        self.pool, self.loop = pool, loop
        self.capacity, self.max_seconds, self.max_event_bytes = capacity, max_seconds, max_event_bytes
        self._factory, self._context, self.probe = factory, contextvars.copy_context(), probe
        self._queue = queue.Queue(maxsize=capacity)
        self._wake, self._completed = asyncio.Event(), asyncio.Event()
        self._stop, self._producer_done = threading.Event(), threading.Event()
        self._lock = threading.RLock()
        self._future, self._terminal = None, None
        self._started = self._closed = self._consumer_finished = self._pulling = False
        self._terminal_delivered = False
        self._deadline = None
        self.max_queue_observed = self.full_waits = self.cleanup_errors = 0
        self.request_id = str(uuid.uuid4())

    def _record(self, name):
        try:
            if self.probe is not None:
                self.probe.record(name)
        except Exception:
            pass

    def _notify(self, complete=False):
        def on_loop():
            self._wake.set()
            if complete:
                self._completed.set()
        try:
            self.loop.call_soon_threadsafe(on_loop)
        except RuntimeError:
            self._stop.set()  # Loop already closed: worker may still be blocked.

    def _encode(self, event):
        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
            raise ValueError('Invalid synthetic event')
        blob = ('data: ' + json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n\n').encode()
        if len(blob) > self.max_event_bytes:
            raise ValueError('Synthetic event exceeds configured byte limit')
        return blob

    def _error(self, exc=None):
        code = getattr(exc, 'code', 'INTERNAL_ERROR')
        if code not in {'AGENT_EXECUTION_FAILED', 'AGENT_PERSISTENCE_FAILED', 'AGENT_DEPENDENCY_UNAVAILABLE'}:
            code = 'INTERNAL_ERROR'
        return self._encode({'type': 'error', 'content': '模拟响应未成功完成。', 'code': code,
                             'request_id': self.request_id, 'retryable': False})

    def _offer(self, blob):
        while not self._stop.is_set():
            if time.monotonic() >= self._deadline:
                raise PollDeadlineExceeded('Cooperative producer deadline exceeded')
            try:
                self._queue.put(blob, timeout=.01)
                with self._lock:
                    self.max_queue_observed = max(self.max_queue_observed, self._queue.qsize())
                self._notify()
                return not self._stop.is_set()
            except queue.Full:
                with self._lock:
                    self.full_waits += 1
                self._record('queue_full')
        return False

    def _produce(self):
        producer, terminal = None, None
        self._record('producer_start')
        try:
            if not self._stop.is_set():
                with self._lock:
                    factory = self._factory
                if factory is None:
                    return
                producer = iter(factory(self._stop))
                while not self._stop.is_set():
                    if time.monotonic() >= self._deadline:
                        raise PollDeadlineExceeded('Cooperative producer deadline exceeded')
                    try:
                        event = next(producer)
                    except StopIteration:
                        break
                    if self._stop.is_set():
                        break
                    if not isinstance(event, dict):
                        raise ValueError('Invalid synthetic event')
                    if event.get('type') == 'error':
                        terminal = self._error()
                        break  # Never pull a later done after an error.
                    if terminal is not None:
                        raise ValueError('Unexpected event following done')
                    blob = self._encode(event)
                    if event['type'] == 'done':
                        terminal = blob  # Out of band; validate exhaustion/cleanup first.
                    elif not self._offer(blob):
                        break
        except Exception as exc:
            terminal = self._error(exc)
        except BaseException as exc:
            terminal = self._error(exc)
            raise  # Future records the abort; never claim successful producer exit.
        finally:
            try:
                close = getattr(producer, 'close', None)
                if close is not None:
                    close()  # Runs on the same producer thread, never on loop.
            except BaseException as exc:
                with self._lock:
                    self.cleanup_errors += 1
                terminal = self._error(exc)
                if not isinstance(exc, Exception):
                    raise
            finally:
                if self._stop.is_set():
                    terminal = None
                    self._drain()
                with self._lock:
                    self._terminal = terminal
                    self._factory = self._context = None
                self._producer_done.set()
                self._record('producer_done')
                self._notify(complete=True)

    def _drain(self):
        for _ in range(self.capacity):
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        """Nonblocking stop signal suitable for an iterable's close callback.

        No thread join, shared resource close or rollback claim. Producer can
        still be running after this method returns.
        """
        with self._lock:
            if self._consumer_finished:
                return
            self._closed = self._consumer_finished = True
            self._stop.set()
            self._terminal = None
            never_started = not self._started
            if never_started:
                self._factory = None
        self._drain()
        if never_started:
            self._context = None
            self._producer_done.set()
            self._completed.set()
        self._record('consumer_close')
        self._notify()
        self.pool._release_if_done(self)

    async def aclose(self, grace=.05):
        if type(grace) not in (int, float) or not 0 <= grace <= 2:
            raise ValueError('PoC close grace must be 0..2 seconds')
        self.close()
        if not self._producer_done.is_set() and grace:
            try:
                await asyncio.wait_for(self._completed.wait(), grace)
            except asyncio.TimeoutError:
                pass  # Explicit residual-running status in snapshot below.
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return {'capacity': self.capacity, 'queue_size': self._queue.qsize(),
                    'max_queue_observed': self.max_queue_observed, 'full_waits': self.full_waits,
                    'queued_bytes_upper_bound': self.capacity * self.max_event_bytes,
                    'producer_finished': self._producer_done.is_set(),
                    'consumer_finished': self._consumer_finished, 'stop_requested': self._stop.is_set(),
                    'terminal_delivered': self._terminal_delivered, 'cleanup_errors': self.cleanup_errors,
                    'client_received_done': None}

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pulling:
            raise RuntimeError('Concurrent async stream pulls unsupported')
        self._pulling = True
        try:
            if self._consumer_finished:
                raise StopAsyncIteration
            self.pool._start(self)
            while True:
                if self._closed:
                    raise StopAsyncIteration
                try:
                    blob = self._queue.get_nowait()
                except queue.Empty:
                    if self._producer_done.is_set():
                        with self._lock:
                            terminal = self._terminal
                            if terminal is not None and not self._terminal_delivered:
                                self._terminal_delivered = True
                            else:
                                terminal = None
                        if terminal is not None:
                            self._record('event_handoff')
                            return terminal
                        self._consumer_finished = True
                        self._record('consumer_eof')
                        self.pool._release_if_done(self)
                        raise StopAsyncIteration
                    self._wake.clear()
                    # Recheck after clear: never lose a producer wake-up.
                    if self._closed or self._producer_done.is_set() or not self._queue.empty():
                        continue
                    remaining = self._deadline - time.monotonic()
                    if remaining <= 0:
                        raise PollDeadlineExceeded('Consumer wait budget exceeded')
                    try:
                        # Bounded timer fallback: this sandbox cannot reliably wake
                        # its selector from worker threads. Still use thread-safe
                        # notifications; never busy-spin or block the loop.
                        await asyncio.wait_for(self._wake.wait(), min(.005, remaining))
                    except asyncio.TimeoutError:
                        continue  # Overall deadline is checked on the next pass.
                else:
                    if self._closed:
                        raise StopAsyncIteration
                    self._record('event_handoff')
                    return blob
        except asyncio.CancelledError:
            self._record('async_cancel')
            self.close()
            raise  # Cancels waiting coroutine, not the running sync function.
        except Exception:
            self.close()
            raise
        finally:
            self._pulling = False


async def encoded_events(stream):
    """Async-generator finally contract for an explicitly owned local consumer.

    Not a replacement for Django response close registration before first pull.
    Use contextlib.aclosing at the caller when cancellation occurs during send.
    """
    try:
        async for blob in stream:
            yield blob
    finally:
        await stream.aclose()
