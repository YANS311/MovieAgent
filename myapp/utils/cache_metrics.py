"""Opt-in, bounded, process-local business-cache probes. Never wrap Sessions.

No logs, export threads, payload serialization, user/key labels or Django imports.
Pass enabled=True explicitly in a controlled diagnostic call; default is off.
"""

from collections import Counter, defaultdict, deque
from time import perf_counter_ns

FAMILIES = frozenset({'hybrid', 'intent', 'kg', 'multi_recall', 'query_rewrite',
                      'explain', 'agent_recall', 'generation', 'other'})
EVENTS = frozenset({'hit', 'miss', 'error', 'bypass'})
TIMINGS = frozenset({'read', 'write', 'generation_lookup', 'orm_rehydration'})


class CacheMetrics:
    def __init__(self, enabled=False, max_samples=256, clock=perf_counter_ns):
        self.enabled = enabled
        self.max_samples = min(1024, max(1, int(max_samples)))
        self.clock = clock
        self.events = Counter()
        self.samples = defaultdict(lambda: deque(maxlen=self.max_samples))

    def _family(self, family):
        return family if family in FAMILIES else 'other'

    def event(self, family, event):
        if not self.enabled:
            return
        try:
            if event in EVENTS:
                self.events[(self._family(family), event)] += 1
        except Exception:
            pass  # Instrumentation must not alter business behavior.

    def _start(self):
        try:
            return self.clock() if self.enabled else None
        except Exception:
            return None

    def _finish(self, family, metric, started):
        if started is None:
            return
        try:
            if metric in TIMINGS:
                self.samples[(self._family(family), metric)].append(max(0, self.clock() - started))
        except Exception:
            pass

    def measure(self, family, metric, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        try:
            started = self._start()
        except Exception:
            started = None
        try:
            return function(*args, **kwargs)
        finally:
            try:
                self._finish(family, metric, started)
            except Exception:
                pass

    def payload_bytes(self, family, serialized_bytes):
        """Measure an already-serialized blob; never serialize or retain content."""
        if not self.enabled:
            return
        try:
            if isinstance(serialized_bytes, (bytes, bytearray, memoryview)):
                size = serialized_bytes.nbytes if isinstance(serialized_bytes, memoryview) else len(serialized_bytes)
                self.samples[(self._family(family), 'serialization_bytes')].append(size)
        except Exception:
            pass

    def snapshot(self):
        try:
            return {'events': dict(self.events),
                    'samples': {key: list(values) for key, values in self.samples.items()}}
        except Exception:
            return {'events': {}, 'samples': {}}


class ObservedBusinessCache:
    """Explicit adapter passed to existing fail-open business helpers only.

    Exceptions and return values are preserved; cache_access decides fallback.
    No __getattr__ forwarding, so this is deliberately not a Session backend.
    None keys bypass the backend. `add`/`get` allow generation builder probes;
    use metrics.measure(..., 'generation_lookup', key_builder, adapter, ...) to
    measure the complete generation path, including add/get on cold tokens.
    """
    def __init__(self, backend, metrics, family='other'):
        self.backend = backend
        self.metrics = metrics
        self.family = family if family in FAMILIES else 'other'

    def _record(self, event):
        try:
            self.metrics.event(self.family, event)
        except Exception:
            pass

    def _call(self, metric, function, *args):
        # A broken observer must not cause duplicate backend calls.
        try:
            started = self.metrics._start()
        except Exception:
            started = None
        try:
            return function(*args)
        finally:
            try:
                self.metrics._finish(self.family, metric, started)
            except Exception:
                pass

    def get(self, key, default=None):
        if key is None:
            self._record('bypass')
            return default
        try:
            value = self._call('read', self.backend.get, key, default)
        except Exception:
            self._record('error')
            raise
        # Business helpers treat None as a miss. For an explicit default that
        # equals a cached value, backend presence is unknowable without an
        # extra query; report the default identity as a semantic miss.
        self._record('miss' if value is default or value is None else 'hit')
        return value

    def set(self, key, value, timeout=None):
        if key is None:
            self._record('bypass')
            return False
        try:
            result = self._call('write', self.backend.set, key, value, timeout)
        except Exception:
            self._record('error')
            raise
        if result is False:
            self._record('error')
        return result

    def add(self, key, value, timeout=None):
        try:
            return self._call('write', self.backend.add, key, value, timeout)
        except Exception:
            self._record('error')
            raise

    def delete(self, key):
        if key is None:
            self._record('bypass')
            return False
        try:
            return self.backend.delete(key)
        except Exception:
            self._record('error')
            raise
