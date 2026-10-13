"""Serial, paced stdlib microbenchmark; no Django/Redis/model imports.

Synthetic snapshots are NOT Django ORM objects; FakeQuerySet is NOT SQL.
Pickle protocol 5 is a controlled comparison, not a verified Redis serializer.
Only pickle blobs created by this script are deserialized.
"""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = 'ccbc04d9e2e3208c906a32fba7da8dd4c7b24d25'


def load_stdlib_helper(name, relative_path):
    source = (ROOT / relative_path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [a.name.split('.')[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module.split('.')[0]]
        else:
            continue
        if any(module not in sys.stdlib_module_names for module in imports):
            raise RuntimeError('Non-stdlib benchmark helper import')
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class SyntheticMovie:
    id: int
    title: str
    summary: str
    score: float
    vote_count: int
    genres: list
    actors: list
    directors: list
    rrf_score: float
    _state: dict
    _prefetched_objects_cache: dict


class FakeQuerySet:
    def __init__(self, manager, ids):
        self.manager = manager
        self.ids = ids

    def prefetch_related(self, *relations):
        self.manager.prefetch_calls += 1
        self.manager.relations = relations
        return self

    def __iter__(self):
        # Shuffled order ensures production restoration must restore order.
        # A shallow copy simulates fresh instances; related lists are immutable
        # in this benchmark. No I/O, SQL, deepcopy or fake sleeps in hydration.
        for movie_id in reversed(self.ids):
            row = self.manager.rows.get(movie_id)
            if row is not None:
                yield SyntheticMovie(**vars(row))


class FakeManager:
    def __init__(self, rows):
        self.rows = {row.id: row for row in rows}
        self.query_calls = 0
        self.prefetch_calls = 0
        self.relations = ()

    def filter(self, *, id__in):
        self.query_calls += 1
        return FakeQuerySet(self, id__in)


class FakeCache:
    def __init__(self):
        self.data = {}
        self.calls = {'get': 0, 'set': 0, 'add': 0, 'delete': 0}

    def get(self, key, default=None):
        self.calls['get'] += 1
        return self.data.get(key, default)

    def set(self, key, value, timeout=None):
        self.calls['set'] += 1
        self.data[key] = value
        return True

    def add(self, key, value, timeout=None):
        self.calls['add'] += 1
        if key in self.data:
            return False
        self.data[key] = value
        return True

    def delete(self, key):
        self.calls['delete'] += 1
        return self.data.pop(key, None) is not None


def synthetic_result(count):
    rows = []
    for i in range(1, count + 1):
        # Fixed, synthetic text sizes; no project dataset or user data.
        genres = [{'id': i % 8, 'name': 'Genre-' + str(i % 8)}]
        actors = [{'id': i * 3 + j, 'name': 'Actor-' + str(i * 3 + j)} for j in range(3)]
        directors = [{'id': i, 'name': 'Director-' + str(i)}]
        rows.append(SyntheticMovie(
            i, 'Movie-' + str(i), ('Synthetic synopsis ' + str(i) + '. ') * 40,
            7.0 + (i % 20) / 10, 100 + i, genres, actors, directors,
            round(1.2 / (60 + i), 4), {'adding': False, 'db': 'synthetic'},
            {'genres': genres, 'actors': actors, 'directors': directors},
        ))
    profile = 'Synthetic profile. ' * 32
    return rows, profile, {'vector': count, 'content': count, 'model': count, 'profile_text': profile}


def production_functions(source=None):
    """Compile only audited pure payload/hydration definitions via AST.

    Payload construction is copied directly from hybrid_recall_recommend's
    cache write block; profile/recall/model paths are never executed.
    """
    source = source if source is not None else (ROOT / 'myapp/views.py').read_text()
    tree = ast.parse(source)
    restore = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_restore_hybrid_cache_payload')
    hybrid = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'hybrid_recall_recommend')
    block = next(n for n in hybrid.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'cache_key is not None')
    construction = block.body[:-1]  # Exclude actual cache write.
    scope = {'math': math, '_HYBRID_CACHE_SCHEMA': 'hybrid_rec_payload_v1'}
    body = ast.parse('def build_payload(result):\n    movies_qs, profile_text, source_stats = result\n').body[0]
    body.body += construction + [ast.Return(value=ast.Name(id='payload', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[restore, body], type_ignores=[]))
    # No imports, calls to models or module initialization in these definitions.
    exec(compile(module, 'isolated-hybrid-payload', 'exec'), scope)
    return scope


def restore_function(scope, manager):
    scope['Movie'] = types.SimpleNamespace(objects=manager)
    return scope['_restore_hybrid_cache_payload']


def percentiles(samples):
    ordered = sorted(samples)
    # Nearest rank, in microseconds. Small samples are explicitly documented.
    return {name: round(ordered[max(0, math.ceil(len(ordered) * q) - 1)] / 1000, 3)
            for name, q in [('p50_us', .5), ('p95_us', .95), ('p99_us', .99)]}


def measure(function, iterations, warmup, deadline):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        if time.monotonic() >= deadline:
            raise RuntimeError('Microbenchmark wall-time budget exceeded')
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
        time.sleep(.002)  # Pace samples, outside timing; no concurrency.
    return {**percentiles(samples), 'samples': len(samples), 'samples_ns': samples}


def result_snapshot(result):
    movies, profile, stats = result
    return [(vars(movie)) for movie in movies], profile, stats


def run(iterations=80, warmup=5, max_seconds=20, source=None):
    if not 20 <= iterations <= 120 or not 0 <= warmup <= 10 or not 1 <= max_seconds <= 30:
        raise ValueError('Bounded benchmark settings required')
    started = time.monotonic()
    cpu_start = time.process_time()
    deadline = started + max_seconds
    scope = production_functions(source)
    keys = load_stdlib_helper('microbench_cache_keys', 'myapp/utils/cache_keys.py')
    observations = load_stdlib_helper('microbench_cache_metrics', 'myapp/utils/cache_metrics.py')
    rows = []
    for count in (30, 100, 500):
        original = synthetic_result(count)
        manager = FakeManager(original[0])
        restore = restore_function(scope, manager)
        payload = scope['build_payload'](original)
        legacy_blob = pickle.dumps(original, protocol=5)
        light_blob = pickle.dumps(payload, protocol=5)
        hydrated = restore(pickle.loads(light_blob))
        assert result_snapshot(hydrated) == result_snapshot(original)
        assert manager.query_calls == 1 and manager.prefetch_calls == 1
        assert manager.relations == ('genres', 'actors', 'directors')
        operations = {
            'A_object_snapshot_build': lambda: (original[0], original[1], original[2]),
            'A_object_snapshot_serialize': lambda: pickle.dumps(original, protocol=5),
            'A_object_snapshot_deserialize': lambda: pickle.loads(legacy_blob),
            'B_light_payload_build': lambda: scope['build_payload'](original),
            'B_light_payload_serialize': lambda: pickle.dumps(payload, protocol=5),
            'B_light_payload_deserialize': lambda: pickle.loads(light_blob),
            'C_light_payload_restore_only': lambda: restore(payload),
            'C_light_payload_decode_restore': lambda: restore(pickle.loads(light_blob)),
        }
        timings = {name: measure(fn, iterations, warmup, deadline) for name, fn in operations.items()}
        rows.append({'movies': count, 'A_bytes': len(legacy_blob), 'B_and_C_bytes': len(light_blob),
                     'snapshot_equivalent': True, 'timings': timings})
    backend = FakeCache()
    key_builder = keys.hybrid_recommendation_cache_key
    key = key_builder(backend, 7, 30)
    generation_calls_cold = dict(backend.calls)
    backend.calls = dict.fromkeys(backend.calls, 0)
    key_builder(backend, 7, 30)
    generation_calls_hot = dict(backend.calls)
    generation_timing = measure(lambda: key_builder(backend, 7, 30), iterations, warmup, deadline)
    metrics = observations.CacheMetrics(enabled=True)
    measured = observations.ObservedBusinessCache(backend, metrics, 'hybrid')
    measured.get('synthetic-missing')
    measured.set(key, 'synthetic-value', 1800)
    measured.get(key)
    measured.get(None)
    metrics.measure('hybrid', 'generation_lookup', key_builder, backend, 7, 30)
    metrics.measure('hybrid', 'orm_rehydration', restore, payload)
    metrics.payload_bytes('hybrid', light_blob)
    # Do not serialize tuple labels or any data/key/profile in observation output.
    snapshot = metrics.snapshot()
    examples = {'/'.join(label): count for label, count in snapshot['events'].items()}
    return {
        'kind': 'CPU-only synthetic microbenchmark; NOT Redis/MySQL/production',
        'baseline_commit': BASELINE_COMMIT, 'python': platform.python_version(),
        'platform': platform.system() + ' ' + platform.machine(),
        'serializer': 'pickle protocol 5; synthetic blobs only', 'iterations': iterations,
        'warmup': warmup, 'sample_pacing_seconds': .002, 'wall_budget_seconds': max_seconds,
        'source_sha256': hashlib.sha256((source if source is not None else (ROOT / 'myapp/views.py').read_text()).encode()).hexdigest(),
        'wall_seconds': round(time.monotonic() - started, 3),
        'cpu_seconds': round(time.process_time() - cpu_start, 3),
        'sizes': rows, 'generation_fake_calls_cold': generation_calls_cold,
        'generation_fake_calls_hot': generation_calls_hot,
        'generation_fake_timing': generation_timing, 'observation_example_events': examples,
    }


def compare_candidate(iterations=80, max_seconds=20):
    """Paired comparison: eliminate only a proven redundant float(float).

    Alternate order, same live synthetic rows, no changes to production source.
    This is a candidate measurement; it does not imply production improvement.
    """
    if not 20 <= iterations <= 120 or not 1 <= max_seconds <= 30:
        raise ValueError('Bounded candidate settings required')
    source = (ROOT / 'myapp/views.py').read_text()
    old_expression = 'round(float(scores[movie.id]), 4)'
    new_expression = 'round(scores[movie.id], 4)'
    if source.count(old_expression) == 1:
        before_source = source
        after_source = source.replace(old_expression, new_expression)
    elif source.count(new_expression) == 1:
        before_source = source.replace(new_expression, old_expression)
        after_source = source
    else:
        raise RuntimeError('Candidate source anchor must match exactly once')
    start = time.monotonic()
    cpu_start = time.process_time()
    deadline = start + max_seconds
    before = production_functions(before_source)
    after = production_functions(after_source)
    cases = []
    for count in (30, 100, 500):
        result = synthetic_result(count)
        payload = before['build_payload'](result)
        funcs = {'before': restore_function(before, FakeManager(result[0])),
                 'after': restore_function(after, FakeManager(result[0]))}
        assert result_snapshot(funcs['before'](payload)) == result_snapshot(funcs['after'](payload))
        for fn in funcs.values():
            for _ in range(5):
                fn(payload)
        samples = {'before': [], 'after': []}
        paired_delta = []
        for i in range(iterations):
            if time.monotonic() >= deadline:
                raise RuntimeError('Candidate wall-time budget exceeded')
            # Three invocations amortize clock noise. Pause only outside timing.
            for name in (('before', 'after') if i % 2 == 0 else ('after', 'before')):
                t0 = time.perf_counter_ns()
                for _ in range(3):
                    funcs[name](payload)
                samples[name].append((time.perf_counter_ns() - t0) / 3)
            paired_delta.append(samples['before'][-1] - samples['after'][-1])
            time.sleep(.004)
        cases.append({'movies': count, 'before': percentiles(samples['before']),
                      'after': percentiles(samples['after']),
                      'paired_saving': percentiles(paired_delta),
                      'after_faster_pairs': sum(d > 0 for d in paired_delta),
                      'pairs': iterations, 'result_equivalent': True,
                      'before_samples_ns': samples['before'], 'after_samples_ns': samples['after'],
                      'paired_saving_ns': paired_delta})
    return {'kind': 'paired synthetic restoration candidate; no SQL/Redis',
            'candidate': old_expression + ' -> ' + new_expression,
            'iterations': iterations, 'inner_repeats': 3, 'warmup': 5,
            'pacing_seconds': .004, 'before_source_sha256': hashlib.sha256(before_source.encode()).hexdigest(),
            'after_source_sha256': hashlib.sha256(after_source.encode()).hexdigest(),
            'wall_seconds': round(time.monotonic() - start, 3),
            'cpu_seconds': round(time.process_time() - cpu_start, 3), 'cases': cases}


def require_cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise RuntimeError('Explicit CPU-only thread limits required')
    if not sys.flags.no_site:
        raise RuntimeError('Run with python3 -S')
    forbidden = {'torch', 'transformers', 'sentence_transformers', 'faiss', 'tensorflow', 'numpy',
                 'django', 'redis', 'pymysql', 'MySQLdb', 'neo4j'}
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise RuntimeError('Unsafe module already imported')
    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                raise RuntimeError('Forbidden benchmark import: ' + fullname)
    sys.meta_path.insert(0, Guard())


if __name__ == '__main__':
    require_cpu_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--max-seconds', type=int, default=20)
    parser.add_argument('--compare-candidate', action='store_true')
    args = parser.parse_args()
    data = (compare_candidate(args.iterations, args.max_seconds) if args.compare_candidate else
            run(args.iterations, args.warmup, args.max_seconds))
    print(json.dumps(data, ensure_ascii=False, indent=2))
