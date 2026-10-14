"""Opt-in feedback analytics. No Django/model/redis imports at module load.

Dedicated standalone Redis 7+ only; never a Session backend or recommendation
input. All keys are explicit, independent of django-redis KEY_PREFIX.
"""

from datetime import datetime, timezone
import hashlib
import logging
import math
import re
import time
from urllib.parse import urlsplit

PREFIX = 'movieagent:analytics:v1'
STREAM_KEY = PREFIX + ':feedback:events'
GROUP = 'feedback-stats-v1'
FIELDS = ('click', 'like', 'dislike', 'collect', 'skip', 'share')
DEFAULT_WEIGHTS = dict(zip(FIELDS, (1, 3, 0, 5, 0, 2)))
STATS_TTL = 8 * 86400
DEDUPE_TTL = 9 * 86400
logger = logging.getLogger('movie_agent.analytics')


def _failure(operation, exc, event_id=None):
    try:
        allowed = ('ANALYTICS_BACKLOG_FULL', 'ANALYTICS_UNCERTAIN_UPDATE', 'ANALYTICS_ID_CONFLICT',
                   'ANALYTICS_WRONGTYPE', 'ANALYTICS_BAD_COUNT', 'ANALYTICS_BAD_SCORE', 'ANALYTICS_ACL_DENIED')
        reason = next((code for code in allowed if str(exc).startswith(code)), 'backend_or_validation')
        logger.error('Analytics failed operation=%s event_id=%s reason=%s error_type=%s',
                     operation, event_id, reason, type(exc).__name__)
    except Exception:
        pass  # A broken log handler must not block cache invalidation either.

# Admission is atomic. No MAXLEN/XTRIM/stream TTL: pending data is not dropped.
APPEND_LUA = """
local kind = redis.call('TYPE', KEYS[1]).ok
if kind ~= 'none' and kind ~= 'stream' then return redis.error_reply('ANALYTICS_WRONGTYPE') end
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[1]) then
  return redis.error_reply('ANALYTICS_BACKLOG_FULL')
end
return redis.call('XADD', KEYS[1], '*', 'event_id', ARGV[2], 'movie_id', ARGV[3],
 'feedback_type', ARGV[4], 'event_timestamp', ARGV[5], 'schema_version', '1')
"""

UPDATE_LUA = """
-- One atomic operation: validate, reserve ID, increment both structures, finish.
local kinds = {'hash', 'zset', 'string'}
for i = 1, 3 do
 local kind = redis.call('TYPE', KEYS[i]).ok
 if kind ~= 'none' and kind ~= kinds[i] then return redis.error_reply('ANALYTICS_WRONGTYPE') end
end
local marker = redis.call('GET', KEYS[3])
if marker then
 if marker == 'done:' .. ARGV[6] then return 0 end
 if marker == 'pending:' .. ARGV[6] then return redis.error_reply('ANALYTICS_UNCERTAIN_UPDATE') end
 return redis.error_reply('ANALYTICS_ID_CONFLICT')
end
local old_count = redis.call('HGET', KEYS[1], ARGV[1])
if old_count and (not string.match(old_count, '^%d+$') or not tonumber(old_count)
 or tonumber(old_count) > 1000000000000) then return redis.error_reply('ANALYTICS_BAD_COUNT') end
local old_score = tonumber(redis.call('ZSCORE', KEYS[2], ARGV[3]) or '0')
if not old_score or old_score ~= old_score or math.abs(old_score) > 1000000000000 then
 return redis.error_reply('ANALYTICS_BAD_SCORE')
end
-- Redis 7 ACL preflight avoids deterministic partial writes from denied commands.
if not redis.acl_check_cmd('SET', KEYS[3], 'pending:' .. ARGV[6], 'EX', ARGV[5])
 or not redis.acl_check_cmd('HINCRBY', KEYS[1], ARGV[1], 1)
 or not redis.acl_check_cmd('ZINCRBY', KEYS[2], ARGV[2], ARGV[3])
 or not redis.acl_check_cmd('EXPIRE', KEYS[1], ARGV[4])
 or not redis.acl_check_cmd('EXPIRE', KEYS[2], ARGV[4]) then
 return redis.error_reply('ANALYTICS_ACL_DENIED')
end
-- Unexpected Redis runtime errors are NOT rolled back by Lua. A pending marker
-- makes such an uncertain update fail closed on replay instead of double count.
redis.call('SET', KEYS[3], 'pending:' .. ARGV[6], 'EX', ARGV[5])
redis.call('HINCRBY', KEYS[1], ARGV[1], 1)
redis.call('ZINCRBY', KEYS[2], ARGV[2], ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
redis.call('SET', KEYS[3], 'done:' .. ARGV[6], 'EX', ARGV[5])
return 1
"""

# Only remove an ACKed entry when there is exactly our one group and no pending
# reference. Checking and deletion are atomic, including a concurrent new group.
PRUNE_LUA = """
local groups = redis.call('XINFO', 'GROUPS', KEYS[1])
if #groups ~= 1 then return 0 end
local name = nil
for i = 1, #groups[1], 2 do if groups[1][i] == 'name' then name = groups[1][i+1] end end
if name ~= ARGV[1] then return 0 end
if #redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1) ~= 0 then return 0 end
return redis.call('XDEL', KEYS[1], ARGV[2])
"""


class AnalyticsError(RuntimeError):
    pass


def _identity(url):
    """Normalize DB/endpoint without exposing credentials or accepting overrides."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ('redis', 'rediss') or not parsed.hostname or parsed.query
            or parsed.fragment or not re.fullmatch(r'/\d+', parsed.path)):
        raise AnalyticsError('Explicit standalone analytics Redis URL with DB required')
    host = parsed.hostname.lower()
    if host in ('localhost', '127.0.0.1', '::1'):
        host = 'loopback'
    return host, parsed.port or 6379, int(parsed.path[1:])


def _cache_identity(url):
    # django cache URLs may omit /0. Treat such a URL as DB 0 for isolation.
    p = urlsplit(url)
    if p.scheme not in ('redis', 'rediss'):
        return None
    if p.query or p.fragment:
        raise AnalyticsError('Cannot verify cache URL isolation with URL overrides')
    return _identity(url if p.path else url + '/0')


def analytics_from_settings(settings, factory=None):
    if getattr(settings, 'REDIS_ANALYTICS_ENABLED', False) is not True:
        return None  # No URL read, redis import, connection or command.
    url = getattr(settings, 'REDIS_ANALYTICS_URL', None)
    if not isinstance(url, str) or not url:
        raise AnalyticsError('Dedicated REDIS_ANALYTICS_URL required')
    identity = _identity(url)
    for config in getattr(settings, 'CACHES', {}).values():
        locations = config.get('LOCATION', [])
        locations = [locations] if isinstance(locations, str) else locations
        for location in locations:
            if isinstance(location, str) and _cache_identity(location) == identity:
                raise AnalyticsError('Analytics cannot share a cache/Session Redis DB')
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(getattr(settings, 'REDIS_ANALYTICS_WEIGHTS', {}))
    if set(weights) != set(FIELDS) or any(type(v) not in (int, float) or not 0 <= v <= 100
                                         or not math.isfinite(v) for v in weights.values()):
        raise AnalyticsError('Invalid analytics display weights')
    capacity = getattr(settings, 'REDIS_ANALYTICS_STREAM_CAPACITY', 100000)
    if type(capacity) is not int or not 1 <= capacity <= 1000000:
        raise AnalyticsError('Invalid analytics stream capacity')
    if factory is None:
        import redis  # Lazy: existing django-redis dependency; never installed here.
        factory = redis.Redis.from_url
    client = factory(url, decode_responses=True, socket_timeout=.5, socket_connect_timeout=.5,
                     retry_on_timeout=False, max_connections=2)
    return FeedbackAnalytics(client, weights=weights, capacity=capacity)


def _positive_id(value):
    text = str(value)
    if not re.fullmatch(r'[1-9][0-9]{0,18}', text) or int(text) > 9223372036854775807:
        raise AnalyticsError('Invalid analytics record ID')
    return text


def validate_event(raw):
    allowed = {'event_id', 'movie_id', 'feedback_type', 'event_timestamp', 'schema_version'}
    raw = {k.decode() if isinstance(k, bytes) else k:
           v.decode() if isinstance(v, bytes) else v for k, v in raw.items()}
    if set(raw) != allowed or raw['schema_version'] != '1' or raw['feedback_type'] not in FIELDS:
        raise AnalyticsError('Invalid analytics event schema')
    stamp = raw['event_timestamp']
    if not isinstance(stamp, str) or len(stamp) > 40:
        raise AnalyticsError('Invalid analytics timestamp')
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        raise AnalyticsError('Analytics timestamp must include timezone')
    return {'event_id': _positive_id(raw['event_id']), 'movie_id': _positive_id(raw['movie_id']),
            'feedback_type': raw['feedback_type'],
            'event_timestamp': parsed.astimezone(timezone.utc).isoformat(timespec='microseconds'),
            'schema_version': '1'}


def event_from_feedback(feedback):
    return validate_event({'event_id': str(feedback.pk), 'movie_id': str(feedback.movie_id),
                           'feedback_type': feedback.feedback_type,
                           'event_timestamp': feedback.created_at.isoformat(), 'schema_version': '1'})


def schedule_feedback_analytics(transaction, feedback, settings):
    """Register before cache invalidation; isolated callback never propagates.

    Delivery is best effort, without an outbox. Do not retry database writes.
    """
    try:
        if getattr(settings, 'REDIS_ANALYTICS_ENABLED', False) is not True:
            return False
        event = event_from_feedback(feedback)
        def deliver():
            try:
                service = analytics_from_settings(settings)
                if service is None:
                    logger.warning('Analytics delivery outcome=disabled_after_commit')
                    return
                service.publish(event)
            except Exception as exc:
                _failure('delivery', exc, event['event_id'])
        transaction.on_commit(deliver)
        return True  # Registered, not a delivery-success receipt.
    except Exception as exc:
        _failure('registration', exc)
        return False


class FeedbackAnalytics:
    def __init__(self, client, weights=None, capacity=100000):
        self.client = client
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if (set(self.weights) != set(FIELDS) or any(type(v) not in (int, float)
                or not 0 <= v <= 100 or not math.isfinite(v) for v in self.weights.values())
                or type(capacity) is not int or not 1 <= capacity <= 1000000):
            raise AnalyticsError('Invalid analytics service configuration')
        self.capacity = capacity
        # Per-instance, single-caller polling state; not a durable Redis lease.
        self._reclaim_cursor = '0-0'
        self._next_lane = 'pending'

    def publish(self, event):
        event = validate_event(event)
        return self.client.eval(APPEND_LUA, 1, STREAM_KEY, self.capacity,
                                event['event_id'], event['movie_id'], event['feedback_type'],
                                event['event_timestamp'])

    def process(self, event):
        event = validate_event(event)
        stamp = datetime.fromisoformat(event['event_timestamp'])
        day = f'{stamp.year:04d}{stamp.month:02d}{stamp.day:02d}'
        fingerprint = hashlib.sha256('|'.join(event.values()).encode()).hexdigest()
        return self.client.eval(UPDATE_LUA, 3,
                                f"{PREFIX}:movie:{event['movie_id']}:day:{day}",
                                f'{PREFIX}:hot:{day}', f"{PREFIX}:dedupe:event:{event['event_id']}",
                                event['feedback_type'], self.weights[event['feedback_type']],
                                event['movie_id'], STATS_TTL, DEDUPE_TTL, fingerprint)

    def counts(self, movie_id, day):
        _day(day)
        data = self.client.hgetall(f'{PREFIX}:movie:{_positive_id(movie_id)}:day:{day}')
        data = {k.decode() if isinstance(k, bytes) else k: int(v) for k, v in data.items()}
        return {field: data.get(field, 0) for field in FIELDS}

    def top(self, day, top_k=10):
        _day(day)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise AnalyticsError('Analytics top_k must be 1..100')
        # Equal scores: Redis reverse byte-lexicographic member order, NOT numeric.
        return [{'movie_id': int(mid), 'analytics_score': float(score)} for mid, score in
                self.client.zrevrange(f'{PREFIX}:hot:{day}', 0, top_k - 1, withscores=True)]

    def ensure_group(self):
        try:
            self.client.xgroup_create(STREAM_KEY, GROUP, id='0-0', mkstream=True)
        except Exception as exc:
            if not str(exc).startswith('BUSYGROUP'):
                raise
        groups = self.client.xinfo_groups(STREAM_KEY)
        names = [g.get('name', g.get(b'name')) for g in groups]
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        if names != [GROUP]:
            raise AnalyticsError('Only the dedicated analytics consumer group is supported')

    def consume_once(self, consumer='manual-p3', count=100, min_idle_ms=60000,
                     reclaim_cursor=None, max_seconds=10, first_lane=None):
        """Bounded nonblocking poll; callers may carry the returned state on restart.

        ``reclaim_cursor`` and ``first_lane`` default to this instance's state.
        A new instance without that state starts at pending/0-0 again. Count=1
        fairness across restarts therefore requires explicit caller checkpointing.
        """
        reclaim_cursor = self._reclaim_cursor if reclaim_cursor is None else reclaim_cursor
        first_lane = self._next_lane if first_lane is None else first_lane
        if (not isinstance(consumer, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', consumer)
                or type(count) is not int or not 1 <= count <= 100
                or type(min_idle_ms) is not int or not 60000 <= min_idle_ms <= 86400000
                or not isinstance(reclaim_cursor, str) or not re.fullmatch(r'\d+-\d+', reclaim_cursor)
                or first_lane not in ('pending', 'fresh')
                or type(max_seconds) not in (int, float) or not 1 <= max_seconds <= 30):
            raise AnalyticsError('Invalid bounded consumer options')
        end = time.monotonic() + max_seconds
        # Alternate even after a fetch failure or an exhausted time budget.
        self._next_lane = 'fresh' if first_lane == 'pending' else 'pending'
        self._reclaim_cursor = reclaim_cursor
        self.ensure_group()
        report = {'processed': 0, 'duplicates': 0, 'acked': 0, 'failed': 0,
                  'prune_failed': 0, 'claimed': 0, 'reclaim_cursor': reclaim_cursor,
                  'deleted_pending': 0, 'failed_ids': [], 'deferred': 0,
                  'fresh': 0, 'next_lane': self._next_lane}
        lanes = {'pending': [], 'fresh': []}

        def fetch(lane, budget):
            if budget <= 0 or time.monotonic() >= end:
                return
            # Fetch errors propagate; never report a failed Redis read as success.
            if lane == 'pending':
                reply = self.client.xautoclaim(STREAM_KEY, GROUP, consumer, min_idle_ms,
                                               self._reclaim_cursor, count=budget)
                cursor, messages = reply[:2]
                cursor = cursor.decode() if isinstance(cursor, bytes) else cursor
                if not isinstance(cursor, str) or not re.fullmatch(r'\d+-\d+', cursor):
                    raise AnalyticsError('Invalid analytics reclaim cursor')
                self._reclaim_cursor = report['reclaim_cursor'] = cursor
                deleted = reply[2] if len(reply) > 2 else []
                report['deleted_pending'] += len(deleted)
                if deleted:
                    _failure('missing_pending', AnalyticsError('Missing analytics pending payload'))
                report['claimed'] += len(messages)
            else:
                # Omit BLOCK (BLOCK=0 would wait forever).
                reply = self.client.xreadgroup(GROUP, consumer, {STREAM_KEY: '>'}, count=budget)
                messages = [message for _, batch in reply for message in batch]
                report['fresh'] += len(messages)
            if len(messages) > budget:
                raise AnalyticsError('Analytics fetch exceeded batch budget')
            lanes[lane].extend(messages)

        second_lane = 'fresh' if first_lane == 'pending' else 'pending'
        # Both lanes have a reserved quota for count>=2; count=1 rotates.
        first_budget = (count + 1) // 2
        fetch(first_lane, first_budget)
        fetch(second_lane, count - len(lanes[first_lane]))
        remaining = count - len(lanes[first_lane]) - len(lanes[second_lane])
        # At most one refill, using the cursor from the previous claim. No scan loop.
        if remaining:
            fetch(first_lane, remaining)
        # Interleave lanes so an entire poison batch cannot precede healthy work.
        batches = []
        for index in range(max(len(lanes[first_lane]), len(lanes[second_lane]))):
            for lane in (first_lane, second_lane):
                if index < len(lanes[lane]):
                    batches.append(lanes[lane][index])
        attempted = 0
        for message_id, event in batches:
            if time.monotonic() >= end:
                break  # Fetched but unprocessed entries remain pending.
            attempted += 1
            try:
                applied = self.process(event)
                if applied not in (0, 1):
                    raise AnalyticsError('Unexpected analytics update result')
                report['processed' if applied else 'duplicates'] += 1
                acked = self.client.xack(STREAM_KEY, GROUP, message_id)
                if acked != 1:
                    raise AnalyticsError('Analytics ACK not confirmed')
                report['acked'] += 1
            except Exception as exc:
                report['failed'] += 1
                report['failed_ids'].append(message_id)
                _failure('consumption', exc)
                continue
            try:
                self.client.eval(PRUNE_LUA, 1, STREAM_KEY, GROUP, message_id)
            except Exception as exc:
                report['prune_failed'] += 1
                _failure('retention', exc)
        report['deferred'] = len(batches) - attempted
        return report


def _day(day):
    if not isinstance(day, str) or not re.fullmatch(r'\d{8}', day):
        raise AnalyticsError('Analytics day must be YYYYMMDD')
    datetime.strptime(day, '%Y%m%d')
