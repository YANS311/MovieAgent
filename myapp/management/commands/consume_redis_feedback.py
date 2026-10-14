"""Explicit bounded analytics consumer. Never registered as an automatic worker.

Django startup still runs AppConfig.ready(): do not invoke this command in the
training environment. Use an independently audited environment and settings.
"""
import json
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from myapp.services.redis_analytics import analytics_from_settings


class Command(BaseCommand):
    help = 'Consume auxiliary feedback statistics with one instance and finite budgets.'
    requires_system_checks = []
    requires_migrations_checks = False

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=100)
        parser.add_argument('--max-polls', type=int, default=10)
        parser.add_argument('--max-seconds', type=float, default=30)

    def handle(self, *args, **options):
        count = options['count']
        max_polls = options['max_polls']
        max_seconds = options['max_seconds']
        if (type(count) is not int or not 1 <= count <= 100
                or type(max_polls) is not int or not 1 <= max_polls <= 1000
                or type(max_seconds) not in (int, float) or not 1 <= max_seconds <= 300):
            raise CommandError('Require count=1..100, max-polls=1..1000, max-seconds=1..300.')
        if getattr(settings, 'REDIS_ANALYTICS_ENABLED', False) is not True:
            raise CommandError('REDIS_ANALYTICS_ENABLED must be True; consumer not started.')
        url = getattr(settings, 'REDIS_ANALYTICS_URL', None)
        if not isinstance(url, str) or not url:
            raise CommandError('Dedicated REDIS_ANALYTICS_URL required; consumer not started.')

        deadline = time.monotonic() + max_seconds
        try:
            # Exactly once, outside the loop. The factory checks Session DB isolation.
            service = analytics_from_settings(settings)
        except Exception:
            raise CommandError('Analytics configuration or connection initialization failed.') from None
        if service is None:
            raise CommandError('Analytics disabled during initialization; consumer not started.')

        totals = dict.fromkeys(('processed', 'duplicates', 'acked', 'failed', 'prune_failed',
                               'claimed', 'fresh', 'deleted_pending', 'deferred'), 0)
        polls = 0
        reason = 'poll_limit'
        fatal = None
        try:
            for _ in range(max_polls):
                remaining = deadline - time.monotonic()
                # Existing consume_once requires >=1 second. Do not round up and
                # start a poll that extends beyond the remaining cooperative budget.
                if remaining < 1:
                    reason = 'time_budget'
                    break
                report = service.consume_once(count=count, max_seconds=min(10, remaining))
                # Omit first_lane/reclaim_cursor: preserve both on this instance.
                polls += 1
                for key in totals:
                    totals[key] += report[key]
                # Per-message failures remain Pending; continue only within budgets.
                if report['claimed'] + report['fresh'] == 0 and polls < max_polls:
                    delay = min(0.1, max(0, deadline - time.monotonic()))
                    if delay:
                        time.sleep(delay)
        except Exception:
            reason = 'infrastructure_error'
            fatal = 'Analytics polling failed; stopped without retrying infrastructure errors.'
        finally:
            try:
                # This client belongs solely to this command, never the Session backend.
                service.client.close()
            except Exception:
                if fatal is None:
                    reason = 'cleanup_error'
                    fatal = 'Analytics client cleanup failed; polling has stopped.'

        # Aggregate counts only: no payloads, event IDs, cursor, URL or credentials.
        self.stdout.write(json.dumps({'polls': polls, 'exit_reason': reason, **totals}, sort_keys=True))
        if fatal:
            raise CommandError(fatal) from None
