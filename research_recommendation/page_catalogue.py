"""Small read-only projection over the configured business Movie model."""
from dataclasses import dataclass
from urllib.parse import urlsplit
from django.apps import apps
from django.conf import settings
from django.core.paginator import Paginator


def safe_poster(value):
    if not isinstance(value, str) or not value or len(value) > 1000:
        return None
    if value.startswith(('/static/', '/media/')) and not value.startswith('//'):
        return value
    try:
        parsed = urlsplit(value)
        if parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        return None
    return None


@dataclass(frozen=True)
class MovieCard:
    id: int
    title: str
    genres: tuple
    poster_url: str | None
    summary: str
    source_movie_id: str | None
    is_sensitive: bool = False


class PageCatalogue:
    def queryset(self):
        model = apps.get_model(getattr(settings, 'TASK60_MOVIE_MODEL', 'myapp.Movie'))
        qs = model.objects.all().order_by('pk')
        fields = {f.name: f for f in model._meta.get_fields()}
        if 'genres' in fields and fields['genres'].many_to_many:
            qs = qs.prefetch_related('genres')
        return qs

    @staticmethod
    def card(movie):
        labels = getattr(movie, 'genre_labels', None)
        if labels is None and hasattr(movie, 'genres'):
            labels = [g.name for g in movie.genres.all()]
        labels = tuple(x for x in (labels or []) if isinstance(x, str) and len(x) <= 100)[:12]
        poster = getattr(movie, 'poster', None)
        local = getattr(movie, 'poster_file', None)
        if local:
            try:
                poster = local.url
            except ValueError:
                pass
        if getattr(movie, 'is_sensitive', False):
            poster = None
        return MovieCard(movie.pk, movie.title, labels, safe_poster(poster),
                         str(getattr(movie, 'summary', '') or '')[:4000],
                         getattr(movie, 'movielens_id', None), bool(getattr(movie, 'is_sensitive', False)))

    def browse(self, query='', page=1):
        qs = self.queryset()
        if query:
            qs = qs.filter(title__icontains=query)
        current = Paginator(qs, 12).get_page(page)
        return current, [self.card(row) for row in current.object_list]

    def get(self, movie_id):
        row = self.queryset().filter(pk=movie_id).first()
        return self.card(row) if row is not None else None

    def get_many(self, ids):
        return {row.pk: self.card(row) for row in self.queryset().filter(pk__in=ids)}
