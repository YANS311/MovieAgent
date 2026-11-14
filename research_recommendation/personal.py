"""CPU-only metadata matching for ordinary accounts. No research service calls."""
from collections import Counter
import hashlib
import json
from django.conf import settings
from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from .page_catalogue import PageCatalogue


def catalogue_label():
    return getattr(settings, 'TASK60_MOVIE_MODEL', 'myapp.Movie')


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = ('username', 'email')


def catalogue():
    from .personal_cache import enabled, cached_catalogue
    return cached_catalogue(_catalogue) if enabled() else _catalogue()


def _catalogue():
    # Fetch only display metadata, never embedding/vector columns or research data.
    adapter = PageCatalogue()
    qs = adapter.queryset()
    fields = {f.name for f in qs.model._meta.fields}
    projection = ['pk', 'title'] + [x for x in
        ('genre_labels', 'poster', 'poster_file', 'summary', 'movielens_id', 'is_sensitive') if x in fields]
    return [adapter.card(row) for row in qs.only(*projection).iterator(chunk_size=256)]


class PreferenceForm(forms.Form):
    genres = forms.MultipleChoiceField(required=False, label='喜欢的类型',
                                        widget=forms.CheckboxSelectMultiple)
    movie_ids = forms.CharField(required=False, max_length=600, label='喜欢的电影ID（最多20部，逗号分隔）')

    def __init__(self, *args, cards, **kwargs):
        super().__init__(*args, **kwargs)
        self.cards = {c.id: c for c in cards}
        self.fields['genres'].choices = [(g, g) for g in sorted({g for c in cards for g in c.genres})]

    def clean_genres(self):
        values = sorted(set(self.cleaned_data['genres']))
        if len(values) > 20:
            raise forms.ValidationError('最多选择20种类型。')
        return values

    def clean_movie_ids(self):
        raw = self.cleaned_data['movie_ids'].strip()
        if not raw:
            return []
        parts = raw.replace('，', ',').split(',')
        if len(parts) > 20 or any(not p.strip().isascii() or not p.strip().isdecimal() or len(p.strip()) > 18 for p in parts):
            raise forms.ValidationError('请输入最多20个有效电影ID，以逗号分隔。')
        ids = sorted({int(p.strip()) for p in parts})
        if any(pk not in self.cards for pk in ids):
            raise forms.ValidationError('包含目录中不存在的电影。')
        return ids


def rank(cards, preference):
    genres = preference.genres if preference and preference.catalogue_model == catalogue_label() else []
    selected = preference.movie_ids if preference and preference.catalogue_model == catalogue_label() else []
    weights = Counter({g: 2 for g in genres})
    for movie in cards:
        if movie.id in selected:
            weights.update(set(movie.genres))
    ranked = []
    for movie in cards:
        if movie.id in selected:
            continue
        matched = sorted(g for g in set(movie.genres) if weights[g])
        score = sum(weights[g] for g in matched)
        ranked.append({'movie': movie, 'score': score, 'matched_genres': matched,
                       'reason': '匹配类型：' + '、'.join(matched) if matched else '目录探索：暂无类型匹配'})
    ranked.sort(key=lambda row: (-row['score'], row['movie'].id))
    fingerprint = hashlib.sha256(json.dumps(
        [(c.id, c.title, c.genres) for c in cards], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    results = ranked[:20]
    provenance = {
        'identity_mode': 'DJANGO_ACCOUNT', 'source': 'METADATA_COLD_START',
        'strategy': 'GENRE_MATCH_V1' if weights else 'CATALOGUE_EXPLORATION_V1',
        'model_name': None, 'checkpoint_identity': None, 'research_profile_id': None,
        'catalogue_model': catalogue_label(), 'catalogue_metadata_sha256': fingerprint,
        'candidate_count': len(ranked), 'preference': {'genres': genres, 'movie_ids': selected},
        'preference_updated_at': preference.updated_at.isoformat() if preference else None,
        'genre_weights': dict(sorted(weights.items())),
        'formula': '2 per explicitly selected genre + 1 per selected movie genre; sum matched weights; tie by business movie PK ascending',
        'exclusions': 'selected business movie IDs',
        'results': [{'movie_id': r['movie'].id, 'score': r['score'], 'matched_genres': r['matched_genres']} for r in results],
    }
    return results, provenance
