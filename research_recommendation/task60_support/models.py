from django.db import models


class Movie(models.Model):
    id = models.PositiveBigIntegerField(primary_key=True)
    title = models.CharField(max_length=255)
    movielens_id = models.CharField(max_length=32, unique=True)
    genre_labels = models.JSONField(default=list)
    poster = models.CharField(max_length=1000, blank=True)
    summary = models.TextField(blank=True)
