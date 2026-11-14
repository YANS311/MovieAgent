from django.db import models


class SyntheticMovie(models.Model):
    title = models.CharField(max_length=100)
    movielens_id = models.CharField(max_length=32, null=True)
