"""Namespaced account feedback and auditable tool calls; never scientific Train labels."""
import uuid
from django.conf import settings
from django.db import models
from django.db.models import Q

class UnifiedMovieFeedback(models.Model):
    user=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    research_movie=models.ForeignKey('research_recommendation.ResearchMovie',null=True,blank=True,on_delete=models.PROTECT)
    business_movie=models.ForeignKey('myapp.Movie',null=True,blank=True,on_delete=models.PROTECT)
    rating=models.FloatField(null=True,blank=True)
    collected=models.BooleanField(default=False)
    preference=models.CharField(max_length=8,default='',choices=[('','未设置'),('like','喜欢'),('dislike','不喜欢')])
    watched=models.BooleanField(default=False)
    updated_at=models.DateTimeField(auto_now=True)
    class Meta:
        constraints=[models.CheckConstraint(condition=(Q(research_movie__isnull=False,business_movie__isnull=True)|Q(research_movie__isnull=True,business_movie__isnull=False)),name='ch5_feedback_one_identity'),models.CheckConstraint(condition=Q(rating__isnull=True)|Q(rating__gte=.5,rating__lte=5),name='ch5_feedback_rating_range'),models.CheckConstraint(condition=Q(preference__in=['','like','dislike']),name='ch5_feedback_preference'),models.UniqueConstraint(fields=['user','research_movie'],name='ch5_feedback_user_research'),models.UniqueConstraint(fields=['user','business_movie'],name='ch5_feedback_user_business')]

class ProductAgentSession(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    user=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True)

class ProductAgentTurn(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    session=models.ForeignKey(ProductAgentSession,on_delete=models.CASCADE)
    user_text=models.CharField(max_length=1500)
    llm_model=models.CharField(max_length=100,blank=True)
    llm_digest=models.CharField(max_length=100,blank=True)
    planning_mode=models.CharField(max_length=40)
    plan=models.JSONField(default=list)
    outcome=models.CharField(max_length=40,default='STARTED')
    response=models.JSONField(default=dict)
    created_at=models.DateTimeField(auto_now_add=True)

class ProductToolCall(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    turn=models.ForeignKey(ProductAgentTurn,on_delete=models.CASCADE,related_name='executions')
    tool_name=models.CharField(max_length=100)
    arguments=models.JSONField(default=dict)
    outcome=models.CharField(max_length=40)
    result=models.JSONField(default=dict)
    latency_ms=models.FloatField(default=0)
    recommendation_trace=models.ForeignKey('research_recommendation.RecommendationTrace',null=True,on_delete=models.SET_NULL)
    created_at=models.DateTimeField(auto_now_add=True)

class PendingPreferenceAction(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    session=models.ForeignKey(ProductAgentSession,on_delete=models.CASCADE)
    user=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    arguments=models.JSONField()
    created_at=models.DateTimeField(auto_now_add=True)
    expires_at=models.DateTimeField()
    applied=models.BooleanField(default=False)
    result=models.JSONField(default=dict)
