from django.db import DatabaseError
from django.http import JsonResponse
from django.utils.cache import add_never_cache_headers
class PrivateProductResponseMiddleware:
    """Cover private success AND exception responses, including trace 404s."""
    def __init__(self,get_response):self.get_response=get_response
    def __call__(self,request):
        response=self.get_response(request)
        if request.path.startswith(('/api/','/recommend','/preferences/','/movie/','/research/','/login/','/register/')):
            add_never_cache_headers(response)
        return response

    def process_exception(self,request,exception):
        if isinstance(exception,DatabaseError):
            response=JsonResponse({'error':'DATABASE_UNAVAILABLE','retryable':True},status=503)
            add_never_cache_headers(response)
            return response
