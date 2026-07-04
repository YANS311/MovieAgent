class RemoveHopByHopHeaders:
    """移除 wsgiref 不允许的 hop-by-hop 响应头，修复 Django 4.2 的 AssertionError"""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if 'Connection' in response:
            del response['Connection']
        return response
