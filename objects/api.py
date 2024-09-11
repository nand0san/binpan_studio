import requests


class ApiClient:
    def __init__(self, base_url, api_key=None, api_secret=None):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret

    def make_request(self, endpoint, method='GET', headers=None, params=None, data=None):
        url = f"{self.base_url}/{endpoint}"
        if headers is None:
            headers = {}
        if self.api_key:
            headers['X-Api-Key'] = self.api_key
        response = requests.request(method, url, headers=headers, params=params, json=data)
        return response

    def get(self, endpoint, params=None):
        return self.make_request(endpoint, 'GET', params=params)

    def post(self, endpoint, data=None):
        return self.make_request(endpoint, 'POST', data=data)

    # Aquí puedes agregar métodos adicionales para PUT, DELETE, etc., según sea necesario.
