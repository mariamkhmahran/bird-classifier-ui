import requests
import json

class Stub:
    """Initiate a stub for REST API to connect with TFServing"""
    def __init__(self, host, port, model_name='birdsClassifier'):
        self.host = host
        self.port = port
        self.model_name = model_name

    def predict(self, data):
        SERVING_ENDPOINT = "http://{}:{}/v1/models/{}:predict".format(self.host, self.port, self.model_name)
        r = requests.post(SERVING_ENDPOINT, data=json.dumps(data))

        return json.loads(r.content.decode('utf-8'))
