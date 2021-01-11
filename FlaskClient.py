import requests
import json

url = 'http://127.0.0.1:5000/yuNet'
session = requests.session()
headers = {'Content-Type': 'application/json'}
data = {'action': 'needAddress'}
response = session.post(url, data=json.dumps(data), headers=headers)
print(json.loads(response.content))
