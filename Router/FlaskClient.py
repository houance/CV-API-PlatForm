import requests
import json

url = 'http://127.0.0.1:4000/YuNet'
session = requests.session()
response = session.post(url)
print(str(response.content, encoding='utf-8'))
