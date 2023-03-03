import requests


data = requests.get('http://127.0.0.1:8002/face_statistic')
print(data.json())