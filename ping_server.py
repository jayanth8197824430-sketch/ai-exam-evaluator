import requests
import time

for i in range(5):
    try:
        response = requests.get('http://127.0.0.1:5000/')
        print('Success!', response.status_code)
        break
    except Exception as e:
        print('Attempt', i+1, 'failed')
        time.sleep(2)
