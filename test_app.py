from app import app
with app.test_client() as client:
    response = client.post('/evaluate')
    print(response.status_code)
    print(response.data.decode('utf-8')[:200])
