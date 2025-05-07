import requests
import pytest
import json

# Base URL for the API
BASE_URL = "http://localhost:5000"  # Replace with your actual API URL

# Test fixture for session management
@pytest.fixture
def api_client():
    session = requests.Session()
    yield session
    session.close()

# Sample test for GET endpoint
def test_get_endpoint(api_client):
    response = api_client.get(f"{BASE_URL}/resource")
    assert response.status_code == 200
    
    data = response.json()
    assert "key" in data  # Replace with actual expected data structure

# Sample test for POST endpoint
def test_post_endpoint(api_client):
    payload = {
        "name": "Test Item",
        "value": 123
    }
    
    response = api_client.post(
        f"{BASE_URL}/resource",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data.get("name") == "Test Item"

# Sample test for authentication
def test_authenticated_endpoint(api_client):
    # First authenticate
    auth_payload = {"username": "testuser", "password": "testpass"}
    auth_response = api_client.post(
        f"{BASE_URL}/login", 
        headers={"Content-Type": "application/json"},
        data=json.dumps(auth_payload)
    )
    
    assert auth_response.status_code == 200
    token = auth_response.json().get("token")
    
    # Then access protected endpoint
    response = api_client.get(
        f"{BASE_URL}/protected-resource",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200

# Error case testing
def test_invalid_request(api_client):
    response = api_client.get(f"{BASE_URL}/nonexistent-endpoint")
    assert response.status_code == 404