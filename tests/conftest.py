import pytest
from fastapi.testclient import TestClient
from backend.api.router import app

@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app) 