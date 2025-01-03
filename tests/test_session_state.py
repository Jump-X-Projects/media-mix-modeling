import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from io import StringIO
import asyncio
import threading
import time

from backend.api.router import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'google_spend': np.random.uniform(100, 1000, 10),
        'facebook_spend': np.random.uniform(100, 1000, 10),
        'revenue': np.random.uniform(1000, 5000, 10)
    })
    return data

def test_session_creation(client):
    """Test session is created and ID is returned"""
    response = client.get("/api/data")
    assert response.status_code == 404  # No data yet
    assert "session_id" in client.cookies

def test_data_upload(client, sample_data):
    """Test data upload and persistence in session"""
    csv_data = StringIO()
    sample_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", csv_data.getvalue())}
    )
    assert response.status_code == 200
    
    # Verify data is stored
    response = client.get("/api/data")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(sample_data)

def test_session_isolation(client, sample_data):
    """Test data isolation between sessions"""
    # First session
    csv_data = StringIO()
    sample_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", csv_data.getvalue())}
    )
    assert response.status_code == 200
    session1_id = client.cookies["session_id"]
    
    # Second session
    client.cookies.clear()
    response = client.get("/api/data")
    assert response.status_code == 404  # No data in new session
    assert client.cookies["session_id"] != session1_id

def test_concurrent_model_training(client, sample_data):
    """Test concurrent model training requests"""
    # Upload data
    csv_data = StringIO()
    sample_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", csv_data.getvalue())}
    )
    assert response.status_code == 200
    
    # Train multiple models concurrently
    def train_model():
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", csv_data.getvalue())},
            data={
                "model_type": "linear",
                "media_columns": "google_spend,facebook_spend",
                "target_column": "revenue"
            }
        )
        assert response.status_code == 200
        return response.json()["model_id"]
    
    threads = []
    model_ids = set()
    for _ in range(3):
        thread = threading.Thread(target=lambda: model_ids.add(train_model()))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify unique model IDs
    assert len(model_ids) == 3

def test_session_cleanup(client, sample_data):
    """Test session cleanup"""
    # Upload data
    csv_data = StringIO()
    sample_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", csv_data.getvalue())}
    )
    assert response.status_code == 200
    
    # Train model
    response = client.post(
        "/api/train",
        files={"file": ("data.csv", csv_data.getvalue())},
        data={
            "model_type": "linear",
            "media_columns": "google_spend,facebook_spend",
            "target_column": "revenue"
        }
    )
    assert response.status_code == 200
    
    # Cleanup session
    response = client.post("/api/cleanup")
    assert response.status_code == 200
    
    # Verify data is cleared
    response = client.get("/api/data")
    assert response.status_code == 404

def test_session_persistence(client, sample_data):
    """Test session data persists across requests"""
    # Upload data
    csv_data = StringIO()
    sample_data.to_csv(csv_data, index=False)
    csv_data.seek(0)
    
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", csv_data.getvalue())}
    )
    assert response.status_code == 200
    
    # Multiple requests should see same data
    for _ in range(3):
        response = client.get("/api/data")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(sample_data)

def test_session_load(client, sample_data):
    """Test session handling under load"""
    # Create multiple sessions with data
    sessions = []
    for i in range(5):
        client.cookies.clear()
        
        csv_data = StringIO()
        sample_data.to_csv(csv_data, index=False)
        csv_data.seek(0)
        
        response = client.post(
            "/api/upload",
            files={"file": ("data.csv", csv_data.getvalue())}
        )
        assert response.status_code == 200
        
        sessions.append(client.cookies["session_id"])
    
    # Verify all sessions maintain data
    for session_id in sessions:
        client.cookies["session_id"] = session_id
        response = client.get("/api/data")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(sample_data) 