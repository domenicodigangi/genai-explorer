import pytest
from fastapi.testclient import TestClient

from api.index import app, _load_data


@pytest.fixture(scope="session", autouse=True)
def load_data():
    """Pre-load data once for all tests."""
    _load_data()


@pytest.fixture(scope="session")
def client():
    """FastAPI test client."""
    return TestClient(app)
