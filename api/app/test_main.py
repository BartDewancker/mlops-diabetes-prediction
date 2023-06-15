from fastapi.testclient import TestClient

from main import app
from schemas import Patient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_predict_diabetes_valid_patient():
    sample = Patient.Config.schema_extra["example"]
    response = client.post(
        "/patient",
        json=sample)
    assert response.status_code == 200
    assert response.json() == True

def test_predict_diabetes_valid_patient_without_id():
    sample = Patient.Config.schema_extra["example"]
    sample = {k:v for k,v in sample.items() if k != "patient_id"}
    response = client.post(
        "/customer",
        json=sample)
    assert response.status_code == 200
    assert response.json() == True