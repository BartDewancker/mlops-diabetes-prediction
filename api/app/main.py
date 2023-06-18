# main.py
import os
from http.client import HTTPException
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from schemas import Patient

load_dotenv()

VERSION = os.environ.get('API_VERSION')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(
    r'outputs/diabetes-prediction-model/pipeline_logistic_regression.pkl')

def patient_to_sample(patient: Patient, model: BaseEstimator) -> pd.DataFrame:
    """
    Creates an input sample for the model from the patient data.

    Input:
    - patient, the patient to create a sample from
    - model, the model to create a sample for

    Returns:
    - pandas DataFrame that can be inputted to the model's prediction method.

    Raises:
    - ValueError when an attribute is missing, or when at least one of the patients's categorical attributes is unknown to the model.
    """

    # check that the value of the categorical attributes is known by the model's one-hot-encoder
    preprocessor = model.named_steps['preprocessor']
    categorical_features = preprocessor.transformers[1][2]

    options_per_feature = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
    for feature, options in zip(categorical_features, options_per_feature):
        if not getattr(patient, feature) in options:
            raise ValueError(
                "At least one of the patient's attributes is unknown to the model.")

    # model expects a dataframe, with columns in specific order
    transformers = model.named_steps['preprocessor'].transformers
    features = []
    for transformation in transformers:
        features += transformation[2]
    sample = pd.DataFrame({f:[getattr(patient, f)] for f in features})

    return sample

@app.get("/")
def read_root():
    """
    Read root
    """
    return {"Hello": "Diabetes Prediction API"}

@app.post("/patient",
    description="Predict if a patient is about to have diabetes.",
    status_code=status.HTTP_200_OK,
    response_model=bool)
async def predict_diabetes(patient: Patient) -> bool:
    """
    Predict if the patien is about to have diabetes.

    Input:
    - patient to make prediction for

    Returns:
    - bool, True if customer is about to have diabetes.

    """
    try:
        sample = patient_to_sample(patient, model)
        return model.predict(sample)[0]
    except ValueError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST) from err


@app.get("/debug/environment")
def get_environment():
    """
    Get environment variables.
    """
    return os.environ
