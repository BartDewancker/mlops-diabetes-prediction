from typing import Literal
from pydantic import BaseModel, validator, Field
import numpy as np

class Patient(BaseModel):
    patient_id: str = Field("",
        description="Patient ID")
    gender: Literal["Female", "Male", "Other"] = Field(
        description="Whether the patient is a female, male or other")
    age: float = Field(
        description="The age of the patient")
    hypertension: bool = Field(
        description="Whether the patient has hypertension or not")
    heart_disease: bool = Field(
        description="Whether the patient has heart disease or not")
    smoking_history: Literal["never", "No Info", "current", "former", "ever", "not current"] = Field(
        description="Some info about the smoking history of the patient")
    bmi: float = Field(
        description="The BMI of the patient")
    HbA1c_level: float = Field(
        description="The HbA1c level of the patient")
    blood_glucose_level: int = Field(
        description="The blood glucose level of the patient")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "1",
                "gender": "Male",
                "age": 67,
                "hypertension": True,
                "heart_disease": False,
                "smoking_history": 'former',
                "bmi": 25.6,
                "HbA1c_level": 5,
                "blood_glucose_level": 140,
            }
        }

    @validator("age", "bmi", "HbA1c_level", "blood_glucose_level")
    def is_finite(cls, value):
        if not np.isfinite(value):
            raise ValueError("must be finite.")
        return value
