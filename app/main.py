from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any , Tuple, List
from pydantic import BaseModel, Field
import os
from app.prediction import MedoptixPredictor

logger = logging.getLogger(__name__)

predictor = MedoptixPredictor()

# step 1 : initialize our fastapi app
app = FastAPI(
    title="Medoptix API",
    description= "AI-powered droput prediction and patient clustering for physical therapy",
    version="1.0.0"
)


'''@app.get("/")
def read_root():
    return {"message": "Welcome to Medoptix API"}'''

# step 2 : Add CORS middleware - utilities they help in  setting up the browser to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,

)


# step 3 :  pydantic model for input data
class PatientData(BaseModel):
    # patient basic information
    age: float = Field(..., le=120, ge=0, description="Patient's age")
    gender : str = Field(..., description="male or female")
    bmi : float = Field(..., le=100, ge=0, description="Body Mass Index")
    smoker : str = Field(..., description="smoker or non-smoker")
    chronic_cond: str = Field(..., description="chronic condition or not")
    injury_type: str = Field(..., description="type of injury")
    refferal_source: str = Field(..., description="source of referral")
    insurance_type: str = Field(..., description="type of insurance")

    # session information
    n_sessions: int = Field(..., ge=0, description="Number of sessions attended")
    avg_session_duration: float = Field(..., ge=0, description="Average session duration in minutes")
    first_week : int = Field(..., ge=0, description="Number of sessions in the first week")
    last_week : int = Field(..., ge=0, description="Number of sessions in the last week")
    mean_pain: float = Field(..., description="Mean pain score")
    mean_pain_delta: float = Field(...,  description="Change in mean pain score")
    home_adherence_mean: float = Field(..., ge=0, description="Mean home exercise adherence score")
    satisfaction_mean: float = Field(..., ge=0, description="Mean patient satisfaction score")


    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "gender" : "Female",
                "bmi" : 22.5,
                "smoker" : "No",
                "chronic_cond": "No",
                "injury_type": "Knee Injury",
                "refferal_source": "Doctor",
                "insurance_type": "Private",
                "n_sessions": 10,
                "avg_session_duration": 45.0,
                "first_week": 3,
                "last_week": 2,
                "mean_pain": 5.0,
                "mean_pain_delta": -1.0,
                "home_adherence_mean": 80.0,
                "satisfaction_mean": 4.5
            }
        }    



class DropoutPredictionResponse(BaseModel):
    patient_id: str 
    dropout_probability: float
    risk_level: str
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]
    model_info: Dict[str, Any]



# step 4 : define the API startup function
@app.on_event("startup")
async def startup_event():
    """
    Load models and other resources on startup
    """
    success = predictor.load_models()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load models")
    

# step 5:  defined the health endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and model availability
    """
    models_loaded = bool(predictor.models)

    model_info = {}

    try: 
        feature_name = predictor.models.get("feature_names")
        column_name = predictor.models.get("columns")

        if feature_name is not None:
            if isinstance(feature_name, np.ndarray):
                feature_name = feature_name.tolist()
            elif isinstance(feature_name, list):
                feature_name = feature_name
            else:
                feature_name = [feature_name]
        
        if column_name is not None:
            if isinstance(column_name, np.ndarray):
                column_name = column_name.tolist()
            elif isinstance(column_name, list):
                column_name = column_name
            else:
                column_name = [column_name]
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        feature_name = []
        column_name = []
        
    return HealthResponse (
        status="Healthy",
        models_loaded=models_loaded,
        available_models=list(predictor.models.keys()) if predictor.models else [],
        model_info=model_info
    )


# --- Fix in prepare_patient_data ---
def prepare_patient_data(patient: PatientData) -> pd.DataFrame:
    """Convert the patient data to DataFrame with columns in the order expected by the model"""
    columns = predictor.models.get("columns", [])

    patient_dict = patient.dict()

    # create a DataFrame with the same columns as the model expects
    df_data = {}
    for col in columns:
        if col in patient_dict:
            df_data[col] = [patient_dict[col]]
        else:
            # provide default values
            if col in ['n_sessions', 'first_week', 'last_week']:
                df_data[col] = [0]
            elif col in ['avg_session_duration', 'mean_pain', 'mean_pain_delta', 'home_adherence_mean', 'satisfaction_mean']:
                df_data[col] = [0.0]
            else:
                df_data[col] = ["Unknown"]

    return pd.DataFrame(df_data, columns=columns)


# --- Fix in predict_dropout ---
@app.post("/predict/dropout", response_model=DropoutPredictionResponse)
async def predict_dropout(patient_data: PatientData):
    """
    Predict dropout probability and provide recommendations based on patient data
    """
    try:
        if not predictor.models:
            raise HTTPException(status_code=500, detail="Models not loaded")

        # Make prediction
        dropout_prob, _, risk_level, recommendations = predictor.predict_dropout(patient_data.dict())

        patient_id = f"pt_{hash(str(patient_data.dict())) % 10000:05d}"

        return DropoutPredictionResponse(
            patient_id=patient_id,
            dropout_probability=dropout_prob,
            risk_level=risk_level,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error in dropout prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



# step 8 : we then called the endpoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000")