from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any , Tuple


from pydantic import BaseModel, Field
import os



# initialize our fastapi app
app = FastAPI(
    title="Medoptix AI APP",
    description="AI-powered dropout predition and patient clustering for physical therapy",
    version="1.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,

)



