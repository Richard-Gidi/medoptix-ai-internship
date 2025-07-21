import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any , Tuple

logger = logging.getLogger(__name__)


# create a class 
# = helps us to load out from the pickle file 
class MedoptixPredictor:

    """ Helps handle all the ML prediction for Medoptix"""

    # step 1 - instantiate
    def __init__(self):
        self.models = {}     # models output comes in a JSON format
        self.load_models

    
    # step 2 - load our model
    def load_models(self) -> bool:
        """ load all trained models"""

        try:
            model_path = "models/"
            self.models["feature_preprocessor"] = joblib.load(f"{model_path}medoptix_prediction_features_preprocessor.pkl")        
            self.models["dropout_model"] = joblib.load(f"{model_path}medoptix_prediction_dropout_model.pkl")        
            self.models["feature_names"] = joblib.load(f"{model_path}medoptix_prediction_features_name.pkl")   

            logger.info("Models loaded successfully")
            return True   
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
        
    
    # step 3 - predict the dropout
    def predict_dropout (self, patient_data: Dict[str, Any]) -> Tuple[float, int, str, list]:
        """ predict droupouts and return all relevant information"""

        try:
            df = pd.DataFrame([patient_data])

            # step 3 a - put in the data and  preprocess and also run it through our model
            X_processed = self.models["feature_preprocessor"].transform (df)
            dropout_prob = self.models["dropout_model"].predict_proba(X_processed)[0,1]

            # step 3b - recommendation and risk level
            risk_level, recommendation = self._get_risk_recommendation(dropout_prob)
            return float(dropout_prob), 0, risk_level, recommendation
        except Exception as e:
            logger.error(f"Error in dropout predictions {str(e)}")
            return 0.5, 0, "Medium", ["Unable to generated recommendation"]
        

    
    # step 4 
    def _get_risk_recommendation(self, dropout_prob:float) -> Tuple[str, list]:
        """ Generating risk level and recommendation"""

        if dropout_prob > 0.7:
            return "High", [
                "Schedule immediate follow up call",
                "Assign dedicated support specialist",
                "offer flexible scheduling options"
            ]
        elif dropout_prob > 0.3:
            return "Medium", [
                "Send wekely checkin messages",
                "monitor attendance closely",
                "offer additional support if needed"
            ]
        else:
            return "Low", [
                "Continue standard care procedures",
                "maintain regular check-ins"
            ]






