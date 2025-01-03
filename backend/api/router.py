from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import uuid
import json
from io import StringIO

from backend.models.model_factory import ModelFactory
from backend.models.data_processor import DataProcessor

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store trained models in memory
model_store = {}

class PredictionRequest(BaseModel):
    model_id: str
    features: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    uncertainty: Optional[Dict[str, List[float]]] = None

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "success": True,
        "data": {
            "models": ["linear", "lightgbm", "xgboost", "bayesian"]
        }
    }

@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    media_columns: str = Form(...),
    target_column: str = Form(...),
    num_epochs: Optional[int] = Form(1000),
    learning_rate: Optional[float] = Form(0.01)
):
    """Train a model on uploaded data"""
    try:
        print(f"Training model: {model_type}")
        print(f"Media columns: {media_columns}")
        print(f"Target column: {target_column}")
        
        # Read data
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Process columns
        media_cols = media_columns.split(',')
        if not all(col in df.columns for col in media_cols + [target_column]):
            missing = [col for col in media_cols + [target_column] if col not in df.columns]
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
            
        # Create and train model
        model = ModelFactory.create_model(
            model_type,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        print(f"Created model: {model.__class__.__name__}")
        
        X = df[media_cols]
        y = df[target_column]
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        model.fit(X, y)
        print("Model training completed")
        
        # Generate model ID and store
        model_id = str(uuid.uuid4())
        model_store[model_id] = {
            'model': model,
            'media_columns': media_cols,
            'target_column': target_column
        }
        print(f"Model stored with ID: {model_id}")
        
        return {"model_id": model_id}
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        if request.model_id not in model_store:
            raise HTTPException(status_code=404, detail="Model not found")
            
        stored = model_store[request.model_id]
        model = stored['model']
        media_cols = stored['media_columns']
        
        # Validate features
        if not all(col in request.features for col in media_cols):
            raise HTTPException(status_code=400, detail="Missing features")
            
        # Create feature DataFrame
        X = pd.DataFrame(request.features)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        # Get uncertainty estimates for Bayesian model
        response = {"predictions": predictions}
        if hasattr(model, 'get_uncertainty_estimates'):
            lower, upper = model.get_uncertainty_estimates(X)
            response["uncertainty"] = {
                "lower": lower.tolist(),
                "upper": upper.tolist()
            }
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    media_columns: str = Form(...),
    target_column: str = Form(...),
    cv_folds: int = Form(3)
):
    """Evaluate model using cross-validation"""
    try:
        # Read data
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Process columns
        media_cols = media_columns.split(',')
        if not all(col in df.columns for col in media_cols + [target_column]):
            raise HTTPException(status_code=400, detail="Invalid column names")
            
        # Create model
        model = ModelFactory.create_model(model_type)
        
        X = df[media_cols]
        y = df[target_column]
        
        # Perform cross-validation
        metrics = model.cross_validate(X, y, cv_folds)
        
        return {"metrics": metrics}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 