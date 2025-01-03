from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import uuid
from io import StringIO

from backend.models.model_factory import ModelFactory

app = FastAPI(title="Media Mix Modeling API")

# Store trained models in memory (in production, use proper model storage)
models = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List available model types"""
    return ModelFactory.get_available_models()

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_column: str = Form(...),
    feature_columns: List[str] = Form(...)
):
    """Train a model on uploaded data"""
    try:
        # Read CSV data
        content = await file.read()
        data = pd.read_csv(StringIO(content.decode()))
        
        # Validate columns
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in data")
        for col in feature_columns:
            if col not in data.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found in data")
        
        # Create and train model
        X = data[feature_columns]
        y = data[target_column]
        
        model = ModelFactory.create_model(model_type)
        model.fit(X, y)
        
        # Generate model ID and store model
        model_id = str(uuid.uuid4())
        models[model_id] = model
        
        # Calculate metrics
        metrics = model.evaluate(X, y)
        
        return {
            "model_id": model_id,
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    data: Dict[str, List[float]],
    include_uncertainty: bool = False
):
    """Make predictions using a trained model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Convert input data to DataFrame
        X = pd.DataFrame(data)
        
        # Get predictions
        model = models[model_id]
        predictions = model.predict(X)
        
        response = {"predictions": predictions.tolist()}
        
        # Add uncertainty estimates for Bayesian models
        if include_uncertainty and hasattr(model, 'get_uncertainty'):
            uncertainty = model.get_uncertainty(X)
            response["uncertainty"] = uncertainty.tolist()
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_column: str = Form(...),
    feature_columns: List[str] = Form(...),
    cv_folds: int = Form(5)
):
    """Evaluate model using cross-validation"""
    try:
        # Read CSV data
        content = await file.read()
        data = pd.read_csv(StringIO(content.decode()))
        
        # Validate columns
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in data")
        for col in feature_columns:
            if col not in data.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found in data")
        
        # Create model and perform cross-validation
        X = data[feature_columns]
        y = data[target_column]
        
        model = ModelFactory.create_model(model_type)
        cv_scores = ModelFactory.cross_validate(model, X, y, cv=cv_folds)
        
        return {
            "cv_scores": {
                "mean_r2": float(np.mean(cv_scores['test_r2'])),
                "std_r2": float(np.std(cv_scores['test_r2'])),
                "mean_rmse": float(np.mean(cv_scores['test_rmse'])),
                "std_rmse": float(np.std(cv_scores['test_rmse'])),
                "mean_mae": float(np.mean(cv_scores['test_mae'])),
                "std_mae": float(np.std(cv_scores['test_mae']))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 