from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
import io
import uuid
import json
from typing import Dict, Any

from backend.data.processor import DataProcessor
from backend.models.model_factory import ModelFactory, ModelCreationError
from backend.models.spend_optimizer import SpendOptimizer

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key",
    session_cookie="session_id"  # Use session_id as cookie name
)

# Store session data
sessions: Dict[str, Dict[str, Any]] = {}

async def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    if 'session_id' not in request.session:
        session_id = str(uuid.uuid4())
        request.session['session_id'] = session_id
        return session_id
    return request.session['session_id']

async def get_session_data(session_id: str = Depends(get_session_id)) -> Dict[str, Any]:
    """Get session data"""
    if session_id not in sessions:
        sessions[session_id] = {
            'data': None,
            'processor': DataProcessor(),
            'models': {}
        }
    return sessions[session_id]

@app.post("/api/upload")
async def upload_data(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    session_data: dict = Depends(get_session_data)
):
    """Upload and validate data"""
    try:
        # Read CSV data
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode()))
        
        # Initialize processor
        processor = session_data['processor']
        
        # Identify columns
        spend_cols = [col for col in data.columns if 'spend' in col.lower()]
        revenue_col = next(col for col in data.columns if 'revenue' in col.lower())
        
        # Setup processor with data
        processor.setup(data, spend_cols, revenue_col)
        
        # Store in session
        session_data['data'] = data
        
        return JSONResponse(
            content={"message": "Data uploaded successfully"},
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            content={"details": str(e)},
            status_code=400
        )

@app.get("/api/data")
async def get_data(session_data: dict = Depends(get_session_data)):
    """Get session data"""
    if session_data['data'] is None:
        return JSONResponse(
            content={"details": "No data uploaded"},
            status_code=404
        )
    return session_data['data'].to_dict(orient='records')

@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    media_columns: str = Form(...),
    target_column: str = Form(...),
    session_data: dict = Depends(get_session_data)
):
    """Train a model"""
    try:
        # Read data if not already uploaded
        if session_data['data'] is None:
            contents = await file.read()
            data = pd.read_csv(io.StringIO(contents.decode()))
            session_data['data'] = data
        else:
            data = session_data['data']
        
        # Parse media columns
        media_cols = media_columns.split(',')
        
        # Validate columns exist in data
        missing_cols = [col for col in media_cols + [target_column] if col not in data.columns]
        if missing_cols:
            return JSONResponse(
                content={"details": f"Missing required columns: {', '.join(missing_cols)}"},
                status_code=400
            )
        
        # Initialize processor
        processor = session_data['processor']
        
        # Setup processor
        processor.setup(data, media_cols, target_column)
        
        # Create and train model
        factory = ModelFactory()
        model = factory.create_model(model_type)
        
        X, y = processor.process(data)
        model.fit(X, y)
        
        # Generate model ID and store
        model_id = str(uuid.uuid4())
        session_data['models'][model_id] = model
        
        return {"model_id": model_id}
        
    except ModelCreationError as e:
        return JSONResponse(
            content={"details": str(e)},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"details": str(e)},
            status_code=400
        )

@app.get("/api/models/{model_id}")
async def get_model(
    model_id: str,
    session_data: dict = Depends(get_session_data)
):
    """Get model by ID"""
    if model_id not in session_data['models']:
        return JSONResponse(
            content={"details": "Model not found"},
            status_code=404
        )
    return {"model_id": model_id}

@app.get("/api/models")
async def list_models(session_data: dict = Depends(get_session_data)):
    """List all models"""
    return {"models": list(session_data['models'].keys())}

@app.post("/api/optimize")
async def optimize_spend(
    params: dict,
    session_data: dict = Depends(get_session_data)
):
    """Optimize spend allocation"""
    if not session_data['models']:
        return JSONResponse(
            content={"details": "No models available"},
            status_code=404
        )
        
    try:
        # Use the latest model
        model_id = list(session_data['models'].keys())[-1]
        model = session_data['models'][model_id]
        
        # Create optimizer
        optimizer = SpendOptimizer(
            model=model,
            feature_names=session_data['processor'].media_columns,
            historical_data=session_data['data']
        )
        
        # Get current spend
        current_spend = session_data['data'][session_data['processor'].media_columns].mean()
        
        # Optimize
        result = optimizer.optimize(
            current_spend=current_spend.values,
            total_budget=params['budget']
        )
        
        return result
        
    except Exception as e:
        return JSONResponse(
            content={"details": str(e)},
            status_code=400
        )

@app.post("/api/cleanup")
async def cleanup_session(session_id: str = Depends(get_session_id)):
    """Clean up session data"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleaned up"}

@app.post("/api/models/{model_id}/predict")
async def predict(
    model_id: str,
    features: dict,
    session_data: dict = Depends(get_session_data)
):
    """Make predictions with a model"""
    try:
        if model_id not in session_data['models']:
            return JSONResponse(
                content={"details": "Model not found"},
                status_code=404
            )
            
        model = session_data['models'][model_id]
        processor = session_data['processor']
        
        # Validate features
        required_features = processor.media_columns
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            return JSONResponse(
                content={"details": f"Missing features: {', '.join(missing_features)}"},
                status_code=400
            )
            
        # Create feature DataFrame
        X = pd.DataFrame(features)
        
        # Make prediction
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
        
    except Exception as e:
        return JSONResponse(
            content={"details": str(e)},
            status_code=400
        ) 