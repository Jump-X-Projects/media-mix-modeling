import os
import joblib
import json
from typing import Dict, Any, Optional
from pathlib import Path

class ModelPersistence:
    def __init__(self, storage_dir: str = "model_storage"):
        """Initialize model persistence with storage directory"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "model_metadata.json"
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize or load metadata file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model_id: str, model: Any, metadata: Dict[str, Any]) -> None:
        """Save model and its metadata to disk"""
        # Save model using joblib
        model_path = self.storage_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        self.metadata[model_id] = {
            "model_path": str(model_path),
            **metadata
        }
        self._save_metadata()
    
    def load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model and its metadata from disk"""
        if model_id not in self.metadata:
            return None
            
        model_info = self.metadata[model_id]
        model_path = model_info["model_path"]
        
        if not os.path.exists(model_path):
            return None
            
        try:
            model = joblib.load(model_path)
            return {
                "model": model,
                **{k: v for k, v in model_info.items() if k != "model_path"}
            }
        except Exception:
            return None
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model and its metadata"""
        if model_id not in self.metadata:
            return False
            
        model_path = self.metadata[model_id]["model_path"]
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
            del self.metadata[model_id]
            self._save_metadata()
            return True
        except Exception:
            return False
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models and their metadata"""
        return {
            model_id: {k: v for k, v in info.items() if k != "model_path"}
            for model_id, info in self.metadata.items()
        } 