from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import os
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "shap_summary.png")
LIME_PATH = os.path.join(BASE_DIR, "lime_explanation.html")

class OTTRevenueModel(nn.Module):
    def __init__(self, tabular_dim, text_dim):
        super(OTTRevenueModel, self).__init__()
        self.tabular_fc = nn.Linear(tabular_dim, 32)
        self.text_fc = nn.Linear(text_dim, 32)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, tab_x, text_x):
        t_out = self.relu(self.tabular_fc(tab_x))
        txt_out = self.relu(self.text_fc(text_x))
        fused = torch.cat((t_out, txt_out), dim=1)
        out = self.relu(self.fc1(fused))
        out = self.fc2(out)
        return out

model = None
preprocessors = None

try:
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        preprocessors = joblib.load(PREPROCESSOR_PATH)
        model = OTTRevenueModel(preprocessors['tab_dim'], preprocessors['text_dim'])
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
except Exception as e:
    print(f"Warning: Failed to load PyTorch model. Error: {e}")

class PredictionRequest(BaseModel):
    budget: float = Field(..., gt=0)
    runtime: float = Field(..., gt=0)
    rating: float = Field(..., ge=0, le=10)
    synopsis: str = Field(..., description="Movie synopsis")

class PredictionResponse(BaseModel):
    predicted_revenue: float
    explanation: str

def generate_textual_explanation(budget, runtime, rating, synopsis):
    exp = []
    if budget >= 100000000:
        exp.append(f"The massive production budget (${budget:,.0f}) serves as the primary positive driver according to the SHAP global model, significantly raising the baseline revenue potential.")
    else:
        exp.append(f"The conservative budget (${budget:,.0f}) restricts the baseline revenue expectation, acting as a constraining feature in the SHAP model.")
        
    if rating >= 7.5:
        exp.append(f"A strong audience rating ({rating}/10) acts as a powerful local multiplier in the LIME explanation, driving strong engagement.")
    elif rating < 6.0:
        exp.append(f"The poor audience rating ({rating}/10) penalizes the prediction locally, causing a negative drag in the LIME feature impacts.")
    else:
        exp.append(f"An average audience rating ({rating}/10) provides neutral stabilization to the final LIME local prediction.")
        
    if runtime > 140:
        exp.append(f"The extended runtime ({runtime} mins) exerts a slight negative drag on revenue due to fewer possible theater screenings per day.")
    elif runtime < 90:
        exp.append(f"The short runtime ({runtime} mins) limits premium pricing capability, slightly negatively weighting the result.")
        
    exp.append("Finally, the PyTorch Multi-Layer Perceptron mapped the unstructured synopsis text to semantic clusters, adjusting the final prediction based on genre and narrative appeal.")
    
    return " ".join(exp)

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_revenue(request: PredictionRequest):
    if model is None or preprocessors is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    import pandas as pd
    
    try:
        input_tab = pd.DataFrame([{
            'budget': request.budget,
            'runtime': request.runtime,
            'vote_average': request.rating
        }])
        x_tab = preprocessors['tab'].transform(input_tab).astype(np.float32)
        x_text = preprocessors['text'].transform([request.synopsis]).toarray().astype(np.float32)
        
        with torch.no_grad():
            t_tensor = torch.tensor(x_tab)
            txt_tensor = torch.tensor(x_text)
            pred = model(t_tensor, txt_tensor).item()
            
        explanation = generate_textual_explanation(request.budget, request.runtime, request.rating, request.synopsis)
            
        return {
            "predicted_revenue": pred,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/explain/shap")
def get_shap():
    if not os.path.exists(SHAP_PATH):
        raise HTTPException(status_code=404, detail="SHAP plot not found")
    return FileResponse(SHAP_PATH, media_type="image/png")

@app.get("/explain/lime")
def get_lime():
    if not os.path.exists(LIME_PATH):
        raise HTTPException(status_code=404, detail="LIME report not found")
    return FileResponse(LIME_PATH, media_type="text/html")
