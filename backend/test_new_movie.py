import os
import joblib
import torch
import pandas as pd
import numpy as np
from app.main import OTTRevenueModel

# Test data based on "The Super Mario Galaxy Movie" (2026)
movie_data = {
    "budget": 150000000.0,
    "runtime": 105.0,
    "rating": 8.2,
    "synopsis": "Mario and Luigi travel across the galaxy to stop Bowser from conquering the entire universe in this highly anticipated animated sequel."
}

def run_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pth")
    preprocessor_path = os.path.join(base_dir, "preprocessor.pkl")

    print(f"Testing new movie: 'The Super Mario Galaxy Movie'")
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("❌ Error: PyTorch model or preprocessor not found. Training failed.")
        return

    try:
        preprocessors = joblib.load(preprocessor_path)
        model = OTTRevenueModel(preprocessors['tab_dim'], preprocessors['text_dim'])
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Preprocess tabular
        input_tab = pd.DataFrame([{
            'budget': movie_data['budget'],
            'runtime': movie_data['runtime'],
            'vote_average': movie_data['rating']
        }])
        x_tab = preprocessors['tab'].transform(input_tab).astype(np.float32)
        
        # Preprocess text
        x_text = preprocessors['text'].transform([movie_data['synopsis']]).toarray().astype(np.float32)
        
        with torch.no_grad():
            t_tensor = torch.tensor(x_tab)
            txt_tensor = torch.tensor(x_text)
            pred = model(t_tensor, txt_tensor).item()

        formatted_revenue = f"${pred:,.2f}"
        print(f"\nPrediction Successful!")
        print(f"Predicted Box Office Revenue: {formatted_revenue}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    run_test()
