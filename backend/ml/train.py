import os
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

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

def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset.csv')
    model_path = os.path.join(base_dir, 'model.pth')
    preprocessor_path = os.path.join(base_dir, 'preprocessor.pkl')
    shap_plot_path = os.path.join(base_dir, 'shap_summary.png')
    lime_plot_path = os.path.join(base_dir, 'lime_explanation.html')

    print("Loading dataset using Polars...")
    df = pl.read_csv(dataset_path).drop_nulls(subset=['revenue'])
    df_pd = df.to_pandas()
    
    features_tabular = ['budget', 'runtime', 'vote_average']
    X_tab = df_pd[features_tabular]
    X_text = df_pd['synopsis'].fillna('')
    y = df_pd['revenue'].values.astype(np.float32)

    tab_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    text_pipeline = TfidfVectorizer(max_features=50)

    print("Preprocessing structured and unstructured data...")
    X_tab_proc = tab_pipeline.fit_transform(X_tab).astype(np.float32)
    X_text_proc = text_pipeline.fit_transform(X_text).toarray().astype(np.float32)

    tab_tensor = torch.tensor(X_tab_proc)
    text_tensor = torch.tensor(X_text_proc)
    y_tensor = torch.tensor(y).view(-1, 1)

    model = OTTRevenueModel(tabular_dim=X_tab_proc.shape[1], text_dim=X_text_proc.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    print("Training PyTorch model...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(tab_tensor, text_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    joblib.dump({
        'tab': tab_pipeline, 
        'text': text_pipeline,
        'tab_dim': X_tab_proc.shape[1],
        'text_dim': X_text_proc.shape[1]
    }, preprocessor_path)
    
    print(f"Model and preprocessor saved to {base_dir}")

    print("Generating SHAP and LIME visualisations...")
    def predict_fn(x_tab_arr):
        with torch.no_grad():
            t_tensor = torch.tensor(x_tab_arr, dtype=torch.float32)
            dummy_text = text_tensor.mean(dim=0, keepdim=True).repeat(x_tab_arr.shape[0], 1)
            preds = model(t_tensor, dummy_text).numpy().flatten()
        return preds

    # SHAP
    explainer = shap.Explainer(predict_fn, X_tab_proc)
    shap_values = explainer(X_tab_proc)
    plt.figure()
    shap.summary_plot(shap_values, X_tab_proc, feature_names=features_tabular, show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    # LIME
    lime_explainer = LimeTabularExplainer(X_tab_proc, feature_names=features_tabular, mode='regression')
    exp = lime_explainer.explain_instance(X_tab_proc[0], predict_fn, num_features=3)
    exp.save_to_file(lime_plot_path)
    print("Explainability plots generated.")

if __name__ == "__main__":
    train_model()
