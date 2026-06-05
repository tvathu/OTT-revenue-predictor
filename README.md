<h1 align="center">🎬 Box Office Oracle AI</h1>

<div align="center">
  <strong>Predicting movie revenues using Advanced Multi-Modal Deep Learning and Explainable AI (XAI).</strong>
</div>
<br />

The **Box Office Oracle AI** is an enterprise-grade web application that leverages Deep Learning to forecast the worldwide box office revenue of a film before it even begins production. By fusing structured financial data with unstructured text narratives, it provides studios and creators with actionable, AI-driven financial insights.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![React](https://img.shields.io/badge/React-UI-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)

---

## 🚀 What It Does

Movie revenues are traditionally hard to predict because success isn't just about the budget—it's about the **story**. 
This platform solves that by analyzing:
1. **Financial Parameters:** Production Budget, Expected Runtime, and Target Audience Rating.
2. **Narrative Appeal:** The actual unstructured synopsis/pitch of the movie.

The system processes this data through a custom neural network and outputs a highly accurate revenue prediction. More importantly, it doesn't just give you a black-box number—it explains *exactly why* it chose that number using Explainable AI (XAI).

## 🧠 How It Works (Architecture)

The system is split into a robust Python backend and a beautiful React frontend, connected via a RESTful API.

### 1. Multi-Modal Deep Learning (`torch.nn.Module`)
The core intelligence is a PyTorch Multi-Layer Perceptron (MLP) built to handle multiple modalities of data:
- **Tabular Data:** Uses `Polars` for lightning-fast memory-efficient processing of historical numerical data (budgets, runtimes).
- **Text Data:** Uses `TfidfVectorizer` to convert unstructured movie synopses into dense mathematical semantic embeddings.
- **Fusion Layer:** The neural network takes both the numerical weights and the text embeddings, concatenates them, and passes them through deep linear layers to map relationships between the storyline and financial success.

### 2. AI Explainability (SHAP & LIME)
Nobody trusts a black-box AI. We integrated advanced explainability tools directly into the inference pipeline:
- **SHAP (SHapley Additive exPlanations):** Calculates the global impact of features (e.g., proving that higher budgets generally drive higher baseline revenues across the industry).
- **LIME (Local Interpretable Model-agnostic Explanations):** Explains the local prediction. It breaks down the math for *your specific movie*, showing exactly which features dragged the revenue down or pushed it up.

### 3. The UI (React & Vite)
A sleek, glassmorphic UI built with React. It dynamically expands upon receiving a prediction to reveal the PyTorch output alongside live interactive charts for LIME and SHAP, providing a stunning analytical dashboard.

---

## 🛠️ Tech Stack Highlights
* **Backend Framework:** FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Scikit-Learn, Polars
* **XAI:** SHAP, LIME
* **Frontend:** React.js, Vite, Vanilla CSS (Glassmorphism design)
* **Orchestration:** Docker, Docker-compose, Nginx Reverse Proxy

---

## 🐳 Quick Start (Docker Deployment)

This application is fully containerized and production-ready. You don't need Python or Node installed—just Docker!

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/box-office-oracle.git
cd box-office-oracle

# 2. Build and run the containers
docker-compose up --build -d
```

* **Frontend UI:** Navigate to `http://localhost:5173`
* **Backend API:** Navigate to `http://localhost:8000/docs` to see the automated Swagger API documentation.

## 💻 Local Development
If you wish to develop without Docker:

**Backend (Python):**
```bash
cd backend
pip install -r requirements.txt
python ml/train.py  # Train the initial Deep Learning model
uvicorn app.main:app --reload
```

**Frontend (React):**
```bash
cd frontend
npm install
npm run dev
```
*(Vite is pre-configured to proxy `/api` traffic to `localhost:8000` automatically).*
