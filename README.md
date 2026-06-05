<h1 align="center">🎬 OTT Revenue prediction system using Deep learning and Explainable AI</h1>

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

---

## 🛠️ How It Was Made In Detail (The Architecture)

This project was built from scratch using a modern, scalable full-stack architecture. Here is exactly how the data flows from the user's browser down to the neural network weights.

### 1. Data Processing & Machine Learning (The Python Backend)
The core intelligence is a PyTorch Multi-Layer Perceptron (MLP) built to handle multiple modalities of data simultaneously:
- **Numerical Processing (Polars & Scikit-Learn):** Historical datasets containing budgets, runtimes, and ratings are ingested using `polars` for lightning-fast memory efficiency. The numbers are normalized using a Scikit-Learn `StandardScaler` so that massive budgets ($150M) and small ratings (8.0) can be mathematically compared on the same scale.
- **Textual Processing (NLP):** We use a `TfidfVectorizer` to convert unstructured English paragraphs (the movie's synopsis) into dense mathematical semantic embeddings. This allows the neural network to understand narrative keywords like "galaxy" or "romance."
- **PyTorch Fusion Layer:** The `torch.nn.Module` takes both the numerical matrix and the text embeddings, passes them through independent linear layers, and then concatenates them together. This combined tensor is passed through hidden layers activated by ReLU functions to map non-linear relationships between the storyline and financial success.

### 2. AI Explainability (SHAP & LIME)
Nobody trusts a black-box AI. We integrated advanced Explainable AI directly into the training and inference pipeline:
- **SHAP (SHapley Additive exPlanations):** During training, SHAP analyzes the global model to calculate the overall impact of features. It generates visual plots proving, for example, that higher budgets generally drive higher baseline revenues across the entire film industry.
- **LIME (Local Interpretable Model-agnostic Explanations):** When a user makes a *specific* prediction, LIME breaks down the math locally. It generates an interactive HTML frame showing exactly which input features dragged the revenue down or pushed it up for that specific movie.
- **Textual Heuristic Interpreter:** A custom Python algorithm analyzes the prediction inputs alongside the XAI metrics to dynamically generate a human-readable text paragraph explaining the model's logic.

### 3. The API (FastAPI)
The backend is served using **FastAPI** and **Uvicorn**. FastAPI creates a blazingly fast REST endpoint (`/predict`) that takes the user's JSON payload, dynamically runs the data through the trained PyTorch `model.pth`, queries the XAI modules, and returns the predicted revenue and textual interpretation in milliseconds.

### 4. The UI (React.js & Vite)
The frontend is a sleek, component-based Single Page Application (SPA) built with React and Vite. 
- It uses a custom **Glassmorphism** design system with vanilla CSS, featuring deep gradient backgrounds and responsive grid layouts.
- **Dynamic Layout Shifting:** The interface dynamically expands from a centered form into a multi-column dashboard upon receiving a successful prediction, rendering the PyTorch results alongside the interactive LIME and SHAP visual assets fetched directly from the API.

### 5. Orchestration & DevOps (Docker)
The entire stack is containerized for enterprise deployment:
- **`backend/Dockerfile`**: Automates the installation of complex C++ data science libraries, trains the PyTorch model upon build (so the weights are fresh), and serves the API.
- **`frontend/Dockerfile`**: A multi-stage build that compiles the React app using Node.js, and serves the static production assets using a highly optimized Nginx reverse proxy.
- **`docker-compose.yml`**: Networks the frontend and backend together flawlessly.

---

## 🐳 Quick Start (Docker Deployment)

This application is fully containerized and production-ready. You don't need Python or Node installed—just Docker!

```bash
# 1. Clone the repository
git clone https://github.com/tvathu/OTTO.git
cd OTTO

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
