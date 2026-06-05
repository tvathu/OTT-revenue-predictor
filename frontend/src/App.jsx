import React from 'react';
import PredictionForm from './components/PredictionForm';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <header>
        <h1 className="hero-title">OTT REVENUE PREDICTION SYSTEM</h1>
        <p className="hero-subtitle">
          Advanced Multi-Modal Deep Learning Revenue Prediction & Explainability System for the Film Industry 
        </p>
      </header>
      <main className="main-content">
        <PredictionForm />
      </main>
    </div>
  );
}

export default App;
