import React, { useState } from 'react';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    budget: '',
    runtime: '',
    rating: '',
    synopsis: ''
  });
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showExplainers, setShowExplainers] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);
    setShowExplainers(false);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          budget: parseFloat(formData.budget),
          runtime: parseFloat(formData.runtime),
          rating: parseFloat(formData.rating),
          synopsis: formData.synopsis
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to fetch prediction');
      }

      const data = await response.json();
      setResult(data.predicted_revenue);
      setExplanation(data.explanation);
      setShowExplainers(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <div className={`layout-container ${showExplainers ? 'expanded' : ''}`}>
      <div className="card form-card">
        <h2 style={{marginTop: 0, marginBottom: '1.5rem', color: '#fff', fontSize: '1.5rem'}}>Enter Movie Parameters</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="synopsis">Movie Synopsis (Unstructured Text)</label>
            <textarea
              id="synopsis"
              name="synopsis"
              className="form-input"
              rows="3"
              value={formData.synopsis}
              onChange={handleChange}
              placeholder="e.g. A gripping sci-fi adventure across the galaxy..."
              required
            ></textarea>
          </div>
          <div className="form-group">
            <label htmlFor="budget">Production Budget ($)</label>
            <input
              type="number"
              id="budget"
              name="budget"
              className="form-input"
              value={formData.budget}
              onChange={handleChange}
              placeholder="e.g. 150000000"
              required
              min="1"
            />
          </div>
          <div className="form-row">
            <div className="form-group half">
              <label htmlFor="runtime">Runtime (Mins)</label>
              <input
                type="number"
                id="runtime"
                name="runtime"
                className="form-input"
                value={formData.runtime}
                onChange={handleChange}
                placeholder="120"
                required
                min="1"
              />
            </div>
            <div className="form-group half">
              <label htmlFor="rating">Rating (0-10)</label>
              <input
                type="number"
                id="rating"
                name="rating"
                className="form-input"
                value={formData.rating}
                onChange={handleChange}
                placeholder="8.5"
                required
                min="0"
                max="10"
                step="0.1"
              />
            </div>
          </div>
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Running AI Model...' : 'Predict Revenue'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        {result !== null && (
          <div className="result-container">
            <p style={{ margin: 0, color: '#cbd5e0' }}>Deep Learning Prediction</p>
            <p className="result-value">{formatCurrency(result)}</p>
            {explanation && (
              <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <p style={{ margin: '0 0 0.5rem 0', color: '#4facfe', fontWeight: 600 }}>AI Interpretation:</p>
                <p style={{ margin: 0, color: '#a0aec0', fontSize: '0.95rem', lineHeight: '1.5' }}>
                  {explanation}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {showExplainers && (
        <div className="card explainers-card">
          <h2 style={{marginTop: 0, marginBottom: '1.5rem', color: '#fff', fontSize: '1.5rem'}}>AI Explainability</h2>
          <div className="explainer-section">
            <h3>SHAP Feature Importance</h3>
            <p className="explainer-desc">Shows the global impact of features on the model's predictions.</p>
            <div className="img-container">
              <img 
                src={`/api/explain/shap?t=${new Date().getTime()}`} 
                alt="SHAP Summary Plot" 
                className="shap-img" 
              />
            </div>
          </div>
          <div className="explainer-section">
            <h3>LIME Local Explanation</h3>
            <p className="explainer-desc">Breaks down exactly why the model made its decision for a local instance.</p>
            <iframe 
              src={`/api/explain/lime?t=${new Date().getTime()}`} 
              className="lime-frame"
              title="LIME Explanation"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
