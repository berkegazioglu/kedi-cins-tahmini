import React, { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import ApiService from './services/api';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [skipDetection, setSkipDetection] = useState(false);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [compareMode, setCompareMode] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await ApiService.getModels();
        setAvailableModels(data.models || []);
        if (data.default_model) {
          setSelectedModel(data.default_model);
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
      }
    };
    fetchModels();
  }, []);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Lütfen bir görüntü seçin');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      let data;
      if (compareMode) {
        // Predict with all models
        data = await ApiService.predictAllModels(selectedFile, skipDetection, 5);
      } else {
        // Predict with selected model
        data = await ApiService.predictBreed(selectedFile, selectedModel, skipDetection, 5);
      }
      
      setResults(data);
    } catch (err) {
      setError(err.message || 'Tahmin sırasında bir hata oluştu');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🐱 Kedi Cinsi Tahmin Sistemi</h1>
        <p className="subtitle">Kedilerinizin cinsini yapay zeka ile belirleyin</p>
      </header>

      <main className="app-main">
        <div className="controls-panel">
          <div className="control-group">
            <label className="control-label">
              <input
                type="checkbox"
                checked={skipDetection}
                onChange={(e) => setSkipDetection(e.target.checked)}
              />
              <span>Kedi Tespitini Atla (YOLO)</span>
            </label>
          </div>

          <div className="control-group">
            <label className="control-label">
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(e) => {
                  setCompareMode(e.target.checked);
                  setResults(null);
                }}
              />
              <span>Tüm Modelleri Karşılaştır</span>
            </label>
          </div>

          {!compareMode && availableModels.length > 0 && (
            <div className="control-group">
              <label className="control-label-block">Model Seçin:</label>
              <select
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  setResults(null);
                }}
                className="model-select"
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <ImageUploader
          onImageSelect={handleImageSelect}
          onPredict={handlePredict}
          isLoading={isLoading}
        />

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {results && (
          <PredictionResults results={results} compareMode={compareMode} />
        )}
      </main>

      <footer className="app-footer">
        <p>🧠 Powered by PyTorch & ResNet-50, EfficientNetB0, MobileNetV3, Ensemble</p>
        <p>📸 YOLO11n ile Kedi Tespiti | 🐈 59 Farklı Kedi Irkı</p>
      </footer>
    </div>
  );
}

export default App;
