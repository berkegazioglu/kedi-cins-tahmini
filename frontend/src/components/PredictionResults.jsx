import React, { useState } from 'react';
import './PredictionResults.css';

const PredictionResults = ({ results, compareMode = false }) => {
  const [activeTab, setActiveTab] = useState('general');

  if (!results) return null;

  // Single model mode
  if (!compareMode && results.success) {
    const { predictions, is_wild_cat, entropy, top_breed, breed_info, model_used } = results;

    return (
      <div className="prediction-results">
        {is_wild_cat && (
          <div className="wild-warning">
            ⚠️ <strong>Dikkat:</strong> Bu görüntü muhtemelen vahşi bir kedi türüne ait (Entropi: {entropy.toFixed(2)})
          </div>
        )}

        <div className="model-info">
          <span className="model-badge">{model_used || 'resnet50'}</span>
        </div>

        <div className="top-prediction">
          <h3>En Olası Cins</h3>
          <div className="breed-name">{top_breed}</div>
          <div className="confidence">{(predictions[0].confidence * 100).toFixed(1)}% güven</div>
          {!is_wild_cat && entropy && (
            <div className="entropy-info">Entropi: {entropy.toFixed(2)}</div>
          )}
        </div>

        <div className="other-predictions">
          <h4>Diğer Olasılıklar</h4>
          {predictions.slice(1).map((pred, idx) => (
            <div key={idx} className="prediction-item">
              <span className="breed-label">{pred.breed}</span>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{width: `${pred.confidence * 100}%`}}
                ></div>
              </div>
              <span className="confidence-value">{(pred.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>

        {breed_info && (
          <div className="breed-info-section">
            <div className="tabs">
              <button 
                className={activeTab === 'general' ? 'active' : ''} 
                onClick={() => setActiveTab('general')}
              >
                📋 Genel Bilgi
              </button>
              <button 
                className={activeTab === 'health' ? 'active' : ''} 
                onClick={() => setActiveTab('health')}
              >
                🏥 Sağlık
              </button>
              <button 
                className={activeTab === 'nutrition' ? 'active' : ''} 
                onClick={() => setActiveTab('nutrition')}
              >
                🍽️ Beslenme
              </button>
              <button 
                className={activeTab === 'care' ? 'active' : ''} 
                onClick={() => setActiveTab('care')}
              >
                ✨ Bakım
              </button>
              <button 
                className={activeTab === 'character' ? 'active' : ''} 
                onClick={() => setActiveTab('character')}
              >
                😺 Karakter
              </button>
            </div>

            <div className="tab-content">
              {activeTab === 'general' && (
                <div className="info-card">
                  <p><strong>Köken:</strong> {breed_info.origin}</p>
                  <p><strong>Boyut:</strong> {breed_info.size}</p>
                  <p><strong>Ağırlık:</strong> {breed_info.weight}</p>
                  <p><strong>Ömür:</strong> {breed_info.lifespan}</p>
                  <p><strong>Aktivite:</strong> {breed_info.activity_level}</p>
                  <p><strong>Tüy Dökme:</strong> {breed_info.shedding}</p>
                  <p><strong>Çocuk Dostu:</strong> {breed_info.child_friendly}</p>
                  <p><strong>Evcil Hayvan Dostu:</strong> {breed_info.pet_friendly}</p>
                </div>
              )}

              {activeTab === 'health' && (
                <div className="info-card">
                  <h4>Sağlık Bilgileri</h4>
                  <ul>
                    {breed_info.health.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {activeTab === 'nutrition' && (
                <div className="info-card">
                  <h4>Beslenme Önerileri</h4>
                  <ul>
                    {breed_info.nutrition.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {activeTab === 'care' && (
                <div className="info-card">
                  <h4>Bakım İpuçları</h4>
                  <ul>
                    {breed_info.care.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                  <p><strong>Tımar Gereksinimi:</strong> {breed_info.grooming}</p>
                </div>
              )}

              {activeTab === 'character' && (
                <div className="info-card">
                  <h4>Karakter Özellikleri</h4>
                  <ul>
                    {breed_info.character.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Compare mode - all models
  if (compareMode && results.success) {
    const { predictions, consensus, timing, models_used } = results;

    return (
      <div className="prediction-results compare-mode">
        <h2>Model Karşılaştırması</h2>
        
        {consensus && (
          <div className="consensus-section">
            <h3>🎯 Konsensüs (En Yaygın Tahmin)</h3>
            <div className="breed-name">{consensus}</div>
          </div>
        )}

        <div className="models-comparison">
          {models_used.map(modelName => {
            const modelPred = predictions[modelName];
            
            if (modelPred.error) {
              return (
                <div key={modelName} className="model-result error">
                  <h4>{modelName}</h4>
                  <p className="error-message">❌ {modelPred.error}</p>
                </div>
              );
            }

            return (
              <div key={modelName} className="model-result">
                <div className="model-header">
                  <h4>{modelName}</h4>
                  <span className="timing">⏱️ {modelPred.inference_time}s</span>
                </div>

                {modelPred.is_wild_cat && (
                  <div className="wild-warning-small">
                    ⚠️ Vahşi Kedi (Entropi: {modelPred.entropy.toFixed(2)})
                  </div>
                )}

                <div className="top-prediction-small">
                  <div className="breed-name-small">{modelPred.top_breed}</div>
                  <div className="confidence-small">
                    {(modelPred.top_confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="other-predictions-small">
                  {modelPred.predictions.slice(1, 4).map((pred, idx) => (
                    <div key={idx} className="prediction-item-small">
                      <span className="breed-label-small">{pred.breed}</span>
                      <span className="confidence-value-small">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        <div className="timing-summary">
          <h4>⏱️ Çalışma Süreleri</h4>
          {Object.entries(timing).map(([model, time]) => (
            <div key={model} className="timing-item">
              <span>{model}</span>
              <span>{time}s</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Error state
  return (
    <div className="prediction-results error">
      <p>❌ {results.error || 'Bir hata oluştu'}</p>
    </div>
  );
};

export default PredictionResults;
