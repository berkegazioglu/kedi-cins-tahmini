import { useState } from 'react'
import './App-new.css'
import { API_BASE_URL } from './config'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [catDetection, setCatDetection] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [entropy, setEntropy] = useState(null)
  const [isWildCat, setIsWildCat] = useState(false)
  const [wildCatInfo, setWildCatInfo] = useState(null)
  const [showAbout, setShowAbout] = useState(false)
  const [showHow, setShowHow] = useState(false)
  const [showPrivacy, setShowPrivacy] = useState(false)
  const [breedInfo, setBreedInfo] = useState(null)
  const [loadingBreedInfo, setLoadingBreedInfo] = useState(false)

  const processFile = (file) => {
    if (file) {
      setSelectedFile(file)
      setPredictions(null)
      setError(null)
      setCatDetection(null)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.onerror = () => {
        setError('Dosya okunamadı. Lütfen geçerli bir görsel dosyası seçin.')
      }
      reader.readAsDataURL(file)
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      processFile(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      processFile(files[0])
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Lütfen bir fotoğraf seçin')
      return
    }

    setLoading(true)
    setError(null)
    setPredictions(null)
    setCatDetection(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)
      formData.append('skip_detection', 'false')
      formData.append('uncertainty_threshold', '0.9')

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Tahmin yapılamadı')
      }

      setPredictions(data.predictions)
      setCatDetection(data.cat_detection)
      setEntropy(data.entropy)
      setIsWildCat(data.is_wild_cat)
      setWildCatInfo(data.wild_cat_info)

      // Fetch breed info if not wild cat
      if (!data.is_wild_cat && data.predictions && data.predictions.length > 0) {
        fetchBreedInfo(data.predictions[0].breed)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchBreedInfo = async (breedName) => {
    setLoadingBreedInfo(true)
    try {
      const response = await fetch(`${API_BASE_URL}/breed-info`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ breed: breedName })
      })
      const data = await response.json()
      if (response.ok && data.success) {
        setBreedInfo(data.info)
      }
    } catch (err) {
      console.log('Breed info error:', err)
    } finally {
      setLoadingBreedInfo(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setPredictions(null)
    setError(null)
    setCatDetection(null)
    setEntropy(null)
    setIsWildCat(false)
    setWildCatInfo(null)
    setBreedInfo(null)
  }

  return (
    <div className="patipedia-app">
      {/* Header */}
      <header className="patipedia-header">
        <div className="logo">
          <span className="logo-icon">🐾</span>
          <span className="logo-text">PatiPedia</span>
        </div>
      </header>

      {/* Main Container */}
      <div className="patipedia-container">
        <h1 className="main-title">Evcil Kedi Türü Tanıma Sistemi</h1>

        {/* Upload Area */}
        {!predictions && !isWildCat && !isUncertain && (
          <div
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {/* Added Threshold Slider */}
            <div className="threshold-container">
              <label className="threshold-label" title="Modelin bir tahmini kabul etmesi için gereken minimum güven oranı">
                Güvenilirlik Eşiği: <strong>%{confidenceThreshold}</strong>
              </label>
              <input
                type="range"
                min="10"
                max="50"
                value={confidenceThreshold}
                onChange={handleThresholdChange}
                className="threshold-slider"
              />
            </div>

            {!preview ? (
              <div className="upload-content">
                <div className="upload-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M7 18a4.6 4.4 0 0 1 0 -9a5 4.5 0 0 1 11 2h1a3.5 3.5 0 0 1 0 7h-1" />
                    <polyline points="9 15 12 12 15 15" />
                    <line x1="12" y1="12" x2="12" y2="21" />
                  </svg>
                </div>
                {loading && (
                  <div className="loading-circle">
                    <div className="spinner"></div>
                    <span className="loading-text">ANALİZ EDİLİYOR...</span>
                  </div>
                )}
                <p className="upload-text">Yüklemek için fotoğrafı buraya bırakabilirsiniz.</p>
                <label className="upload-button">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                  <span className="button-icon">📸</span>
                  <span>FOTOĞRAF YÜKLE</span>
                </label>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <button className="remove-button" onClick={handleReset}>×</button>
                {!predictions && !loading && (
                  <button className="analyze-button" onClick={handlePredict}>
                    ANALİZ ET
                  </button>
                )}
                {loading && (
                  <div className="analyzing-overlay">
                    <div className="spinner-large"></div>
                    <p>Analiz ediliyor...</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <span>⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {/* Wild Cat OR Uncertain Alert */}
        {(isWildCat || isUncertain) && (
          <div className="wild-cat-alert">
            <h2>{isWildCat ? '🦁 Vahşi Kedi Tespit Edildi!' : '❌ Belirsiz / Veri Seti Dışı'}</h2>

            <div className="wild-cat-gif">
              {isWildCat ? (
                <img
                  src="/wild-cat.gif"
                  alt="Vahşi Kedi"
                  onError={(e) => {
                    e.target.style.display = 'none'
                    e.target.parentElement.innerHTML = '<div class="wild-cat-emoji">🦁🐆🐅</div>'
                  }}
                />
              ) : (
                <div className="wild-cat-emoji">❓</div>
              )}
            </div>

            <p className="wild-cat-text">
              {isWildCat ? (
                "Bu görseldeki hayvan vahşi bir kedi türüdür."
              ) : (
                <>
                  Modelin tahmini güvenilirlik eşiğinin (<strong>%{confidenceThreshold}</strong>) altında kaldı.
                  <br />
                  Bu muhtemelen <strong>veri setinde olmayan</strong> veya <strong>karışık/belirsiz</strong> bir tür.
                </>
              )}
              <br />
              Sistemimiz yalnızca <strong>59 ev kedisi ırkı</strong> için eğitilmiştir.
            </p>

            {/* Nearest Matches (Orange/Faded) */}
            {predictions && (
              <div className="uncertain-results">
                <h3>En Yakın Eşleşmeler (Güvenilir Değil):</h3>
                {predictions.slice(0, 3).map((pred, index) => (
                  <div key={index} className="uncertain-item">
                    <span className="alt-name">{pred.breed}</span>
                    <span className="alt-confidence">%{pred.confidence.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            )}

            <button className="reset-button" onClick={handleReset}>
              Yeni Fotoğraf Yükle
            </button>
          </div>
        )}

        {/* Results (Normal) */}
        {predictions && !isWildCat && !isUncertain && (
          <div className="results-container">
            <div className="result-card">
              {preview && (
                <div className="result-image">
                  <img src={preview} alt="Analyzed cat" />
                </div>
              )}
              <div className="result-content">
                <div className="result-label">SONUÇ:</div>
                <h2 className="result-breed">Kedi Türü: {predictions[0].breed}</h2>
                <p className="result-confidence">Kesinlik: %{predictions[0].confidence.toFixed(1)}</p>
                <p className="result-description">Başarılı, tanımlı ve sosyal bir ırkıdır</p>

                {predictions.length > 1 && (
                  <div className="alternative-results">
                    <h3>Diğer Olası Türler:</h3>
                    {predictions.slice(1, 3).map((pred, index) => (
                      <div key={index} className="alternative-item">
                        <span className="alt-name">{pred.breed}</span>
                        <span className="alt-confidence">%{pred.confidence.toFixed(1)}</span>
                      </div>
                    ))}
                  </div>
                )}

                <button className="reset-button" onClick={handleReset}>
                  Yeni Analiz Yap
                </button>
              </div>
            </div>

            {/* Breed Wiki Information */}
            {breedInfo && (
              <div className="breed-wiki-section">
                <h2 className="wiki-title">📚 {predictions[0].breed} Hakkında</h2>
                <div className="wiki-content">
                  {breedInfo.split('\n\n').map((section, index) => {
                    const lines = section.split('\n')
                    const title = lines[0]
                    const content = lines.slice(1).join('\n')

                    return (
                      <div key={index} className="wiki-section">
                        <h3 className="wiki-section-title">{title}</h3>
                        {content.split('\n').map((line, i) => (
                          line.trim() && <p key={i} className="wiki-text">{line}</p>
                        ))}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {loadingBreedInfo && (
              <div className="breed-wiki-loading">
                <div className="spinner"></div>
                <p>Irk bilgileri yükleniyor...</p>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <footer className="patipedia-footer">
          <button onClick={() => setShowAbout(true)}>Hakkında</button>
          <button onClick={() => setShowHow(true)}>Nasıl Çalışır?</button>
          <button onClick={() => setShowPrivacy(true)}>Gizlilik</button>
        </footer>
      </div>

      {/* Modals */}
      {showAbout && (
        <div className="modal-overlay" onClick={() => setShowAbout(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowAbout(false)}>×</button>
            <h2>Hakkında</h2>
            <p>PatiPedia, son teknoloji derin öğrenme algoritmaları ile geliştirilen, kedi cinsi tanıma ve bilgi platformudur. Sistem, iki aşamalı yapay zeka mimarisi kullanarak 59 farklı kedi ırkını yüksek doğrulukla tanıyabilir.</p>
          </div>
        </div>
      )}

      {showHow && (
        <div className="modal-overlay" onClick={() => setShowHow(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowHow(false)}>×</button>
            <h2>Nasıl Çalışır?</h2>
            <ol>
              <li>Kedi fotoğrafınızı yükleyin</li>
              <li>YOLO modeli fotoğrafta kedi olup olmadığını kontrol eder</li>
              <li>ResNet50 modeli kedi cinsini tahmin eder</li>
              <li>Entropi analizi ile vahşi kediler tespit edilir</li>
            </ol>
          </div>
        </div>
      )}

      {showPrivacy && (
        <div className="modal-overlay" onClick={() => setShowPrivacy(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowPrivacy(false)}>×</button>
            <h2>Gizlilik</h2>
            <p>Yüklediğiniz fotoğraflar yalnızca analiz için kullanılır ve saklanmaz. Tüm işlemler yerel sunucuda gerçekleştirilir ve verileriniz üçüncü taraflarla paylaşılmaz.</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
