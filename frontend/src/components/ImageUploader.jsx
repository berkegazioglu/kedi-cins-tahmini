import React, { useState } from 'react';
import './ImageUploader.css';

const ImageUploader = ({ onImageSelect, onPredict, isLoading }) => {
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Lütfen geçerli bir görüntü dosyası seçin');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);

    // Pass to parent
    onImageSelect(file);
  };

  return (
    <div className="image-uploader">
      <div 
        className={`upload-area ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <button 
              onClick={() => {
                setPreview(null);
                onImageSelect(null);
              }}
              className="remove-button"
              disabled={isLoading}
            >
              ✕ Kaldır
            </button>
          </div>
        ) : (
          <div className="upload-prompt">
            <div className="upload-icon">📸</div>
            <p>Bir kedi fotoğrafı sürükleyip bırakın</p>
            <p className="upload-hint">veya</p>
            <label htmlFor="file-upload" className="file-upload-label">
              Dosya Seç
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleChange}
              style={{ display: 'none' }}
            />
          </div>
        )}
      </div>

      {preview && (
        <button 
          onClick={onPredict} 
          className="predict-button"
          disabled={isLoading}
        >
          {isLoading ? '⏳ Analiz Ediliyor...' : '🔍 Cinsi Tahmin Et'}
        </button>
      )}
    </div>
  );
};

export default ImageUploader;
