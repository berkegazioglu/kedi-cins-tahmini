import React, { useState, useRef } from 'react';
import './ImageUploader.css';

const SAMPLE_THUMBS = [
  'https://whatbreedismycat.app/cat/persian.jpg',
  'https://whatbreedismycat.app/cat/maine-coon.jpg',
  'https://whatbreedismycat.app/cat/british-shorthair.jpg',
  'https://whatbreedismycat.app/cat/siamese.jpg',
];

const ImageUploader = ({ onImageSelect, onPredict, isLoading, preview, results, error, onReset }) => {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onloadend = () => onImageSelect(file, reader.result);
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDrag(e.type === 'dragenter' || e.type === 'dragover');
  };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setDrag(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };
  const handleChange = (e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); };

  const handleSample = async (src) => {
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const file = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
      handleFile(file);
    } catch { /* ignore network errors */ }
  };

  /* ─── STATE: RESULT ─── */
  if (results) {
    const top = results.predictions?.[0];
    const others = results.predictions?.slice(1, 3) || [];
    return (
      <div className="uc-card uc-result">
        <div className="uc-result-inner">
          <img src={preview} alt="kedi" className="uc-result-img" />
          <div className="uc-result-info">
            <p className="uc-lbl">Cat Breed Match:</p>
            <div className="uc-top-breed">
              <strong>{top?.class_name}</strong>
              <span className="uc-conf">{top ? (top.confidence * 100).toFixed(1) + '% confident' : ''}</span>
            </div>
            {others.length > 0 && (
              <>
                <p className="uc-lbl" style={{marginTop:'0.85rem'}}>Other Breed Match:</p>
                {others.map((o, i) => (
                  <div key={i} className="uc-other-breed">
                    {o.class_name}
                    <span className="uc-conf">{(o.confidence * 100).toFixed(1)}% confident</span>
                  </div>
                ))}
              </>
            )}
            <button className="uc-reset" onClick={onReset}>↩ Yeni fotoğraf dene</button>
          </div>
        </div>
      </div>
    );
  }

  /* ─── STATE: PREVIEW ─── */
  if (preview) {
    return (
      <div className="uc-card">
        <img src={preview} alt="önizleme" className="uc-preview-img" />
        <button
          className="btn-pink uc-analyze-btn"
          onClick={() => onPredict()}
          disabled={isLoading}
        >
          {isLoading
            ? <><span className="uc-spinner" /> Analiz ediliyor...</>
            : '🔍 Cinsi Analiz Et'}
        </button>
        {error && <p className="uc-error">⚠️ {error}</p>}
        <button className="uc-reset" onClick={onReset}>✕ Yeni fotoğraf seç</button>
      </div>
    );
  }

  /* ─── STATE: UPLOAD ─── */
  return (
    <div className="uc-card">
      <div
        className={`uc-drop ${drag ? 'drag-over' : ''}`}
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag} onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} />
        <p className="uc-drop-text">Yüklemek için buraya fotoğraf tıklayın veya sürükleyin.</p>
        <p className="uc-drop-fmt">JPG, JPEG, PNG, WEBP &nbsp;·&nbsp; 10 MB&apos;tan Az</p>
        <button
          className="btn-pink uc-upload-btn"
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
        >
          Kedi Fotoğrafı Yükle ⬆
        </button>
      </div>
      <div className="uc-samples">
        <p className="uc-samples-lbl">Fotoğrafınız yok mu? Şunlardan birini deneyin:</p>
        <div className="uc-thumbs">
          {SAMPLE_THUMBS.map((src, i) => (
            <img key={i} src={src} alt={`örnek ${i+1}`} className="uc-thumb" onClick={() => handleSample(src)} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;
