import React from 'react';
import './PredictionResults.css';

const PredictionResults = ({ results, preview, onReset }) => {
  if (!results) return null;
  const { predictions = [], entropy = 0, detection, breed_info, wild_warning } = results;
  const top = predictions[0];
  const others = predictions.slice(1);
  const ENTROPY_WARN = 2.5;

  return (
    <div className="pr-wrap">
      <div className="pr-header">
        <h2>🎯 Tahmin Sonuçları</h2>
        <button className="pr-new-btn" onClick={onReset}>+ Yeni Fotoğraf</button>
      </div>

      {/* Uyarılar */}
      {wild_warning && (
        <div className="pr-alert pr-alert-danger">
          <strong>🦁 Vahşi Kedi Uyarısı</strong>
          <p>{wild_warning}</p>
        </div>
      )}
      {detection && !detection.skipped && (
        <div className={`pr-alert ${detection.cat_detected ? 'pr-alert-success' : 'pr-alert-warn'}`}>
          🔍 Kedi Tespiti: {detection.cat_detected ? '✅ Başarılı' : '❌ Başarısız'}
          {detection.confidence > 0 && ` (Güven: %${(detection.confidence*100).toFixed(1)})`}
        </div>
      )}
      {entropy > ENTROPY_WARN && (
        <div className="pr-alert pr-alert-warn">
          🔴 Yüksek belirsizlik (entropi: {entropy.toFixed(2)}) — Farklı bir fotoğraf deneyebilirsiniz.
        </div>
      )}

      <div className="pr-body">
        {/* Sol: Fotoğraf + Top Tahmin */}
        <div className="pr-left">
          {preview && <img src={preview} alt="kedi" className="pr-photo" />}
          {top && (
            <div className="pr-top-card">
              <p className="pr-match-lbl">En İyi Eşleşme</p>
              <p className="pr-breed-name">{top.breed}</p>
              <p className="pr-conf-val">{(top.confidence * 100).toFixed(1)}% eşleşme</p>
              <div className="pr-bar-wrap">
                <div className="pr-bar" style={{width: `${(top.confidence*100).toFixed(0)}%`}} />
              </div>
            </div>
          )}
        </div>

        {/* Sağ: Diğer Tahminler + Irk Bilgisi */}
        <div className="pr-right">
          {others.length > 0 && (
            <div className="pr-others">
              <h3>Diğer Olası Irklar</h3>
              {others.map((p, i) => (
                <div key={i} className="pr-other-row">
                  <div className="pr-other-info">
                    <span className="pr-other-name">{p.breed}</span>
                    <span className="pr-other-conf">{(p.confidence*100).toFixed(1)}%</span>
                  </div>
                  <div className="pr-bar-wrap">
                    <div className="pr-bar pr-bar-other" style={{width:`${(p.confidence*100).toFixed(0)}%`}} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {breed_info && (
            <div className="pr-breed-info">
              <h3>Irk Hakkında</h3>
              {breed_info.description && <p>{breed_info.description}</p>}
              <div className="pr-info-tags">
                {breed_info.temperament && (
                  <div className="pr-tag-row">
                    <span className="pr-tag-lbl">Karakter</span>
                    <span>{breed_info.temperament}</span>
                  </div>
                )}
                {breed_info.origin && (
                  <div className="pr-tag-row">
                    <span className="pr-tag-lbl">Köken</span>
                    <span>{breed_info.origin}</span>
                  </div>
                )}
                {breed_info.life_span && (
                  <div className="pr-tag-row">
                    <span className="pr-tag-lbl">Yaşam</span>
                    <span>{breed_info.life_span} yıl</span>
                  </div>
                )}
                {breed_info.weight && (
                  <div className="pr-tag-row">
                    <span className="pr-tag-lbl">Ağırlık</span>
                    <span>{breed_info.weight} kg</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
