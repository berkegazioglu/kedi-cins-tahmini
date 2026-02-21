import React, { useEffect, useRef, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import './ResultPage.css';

/* ── Breed adından Unsplash URL eşleştirmesi (runtime'da browser yükler) ── */
const BREED_IMG_MAP = {
  'Persian':                  'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=300&fit=crop',
  'Maine Coon':               'https://images.unsplash.com/photo-1594900060009-89cda7e5a3cd?w=400&h=300&fit=crop',
  'British Shorthair':        'https://images.unsplash.com/photo-1533743983669-94fa5c4338ec?w=400&h=300&fit=crop',
  'Siamese':                  'https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=400&h=300&fit=crop',
  'Siyam':                    'https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=400&h=300&fit=crop',
  'Abyssinian':               'https://images.unsplash.com/photo-1550159930-40066082a4fc?w=400&h=300&fit=crop',
  'Habeş Kedisi':             'https://images.unsplash.com/photo-1550159930-40066082a4fc?w=400&h=300&fit=crop',
  'Bengal':                   'https://images.unsplash.com/photo-1601979031925-424e53b6caaa?w=400&h=300&fit=crop',
  'Russian Blue':             'https://images.unsplash.com/photo-1519479651571-a2c9e714fee2?w=400&h=300&fit=crop',
  'Birman':                   'https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=400&h=300&fit=crop',
  'American Shorthair':       'https://images.unsplash.com/photo-1568043432885-75710c77fd61?w=400&h=300&fit=crop',
  'Bombay':                   'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400&h=300&fit=crop',
  'Norwegian Forest Cat':     'https://images.unsplash.com/photo-1548247416-ec66f4900b2e?w=400&h=300&fit=crop',
  'Sphynx':                   'https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=400&h=300&fit=crop',
  'Sfenks':                   'https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=400&h=300&fit=crop',
  'Scottish Fold':            'https://images.unsplash.com/photo-1571566882372-1598d88abd90?w=400&h=300&fit=crop',
  'İskoç Fold':               'https://images.unsplash.com/photo-1571566882372-1598d88abd90?w=400&h=300&fit=crop',
  'Chartreux':                'https://images.unsplash.com/photo-1561948955-570b270e7c36?w=400&h=300&fit=crop',
  'Turkish Angora':           'https://images.unsplash.com/photo-1590418606746-018840f9ced2?w=400&h=300&fit=crop',
  'Ankara Kedisi':            'https://images.unsplash.com/photo-1590418606746-018840f9ced2?w=400&h=300&fit=crop',
  'Ocicat':                   'https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=400&h=300&fit=crop',
  'Tonkinese':                'https://images.unsplash.com/photo-1535293478971-0a5e7a6c2230?w=400&h=300&fit=crop',
  'Burmese':                  'https://images.unsplash.com/photo-1535295972055-1c762f4483e5?w=400&h=300&fit=crop',
  'Savannah':                 'https://images.unsplash.com/photo-1518791841217-8f162f1912da?w=400&h=300&fit=crop',
  'Munchkin':                 'https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=400&h=300&fit=crop',
  'Cornish Rex':              'https://images.unsplash.com/photo-1591825729269-caeb344f6df2?w=400&h=300&fit=crop',
  'Exotic Shorthair':         'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=300&fit=crop',
  'Egzotik Shorthair':        'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=300&fit=crop',
  'Ragdoll':                  'https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=400&h=300&fit=crop',
  'Manx':                     'https://images.unsplash.com/photo-1609220136736-443140cfeaa8?w=400&h=300&fit=crop',
};

function breedImg(breedName) {
  if (!breedName) return null;
  return BREED_IMG_MAP[breedName] || null;
}

/* ── Breed adından cat_breed_info.json key eşleştirmesi ── */
const BREED_KEY_MAP = {
  'İskoç Fold':  'Scottish Fold',
  'Siyam':       'Siamese',
  'Sfenks':      'Sphynx',
  'Habeş Kedisi': 'Abyssinian',
  'Ankara Kedisi': 'Turkish Angora',
  'Egzotik Shorthair': 'Exotic Shorthair',
};

function breedKey(name) {
  return BREED_KEY_MAP[name] || name;
}

export default function ResultPage() {
  const location  = useLocation();
  const navigate  = useNavigate();
  const state     = location.state;
  const resultCardRef = useRef(null);

  const [breedDb, setBreedDb]     = useState(null);
  const [navScrolled, setNavScrolled] = useState(false);

  /* Back-guard: state yoksa ana sayfaya yönlendir */
  useEffect(() => {
    if (!state?.predictions?.length) {
      navigate('/', { replace: true });
    }
  }, [state, navigate]);

  /* Navbar scroll efekti */
  useEffect(() => {
    const fn = () => setNavScrolled(window.scrollY > 20);
    window.addEventListener('scroll', fn);
    return () => window.removeEventListener('scroll', fn);
  }, []);

  /* Breed bilgi JSON'u yükle */
  useEffect(() => {
    fetch('/cat_breed_info.json')
      .then(r => r.json())
      .then(setBreedDb)
      .catch(() => {});
  }, []);

  if (!state?.predictions?.length) return null;

  const { predictions = [], preview } = state;
  const top    = predictions[0];
  const others = predictions.slice(1, 3);

  const key     = breedKey(top?.breed);
  const info    = breedDb?.[key] || null;

  /* İndir: result kartını PNG olarak kaydet */
  const handleDownload = async () => {
    try {
      const html2canvas = (await import('https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.esm.js')).default;
      const canvas = await html2canvas(resultCardRef.current, { scale: 2, useCORS: true });
      const a = document.createElement('a');
      a.download = `kedi-irki-${(top?.breed || 'sonuc').replace(/\s+/g, '-').toLowerCase()}.png`;
      a.href = canvas.toDataURL('image/png');
      a.click();
    } catch {
      alert('İndir özelliği şu an kullanılamıyor.');
    }
  };

  const pct = (v) => `% ${(v * 100).toFixed(0)}`;

  return (
    <div className="rp-page">
      {/* ── NAVBAR ── */}
      <nav className={`navbar ${navScrolled ? 'scrolled' : ''}`}>
        <div className="nav-inner">
          <Link to="/" className="nav-logo">
            <img src="/kedi-ai-logo.svg" alt="Kedi.ai" className="nav-logo-img" /> Kedi.ai
          </Link>
          <ul className="nav-links">
            <li><Link to="/">Anasayfa</Link></li>
            <li><Link to="/cat-translator">Kedi Çevirmeni</Link></li>
            <li><Link to="/#how-it-works">Nasıl Çalışır</Link></li>
            <li><Link to="/#breeds-table">Kedi Irkları Tablosu</Link></li>
            <li><Link to="/#faq">SSS</Link></li>
          </ul>
        </div>
      </nav>

      {/* ── BAŞLIK ── */}
      <div className="rp-header">
        <h1 className="rp-title">Kedi Irkı Sonuçları</h1>
        <p className="rp-disclaimer">
          Not: Bu yapay zeka sonuçları görsel özelliklere dayanmaktadır ve %100 doğru olmayabilir.
          Sadece <a href="#">eğlence</a> ve referans amaçlıdır.
        </p>
      </div>

      {/* ── SONUÇ KARTI ── */}
      <div className="rp-container" ref={resultCardRef}>
        <div className="rp-card">

          {/* Top ─ fotoğraf + tahminler */}
          <div className="rp-top-row">
            {/* Sol: Yüklenen fotoğraf */}
            <div className="rp-photo-col">
              {preview && (
                <img src={preview} alt="Yüklenen kedi" className="rp-photo" />
              )}
            </div>

            {/* Sağ: Tahmin sonuçları */}
            <div className="rp-matches-col">
              <p className="rp-match-section-lbl">Kedi Irkı Eşleşmesi:</p>

              {/* Top match ─ pembe çerçeve */}
              {top && (
                <div className="rp-top-match">
                  <div className="rp-top-match-info">
                    <p className="rp-top-breed">{top.breed}</p>
                    <p className="rp-top-pct">{pct(top.confidence)} güven</p>
                  </div>
                  {breedImg(top.breed) && (
                    <img src={breedImg(top.breed)} alt={top.breed} className="rp-breed-thumb" />
                  )}
                </div>
              )}

              {/* Diğer eşleşmeler */}
              {others.length > 0 && (
                <>
                  <p className="rp-match-section-lbl" style={{ marginTop: '1.2rem' }}>Diğer Irk Eşleşmesi:</p>
                  {others.map((p, i) => (
                    <div className="rp-other-match" key={i}>
                      {breedImg(p.breed) ? (
                        <img src={breedImg(p.breed)} alt={p.breed} className="rp-other-thumb" />
                      ) : (
                        <div className="rp-other-thumb rp-thumb-placeholder">🐱</div>
                      )}
                      <div>
                        <p className="rp-other-breed">{p.breed}</p>
                        <p className="rp-other-pct">{pct(p.confidence)} güven</p>
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>

          {/* Alt ─ detaylı ırk bilgisi */}
          {info && (
            <div className="rp-info-block">
              <h2 className="rp-breed-heading">
                {info.name_tr || top.breed} <span className="rp-paw">🐾</span>
              </h2>

              {/* Temel Bilgiler */}
              <div className="rp-section">
                <p className="rp-section-lbl">Temel Bilgiler</p>
                <div className="rp-info-grid">
                  <div className="rp-info-cell"><span className="rp-info-key">Beden:</span> {info.size || '—'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Yükseklik:</span> {info.height || '20-30 cm'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Ağırlık:</span> {info.weight || '—'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Ömür:</span> {info.lifespan || '—'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Köken:</span> {info.origin || '—'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Nadirliği:</span> {info.rarity || 'Nadir Değil'}</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Karışık:</span> Safkan</div>
                  <div className="rp-info-cell"><span className="rp-info-key">Beğendikleri:</span> {info.nutrition ? info.nutrition.split('.')[0] : '—'}</div>
                </div>
              </div>

              {/* Sağlık + Mizaç ─ yan yana */}
              <div className="rp-two-col">
                <div className="rp-two-item">
                  <p className="rp-section-lbl">Dikkat Çeken Sağlık Eğilimleri ve Özel Bakım İhtiyaçları</p>
                  <p className="rp-body-text">
                    {info.health && <><span className="rp-inline-link">{info.health.split('.')[0]}</span>{'. '}</>}
                    {info.care}
                  </p>
                </div>
                <div className="rp-two-item">
                  <p className="rp-section-lbl">Tipik Mizaç ve Davranış</p>
                  <p className="rp-body-text">{info.character}</p>
                </div>
              </div>

              {/* Diyet ve İlginç Bilgiler */}
              <div className="rp-section">
                <p className="rp-section-lbl">Diyet ve İlginç Bilgiler</p>
                <div className="rp-info-grid rp-diet-grid">
                  <div className="rp-diet-cell">
                    <p className="rp-diet-key">Özel</p>
                    <p className="rp-diet-val rp-inline-link">{info.nutrition?.split('.')[1]?.trim() || info.nutrition || '—'}</p>
                  </div>
                  <div className="rp-diet-cell">
                    <p className="rp-diet-key">Fiziksel</p>
                    <p className="rp-diet-val">
                      {info.size && `${info.size} büyüklükte, `}
                      {info.shedding && `${info.shedding} tüy dökümü. `}
                      {info.grooming && `Bakım: ${info.grooming}.`}
                    </p>
                  </div>
                  <div className="rp-diet-cell">
                    <p className="rp-diet-key">Dış görünüş</p>
                    <p className="rp-diet-val rp-inline-link">{info.care?.split('.')[0]}.</p>
                  </div>
                  <div className="rp-diet-cell">
                    <p className="rp-diet-key">Dikkat çekici</p>
                    <p className="rp-diet-val rp-inline-link">
                      Çocuklarla uyum: {info.child_friendly || '—'}.{' '}
                      Diğer hayvanlarla: {info.pet_friendly || '—'}.{' '}
                      Aktivite: {info.activity_level || '—'}.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── BUTONLAR ── */}
      <div className="rp-actions">
        <button className="rp-btn-download" onClick={handleDownload}>
          ⬇ Sonucu İndir
        </button>
        <button className="rp-btn-reset" onClick={() => navigate('/')}>
          Başka bir fotoğrafa bakın.
        </button>
      </div>

      {/* ── FOOTER ── */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <div className="footer-logo"><img src="/kedi-ai-logo.svg" alt="Kedi.ai" className="footer-logo-img" /> Kedi.ai</div>
            <p>Kedinizin fotoğrafından ırkını bulmanıza yardımcı olan ücretsiz bir yapay zeka aracıdır.</p>
          </div>
          <div className="footer-cols">
            <div className="footer-col">
              <h4>Eğlenceli Araçlar</h4>
              <Link to="/cat-translator">Kedi Çevirmeni</Link>
            </div>
            <div className="footer-col">
              <h4>Hakkında</h4>
              <Link to="/about">Hakkımızda</Link>
              <a href="#">Gizlilik Politikası</a>
              <a href="#">Hizmet Şartları</a>
            </div>
            <div className="footer-col">
              <h4>Temas etmek</h4>
              <a href="mailto:merhaba@kedimincins.ne">merhaba@kedimin cinsi ne.</a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2026 Kedi.ai Tüm Hakları saklıdır.</p>
        </div>
      </footer>
    </div>
  );
}
