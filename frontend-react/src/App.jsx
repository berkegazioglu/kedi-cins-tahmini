import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import ApiService from './services/api';
import './App.css';

const SAMPLE_CATS = [
  { id: 1,  img: 'https://whatbreedismycat.app/cat/persian.jpg',           breed: 'Persian',             confidence: 91 },
  { id: 2,  img: 'https://whatbreedismycat.app/cat/maine-coon.jpg',        breed: 'Maine Coon',          confidence: 89 },
  { id: 3,  img: 'https://whatbreedismycat.app/cat/british-shorthair.jpg', breed: 'British Shorthair',   confidence: 93 },
  { id: 4,  img: 'https://whatbreedismycat.app/cat/siamese.jpg',           breed: 'Siyam',               confidence: 87 },
  { id: 5,  img: 'https://whatbreedismycat.app/cat/abyssinian.jpg',        breed: 'Habeş Kedisi',        confidence: 86 },
  { id: 6,  img: 'https://whatbreedismycat.app/cat/bengal.jpg',            breed: 'Bengal',              confidence: 88 },
  { id: 7,  img: 'https://whatbreedismycat.app/cat/russian-blue.jpg',      breed: 'Russian Blue',        confidence: 90 },
  { id: 8,  img: 'https://whatbreedismycat.app/cat/birman.jpg',            breed: 'Birman',              confidence: 88 },
  { id: 9,  img: 'https://whatbreedismycat.app/cat/american-shorthair.jpg',breed: 'American Shorthair',  confidence: 85 },
  { id: 10, img: 'https://whatbreedismycat.app/cat/bombay.jpg',            breed: 'Bombay',              confidence: 89 },
  { id: 11, img: 'https://whatbreedismycat.app/cat/norwegian-forest-cat.jpg', breed: 'Norveç Orman Kedisi', confidence: 84 },
  { id: 12, img: 'https://whatbreedismycat.app/cat/sphynx.jpg',            breed: 'Sfenks',              confidence: 83 },
  { id: 13, img: 'https://whatbreedismycat.app/cat/scottish-fold.jpg',     breed: 'Scottish Fold',       confidence: 92 },
  { id: 14, img: 'https://whatbreedismycat.app/cat/chartreux.jpg',         breed: 'Chartreux',           confidence: 86 },
  { id: 15, img: 'https://whatbreedismycat.app/cat/turkish-angora.jpg',    breed: 'Ankara Kedisi',       confidence: 87 },
  { id: 16, img: 'https://whatbreedismycat.app/cat/ocicat.jpg',            breed: 'Ocicat',              confidence: 85 },
  { id: 17, img: 'https://whatbreedismycat.app/cat/tonkinese.jpg',         breed: 'Tonkinese',           confidence: 83 },
  { id: 18, img: 'https://whatbreedismycat.app/cat/burmese.jpg',           breed: 'Burmese',             confidence: 90 },
  { id: 19, img: 'https://whatbreedismycat.app/cat/savannah.jpg',          breed: 'Savana',              confidence: 91 },
  { id: 20, img: 'https://whatbreedismycat.app/cat/munchkin.jpg',          breed: 'Munchkin',            confidence: 81 },
  { id: 21, img: 'https://whatbreedismycat.app/cat/cornish-rex.jpg',       breed: 'Cornish Rex',         confidence: 88 },
  { id: 22, img: 'https://whatbreedismycat.app/cat/exotic-shorthair.jpg',  breed: 'Egzotik Shorthair',   confidence: 81 },
  { id: 23, img: 'https://whatbreedismycat.app/cat/havana-brown.jpg',      breed: 'Havana Kedisi',       confidence: 82 },
  { id: 24, img: 'https://whatbreedismycat.app/cat/manx.jpg',              breed: 'Manks',               confidence: 85 },
  { id: 25, img: 'https://whatbreedismycat.app/cat/american-bobtail.jpg',  breed: 'Amerikan Bobtail',    confidence: 80 },
  { id: 26, img: 'https://whatbreedismycat.app/cat/selkirk-rex.jpg',       breed: 'Selkirk Rex',         confidence: 84 },
  { id: 27, img: 'https://whatbreedismycat.app/cat/snowshoe.jpg',          breed: 'Snowshoe',            confidence: 81 },
];

const FAQ_ITEMS = [
  { q: 'Kedimin hangi cins olduğunu nasıl öğrenebilirim?', a: 'Kedinizin fotoğrafını yükleyin ve yapay zekamız saniyeler içinde olası 3 ırkı tahmin etsin. Her tahmin, eşleşme yüzdesi ve ırk özellikleri içerir.' },
  { q: 'En iyi kedi cinsi belirleme uygulaması hangisi?', a: 'Uygulamamız, 59 farklı kedi ırkını tanıyabilen derin öğrenme modeli kullanır. YOLO11n + Ensemble Model kombinasyonu ile yüksek doğruluk sağlar.' },
  { q: 'Yapay zekâ tespiti ne kadar doğru?', a: 'Modelimiz %63+ genel doğruluk oranıyla çalışmaktadır. Bazı yaygın ırklar için bu oran çok daha yüksektir.' },
  { q: 'Kedim safkan mı?', a: 'Yapay zekamız görsel özellikler üzerinden tahmin yapar. Kesin yanıt için DNA testi önerilir, ancak uygulamamız iyi bir başlangıç noktası sunar.' },
  { q: 'Yapay zekâ melez kedileri tanıyabilir mi?', a: 'Evet, modelimiz karma özellikleri analiz ederek en yakın ırk eşleşmesini bulur. Melez kediler için birden fazla ırk önerisi sunulur.' },
  { q: 'Yapay zekâ tespiti ile DNA testi arasındaki fark nedir?', a: 'Yapay zekâ görsel analiz yapar ve ücretsizdir; DNA testi genetik geçmişi kesin olarak ortaya çıkarır ancak ücretlidir.' },
  { q: 'Kedimin tüy rengi veya deseni bana onun cinsini söyleyebilir mi?', a: 'Evet, tüy deseni önemli bir göstergedir. Ancak tek başına yeterli değildir; yüz şekli, kulak yapısı ve vücut oranları da önemlidir.' },
  { q: 'Fotoğrafım güvende ve gizli mi?', a: 'Evet, fotoğrafınız analiz tamamlandıktan hemen sonra silinir. Verilerinizi asla saklamıyor veya paylaşmıyoruz.' },
  { q: 'Kedi cinsi belirleme aracını ücretsiz kullanabilir miyim?', a: 'Evet, tamamen ücretsizdir. Kayıt olmaya gerek yoktur, sınırsız kullanabilirsiniz.' },
  { q: 'Fotoğrafımdan kedi cinsi nasıl belirlenir?', a: 'Yüklenen fotoğraf önce YOLO11n modeliyle kedi bölgesi tespit edilir, ardından Ensemble Model (ResNet50 + EfficientNet + MobileNet) ile ırk sınıflandırması yapılır.' },
];

const BREEDS_TABLE = [
  ['Habeş Kedisi',    'Orta',          'Kırmızı, tarçın, mavi, leylak — agouti',  'Badem, amber/yeşil/altın',     'Aktif, meraklı, atletik, oyuncu'],
  ['Bengal',          'Orta-Büyük',    'Mermer/benekli — kahverengi, kar',         'Büyük, oval',                  'Enerjik, sosyal, konuşkan, sevgi dolu'],
  ['Birman',          'Orta-Büyük',    'Renk noktalı, beyaz eldivenli',            'Derin mavi',                   'Nazik, sakin, sevecen, uysal'],
  ['British Shorthair','Büyük',        'Tüm renkler, en yaygın mavi',              'Geniş, yuvarlak — sarı/bakır', 'Sakin, sadık, bağımsız, kolay bakımlı'],
  ['Maine Coon',      'Çok Büyük',     'Tüm renkler ve desenler',                  'Geniş, oval — tüm renkler',    'Neşeli, zeki, köpek gibi, aile dostu'],
  ['Persian',         'Orta-Büyük',    'Tüm renkler ve desenler',                  'Büyük, yuvarlak — canlı',      'Sakin, bağımlı, sevgi dolu, ev kedisi'],
  ['Ragdoll',         'Büyük-Çok Büyük','Renk noktalı, mitted, bicolor',           'Oval, mavi',                   'Uysal, sakin, kucakta durmayı sever'],
  ['Russian Blue',    'Orta',          'Mavi-gümüş',                               'Koyu yeşil',                   'Çekingen, sadık, zeki, sessiz'],
  ['Siyam',           'Orta',          'Renk noktalı (seal, blue, chocolate)',      'Badem, mavi',                  'Çok konuşkan, sosyal, bağımlı, oyuncu'],
  ['Scottish Fold',   'Orta',          'Tüm renkler ve desenler',                  'Geniş, yuvarlak',              'Uyumlu, oyuncu, meraklı, sakin'],
];

const VISIBLE_COUNT = 4;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openFaq, setOpenFaq] = useState(null);
  const [carouselIdx, setCarouselIdx] = useState(0);
  const [testiIdx, setTestiIdx] = useState(0);
  const navigate = useNavigate();
  const TESTIMONIALS = [
    {
      quote: '“ Evcil hayvan haftası boyunca sınıfta kullandık! Öğrencilerim kedilerinin olası cinslerini görmeyi çok sevdiler. Genetik ve hayvan özellikleri hakkında harika sohbetlere yol açtı. ”',
      author: 'Jake Reynolds',
      title: 'İlkokul Öğretmeni',
      stars: 5,
      rating: '4.8/5',
    },
    {
      quote: '“ Namık Kemal Üniversitesi Bilgisayar Mühendisliği 4. Sınıf öğrencilerimizin geliştirmiş olduğu Evcil Kedi Türü Tespit uygulaması tutarlı sonuçlarıyla göze çarpıyor. ”',
      author: 'Doç. Dr. Pınar Cihan',
      title: 'Dekan Yardımcısı, Namık Kemal Üniversitesi',
      stars: 5,
      rating: '5.0/5',
    },
  ];
  const testi = TESTIMONIALS[testiIdx];
  const [navScrolled, setNavScrolled] = useState(false);
  const uploadRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => setNavScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleImageSelect = (file, previewUrl) => {
    setSelectedFile(file);
    setPreview(previewUrl);
    setResults(null);
    setError(null);
  };

  const handlePredict = async (file) => {
    const f = file || selectedFile;
    if (!f) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await ApiService.predictBreed(f, { topK: 5 });
      if (response.success) {
        setResults(response);
        navigate('/result', {
          state: { predictions: response.predictions, preview, breedInfo: response.breed_info },
        });
      } else {
        setError(response.error || 'Tahmin yapılamadı');
      }
    } catch (err) {
      setError(err.message || 'Bir hata oluştu');
    } finally {
      setIsLoading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  const scrollToUpload = () => {
    uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const maxIdx = Math.max(0, SAMPLE_CATS.length - VISIBLE_COUNT);

  return (
    <div className="page">
      {/* ── NAVBAR ── */}
      <nav className={`navbar ${navScrolled ? 'scrolled' : ''}`}>
        <div className="nav-inner">
          <a href="#" className="nav-logo" onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>
            <img src="/kedi-ai-logo.svg" alt="Kedi.ai" className="nav-logo-img" /> Kedi.ai
          </a>
          <ul className="nav-links">
            <li><a href="#hero" onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>Anasayfa</a></li>
            <li><Link to="/cat-translator">Kedi Çevirmeni</Link></li>
            <li><a href="#how-it-works">Nasıl Çalışır</a></li>
            <li><a href="#breeds-table">Kedi Irkları Tablosu</a></li>
            <li><a href="#faq">SSS</a></li>
          </ul>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="hero" id="hero">
        <div className="hero-inner">
          <div className="hero-left">
            <h1 className="hero-title">Yapay Zeka Kedi Irkı<br />Tanımlayıcısı</h1>
            <p className="hero-desc">
              Kedinizin fotoğrafını yükleyin ve yapay zekanın olası ırkını belirlemesine izin verin.
              Her biri eşleşme yüzdesi, ırk özellikleri ve bakım ipuçları içeren <strong>3 olası ırk türü</strong> elde
              edeceksiniz—ücretsiz, hızlı ve akıllı.
            </p>
            <div className="hero-proof">
              <div className="proof-avatars">
                <img src="https://images.unsplash.com/photo-1548247416-ec66f4900b2e?w=80&h=80&fit=crop&auto=format" alt="Balinese" />
                <img src="https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=80&h=80&fit=crop&auto=format" alt="Bombay" />
                <img src="https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=80&h=80&fit=crop&auto=format" alt="Siamese" />
              </div>
              <span>Kaggle'den toplanan 100.000'den fazla kedi fotoğrafı ile eğitildi.</span>
            </div>
          </div>
          <div className="hero-right" id="upload" ref={uploadRef}>
            <ImageUploader
              onImageSelect={handleImageSelect}
              onPredict={handlePredict}
              isLoading={isLoading}
              preview={preview}
              results={results}
              error={error}
              onReset={resetUpload}
            />
          </div>
        </div>
      </section>

      {/* ── DETAILED RESULTS (below hero, only when results exist) ── */}
      {results && (
        <section className="results-section" ref={resultsRef}>
          <div className="section-container">
            <PredictionResults results={results} preview={preview} onReset={resetUpload} />
          </div>
        </section>
      )}

      {/* ── SAMPLE RESULTS CAROUSEL ── */}
      <section className="samples-section">
        <div className="section-container">
          <h2 className="section-title">Gerçek Kedilerden Gerçek Sonuçlar</h2>
          <p className="section-desc">
            Yapay zekâ aracımızın kullanıcı fotoğraflarından farklı kedi ırklarını nasıl belirlediğini görün.
            Her kedi için özelliklerini, görünümünü ve sağlığı hakkında ayrıntılı bilgi içeren 3 olası ırk eşleşmesi elde edilir.
          </p>
          <div className="carousel-wrap">
            <button className="car-btn" onClick={() => setCarouselIdx(i => Math.max(i - 1, 0))} disabled={carouselIdx === 0}>‹</button>
            <div className="carousel-track">
              {SAMPLE_CATS.slice(carouselIdx, carouselIdx + VISIBLE_COUNT).map(cat => (
                <div className="car-item" key={cat.id}>
                  <img src={cat.img} alt={cat.breed} />
                  <p>{cat.breed} <strong className="conf-pink">%{cat.confidence}</strong></p>
                </div>
              ))}
            </div>
            <button className="car-btn" onClick={() => setCarouselIdx(i => Math.min(i + 1, maxIdx))} disabled={carouselIdx === maxIdx}>›</button>
          </div>
          <div className="car-dots">
            {Array.from({ length: maxIdx + 1 }).map((_, i) => (
              <button key={i} className={`dot ${i === carouselIdx ? 'active' : ''}`} onClick={() => setCarouselIdx(i)} />
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="hiw-section" id="how-it-works">
        <div className="section-container hiw-grid">
          <div className="hiw-mockup">
            <div className="result-mock-card">
              <span className="mock-deco mock-deco-tl">❝</span>
              <span className="mock-deco mock-deco-br">❞</span>
              <div className="mock-deco-hearts">💕</div>
              <div className="mock-deco-gift">🎁</div>
              <div className="mock-cat-frame">
                <img src="https://whatbreedismycat.app/cat/persian.jpg" alt="Persian" className="mock-cat-img" />
              </div>
              <div className="mock-results">
                <p className="mock-lbl">Kedi Cinsi Eşleşmesi:</p>
                <div className="mock-top-match">
                  <div>
                    <div className="mock-breed">Persian</div>
                    <div className="mock-conf">88.5% güven</div>
                  </div>
                  <img src="https://whatbreedismycat.app/cat/persian.jpg" className="mock-breed-thumb" alt="Persian" />
                </div>
                <p className="mock-lbl" style={{ marginTop: '1.1rem', marginBottom: '0.65rem' }}>Diğer Cins Eşleşmeleri:</p>
                <div className="mock-other-row">
                  <img src="https://whatbreedismycat.app/cat/birman.jpg" className="mock-breed-thumb" alt="diğer" />
                  <div>
                    <div className="mock-other-name">İngiliz Uzun Tüylü</div>
                    <div className="mock-other-conf">75.2% güven</div>
                  </div>
                </div>
                <div className="mock-other-row">
                  <img src="https://whatbreedismycat.app/cat/maine-coon.jpg" className="mock-breed-thumb" alt="diğer" />
                  <div>
                    <div className="mock-other-name">Himalaya Kedisi</div>
                    <div className="mock-other-conf">60.1% güven</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="hiw-text">
            <h2>Yapay Zekamız Kedinizin Cinsini Nasıl Buluyor?</h2>
            <p>Yapay zekamız, Uluslararası Kedi Birliği (TICA) gibi güvenilir kaynaklardan alınan ırk standartları kullanılarak eğitilmiştir. Bu, modelimizin yüz şekli, kulak boyutu, tüy tipi ve daha fazlası gibi bilinen özelliklere sahip binlerce safkan kedi örneği gördüğü anlamına gelir.</p>
            <p>Fotoğraf yüklediğinizde, yapay zekâ kedinizin fiziksel özelliklerine bakar ve bunları bilinen kalıplarla karşılaştırır.</p>
            <p>Sonuç? Kedinizin en çok benzediği 3 kedi ırkını, eşleşme yüzdesini, ortak özelliklerini ve bakım ipuçlarını öğreneceksiniz.</p>
            <p style={{ color: '#888', fontSize: '0.9rem' }}>Ama şöyle bir durum var: Kedilerin çoğu melezdir ve fotoğraflar her şeyi tam olarak gösteremez. Aracımız görsel özelliklere dayanarak aklılıca bir tahminde bulunur, ancak mükemmel değildir. Kedinizin tüm DNA'sını veya sağlık geçmişini göremez.</p>
          </div>
        </div>
      </section>

      {/* ── STEPS ── */}
      <section className="steps-section">
        <div className="section-container steps-grid">
          <div className="steps-left">
            <h2>Yapay Zeka Aracımızla Kedinizin Cinsini Nasıl Belirleyebilirsiniz?</h2>
            <p className="section-desc" style={{ textAlign: 'left', marginBottom: '2rem' }}>
              Kedinizin fotoğrafını yüklemeniz yeterli; yapay zekamız hemen taramaya başlayacak. Birkaç saniye içinde, tatlı bilgilerle birlikte 5 olası ırk kimliği göreceksiniz.
            </p>
            {[
              { n: '1', title: 'Net bir fotoğraf yükleyin', desc: 'Kedinizin önden çekilmiş bir fotoğrafını çekin veya galerinizden birini seçin. Fotoğrafın parlak, net ve kedinizin yüzüne odaklanmış olduğundan emin olun. Bulanık yüzü, yoğun gölgeli ve kadraja başka evcil hayvanların bulunduğu fotoğraflardan kaçının.' },
              { n: '2', title: 'Yapay Zekanın Otomatik Olarak Tarama Yapmasına İzin Verin', desc: 'Fotoğrafı yüklediğiniz anda yapay zekamız hemen çalışmaya başlar; hiçbir düğmeye basmanıza gerek yok. Kedinizin yüz şekline, kulaklarına, tüylerine ve diğer özelliklerine bakar. Tarama yaklaşık 5 saniye sürer.' },
              { n: '3', title: 'Irk Eşleştirme Sonuçlarını Alın', desc: 'Kedinizin en çok benzediği 3 kedi ırkını göreceksiniz. Her birinin eşleşme yüzdesi, temel özellikleri, davranış bilgileri ve bakım ipuçları bulunmaktadır. Bunlar rastgele tahminler değil; görsel özelliklere ve ırk standartlarına dayanmaktadır.' },
              { n: '4', title: 'Sonuçlarınızı İndirin veya Paylaşın', desc: 'Sonuçlarınızın bir kopyasını indirebilir veya sosyal medyada paylaşabilirsiniz. Kedinizin "gizli" ırkını arkadaşlarınızla paylaşın!' },
            ].map(step => (
              <div className="step-row" key={step.n}>
                <div className="step-num">{step.n}</div>
                <div className="step-body">
                  <h3>{step.title}</h3>
                  <p>{step.desc}</p>
                </div>
              </div>
            ))}
            <button className="btn-pink" onClick={scrollToUpload}>Kedimi Şimdi Tanımla →</button>
          </div>
          <div className="steps-right">
            <div className="steps-mock-card">
              <div className="steps-mock-drop" onClick={scrollToUpload}>
                <div className="smd-icon">📷</div>
                <p>Click or drag photo here to upload</p>
                <small>JPG, JPEG, PNG, WEBP · Less than 10MB</small>
                <button className="btn-pink smd-btn">Upload Cat Photo</button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── FEATURE: NEW CAT ── */}
      <section className="feature-sec">
        <div className="section-container feat-grid">
          <div className="feat-img">
            <img src="https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=650&h=500&fit=crop" alt="yeni kedi" />
          </div>
          <div className="feat-text">
            <h2>Yeni bir kedi mi edindiniz?</h2>
            <p>Bir kedi sahiplendiniz ama hangi cins olduğunu bilmiyor musunuz? İster barınaktan ister sokaktan gelmiş olsun, olası cins karışımını bilmek, özelliklerini, davranışlarını ve bakım ihtiyaçlarını anlamanıza yardımcı olur. Yeni tüylü dostunuzla bağ kurmanın ve nereden geldiğini öğrenmenin basit bir yoludur.</p>
            <button className="btn-pink" onClick={scrollToUpload}>Kedimi Şimdi Tanımla →</button>
          </div>
        </div>
      </section>

      <section className="feature-sec bg-light-gray">
        <div className="section-container feat-grid reverse">
          <div className="feat-text">
            <h2>Arkadaşlarınızla mı yoksa çevrimiçi mi paylaşıyorsunuz?</h2>
            <p>Sevimli kedi fotoğrafları mı paylaşıyorsunuz? İnsanlar "Bu hangi cins?" diye soracaklar. Artık eğlenceli bir cevabınız olacak; üstelik cins isimleri ve ilginç bilgiler içeren paylaşabilir bir sonuç kartınız da olacak. Instagram, TikTok veya sadece grup sohbetinizi neşelendirmek için harika.</p>
            <button className="btn-pink" onClick={scrollToUpload}>Kedimi Şimdi Tanımla →</button>
          </div>
          <div className="feat-img">
            <img src="https://images.unsplash.com/photo-1548802673-380ab8ebc7b7?w=650&h=500&fit=crop" alt="paylaşım" />
          </div>
        </div>
      </section>

      <section className="feature-sec">
        <div className="section-container feat-grid">
          <div className="feat-img">
            <img src="https://whatbreedismycat.app/cat/british-shorthair.webp" alt="British Shorthair kedi" />
          </div>
          <div className="feat-text">
            <h2>Çocukların Evcil Hayvanlar Hakkında Bilgi Edinmelerine Yardımcı Olmak?</h2>
            <p>Çocuklar soru sormayı severler ve kediler sürprizlerle doludur. Irk sonuçlarını bir öğrenme anına dönüştürün. Irkların nereden geldiği, özellikleri ve onlara nasıl bakılacağı hakkında konuşun. Eğlenceli, uygulamalı ve hayvanlara karşı merak uyandırıcı.</p>
            <button className="btn-pink" onClick={scrollToUpload}>Kedimi Şimdi Tanımla →</button>
          </div>
        </div>
      </section>

      {/* ── BREED TABLE ── */}
      <section className="table-section" id="breeds-table">
        <div className="section-container">
          <h2 className="section-title">Kedi Irkları Tablosu: 10 Popüler Irkı Bir Bakışta Karşılaştırın</h2>
          <p className="section-desc">
            Kedinizin ırkını görünüşüne göre belirlemeye çalışıyor musunuz? Uluslararası Kedi Birliği (TICA) ve Kedi Severler Birliği'nin (CFA) resmi verilerini kullanarak 10 yaygın kedi ırkını bir tablo oluşturduk.
          </p>
          <div className="table-wrap">
            <table className="breeds-table">
              <thead>
                <tr>
                  <th>Irk Adı</th>
                  <th>Boyut</th>
                  <th>Tüy Rengi ve Deseni</th>
                  <th>Göz Rengi ve Şekli</th>
                  <th>Tipik Mizaç ve Davranış</th>
                </tr>
              </thead>
              <tbody>
                {BREEDS_TABLE.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j}>{j === 0 ? <strong>{cell}</strong> : cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── TRUST ── */}
      <section className="trust-section">
        <div className="section-container">
          <h2 className="section-title">Kedi Sahipleri Neden Bize Güveniyor?</h2>
          <div className="trust-grid">
            {[
              { icon: '⬆️', title: 'Kullanımı Kolay ve Ücretsiz', desc: 'Kayıt olmaya gerek yok. Uzun formlar doldurmaya gerek yok. Sadece bir fotoğraf yükleyin ve kedinizin ırk sonuçlarını saniyeler içinde görün. Herkesin kullanabileceği şekilde tasarlandı; teknik beceriye gerek yok.' },
              { icon: '⭐', title: 'Güvenilir Irk Verileriyle Eğitilmiştir', desc: "Yapay zekamız, Uluslararası Kedi Birliği'nin (TICA) resmi ırk standartları kullanılarak eğitilmiştir; bu da sonuçların daha güvenilir ve uzman bilgisine dayalı hale gelmektedir." },
              { icon: '🛡️', title: 'Varsayılan olarak özel', desc: 'Fotoğrafınızı asla kaydetmiyoruz. Tarama işleminden hemen sonra silinir. Hesap yok, veri takibi yok; sadece siz ve kedinizin sonucu. %100 gizlilik.' },
              { icon: '🎂', title: 'Bir Irk İsminden Daha Fazlası', desc: 'Sadece bir etiket almıyorsunuz. Size eğlegen yüzdesini, temel özelliklerini, davranış ipuçlarını ve bakım bilgilerini veriyoruz. Bu, kedinizi sadece adlandırmanıza değil, anlamanıza da yardımcı olur.' },
            ].map((item, i) => (
              <div className="trust-card" key={i}>
                <div className="trust-icon">{item.icon}</div>
                <h3>{item.title}</h3>
                <p>{item.desc}</p>
              </div>
            ))}
          </div>
          <div className="testimonial-box">
              <button className="testi-arrow testi-prev" onClick={() => setTestiIdx(i => (i - 1 + TESTIMONIALS.length) % TESTIMONIALS.length)}>&#8249;</button>
              <div className="testi-content">
                <p className="testimonial-quote">{testi.quote}</p>
                <p className="testimonial-author">{testi.author}</p>
                <p className="testimonial-title">{testi.title}</p>
                <p className="testimonial-stars">
                  {'★'.repeat(testi.stars)} <span>{testi.rating}</span>
                </p>
                <div className="testi-dots">
                  {TESTIMONIALS.map((_, i) => (
                    <button key={i} className={`testi-dot${i === testiIdx ? ' active' : ''}`} onClick={() => setTestiIdx(i)} />
                  ))}
                </div>
              </div>
              <button className="testi-arrow testi-next" onClick={() => setTestiIdx(i => (i + 1) % TESTIMONIALS.length)}>&#8250;</button>
            </div>
        </div>
      </section>

      {/* ── FAQ ── */}
      <section className="faq-section" id="faq">
        <div className="section-container">
          <h2 className="section-title">Sıkça Sorulan Sorular</h2>
          <div className="faq-list">
            {FAQ_ITEMS.map((item, i) => (
              <div className={`faq-item ${openFaq === i ? 'open' : ''}`} key={i}>
                <button className="faq-q" onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                  <span>{item.q}</span>
                  <span className="faq-icon">{openFaq === i ? '−' : '+'}</span>
                </button>
                {openFaq === i && <div className="faq-a">{item.a}</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-section">
        <div className="section-container">
          <h2>Kedinizin cinsini keşfetmeye hazır mısınız?</h2>
          <p>Fotoğraf yükleyin ve kedinizin en iyi 3 ırk eşleşmesini saniyeler içinde görün—ücretsiz, hızlı ve gizli.</p>
          <button className="btn-pink btn-lg" onClick={scrollToUpload}>Kedimi Şimdi Tanımla →</button>
        </div>
      </section>

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
              <a href="/cat-translator">Kedi Çevirmeni</a>
            </div>
            <div className="footer-col">
              <h4>Hakkında</h4>
              <a href="#">Hakkımızda</a>
              <a href="#">Gizlilik Politikası</a>
              <a href="#">Hizmet Şartları</a>
            </div>
            <div className="footer-col">
              <h4>Temas etmek</h4>
              <a href="mailto:info@kedimhangicins.com">info@kedimhangicins.com</a>
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

export default App;

