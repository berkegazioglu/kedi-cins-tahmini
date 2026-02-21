import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './AboutPage.css';

const AboutPage = () => {
  const [navScrolled] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="about-page">

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

      {/* ── HEADER ── */}
      <section className="ab-hero">
        <div className="ab-container">
          <h1 className="ab-title">Hakkımızda</h1>
          <p className="ab-subtitle">Kedi.ai'nin arkasındaki ekiple tanışın.</p>
        </div>
      </section>

      {/* ── INTRO ── */}
      <section className="ab-section ab-intro-section">
        <div className="ab-container">
          <div className="ab-intro-box">
            <div className="ab-intro-icon">🐱</div>
            <div>
              <p className="ab-intro-lead">Merhaba! Biz sadece küçük bir kedi meraklıları grubuyuz.</p>
              <p className="ab-intro-sub">Aynı şeyi sormaktan vazgeçmeyenler:</p>
              <p className="ab-intro-quote">"Kanepemin üzerindeki bu küçük tüy yumağı ne tür bir kedi?"</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── STORY ── */}
      <section className="ab-section">
        <div className="ab-container">
          <h2 className="ab-section-title">Bizim Hikayemiz</h2>
          <p className="ab-text">
            Her şey, içimizden birinin barınaktan bir yavru kedi sahiplenmesiyle başladı. Kedinin tüyleri İran kedisinin tüylerine, yüzü Altın
            Çinçilla'nın yüzüne ve Bengal kedisinin tavırına benziyordu. Google'da arama yaptık, forumlara katıldık, hatta birkaç uygulamaya döndük
            — ama hiçbiri net bir cevap vermedi. Ve çoğu, sonucun neden çıktığını açıklamadı.
          </p>
          <p className="ab-text">
            Dolayısıyla, kedi severler ve alet yapımcıları olarak şöyle dedik: <strong>"Daha iyisini yapalım."</strong>
          </p>
          <p className="ab-text">İşte Kedi.ai böyle doğdu.</p>

          <div className="ab-cats-grid">
            <img src="https://whatbreedismycat.app/cat/persian.jpg"          alt="kedi 1" className="ab-cat-img" />
            <img src="https://whatbreedismycat.app/cat/maine-coon.jpg"       alt="kedi 2" className="ab-cat-img" />
            <img src="https://whatbreedismycat.app/cat/bengal.jpg"           alt="kedi 3" className="ab-cat-img" />
            <img src="https://whatbreedismycat.app/cat/british-shorthair.jpg" alt="kedi 4" className="ab-cat-img" />
          </div>
        </div>
      </section>

      {/* ── WHAT MAKES US DIFFERENT ── */}
      <section className="ab-section ab-diff-section">
        <div className="ab-container">
          <h2 className="ab-section-title">Bizi Farklı Kılan Nedir?</h2>
          <p className="ab-text">
            Biz, sadece cins adı vermekle kalmayan, size gerçek bilgiler sunan yapay zeka destekli bir araç geliştirdik.
          </p>
          <div className="ab-feat-grid">
            <div className="ab-feat-item">
              <span className="ab-feat-dot" />
              <div>
                <p className="ab-feat-title">Fotoğraf Analizi</p>
                <p className="ab-feat-desc">Gelişmiş yapay zeka, fiziksel özellikleri ve desenleri tanımlar.</p>
              </div>
            </div>
            <div className="ab-feat-item">
              <span className="ab-feat-dot" />
              <div>
                <p className="ab-feat-title">Irk Bilgileri</p>
                <p className="ab-feat-desc">Kedinizin hangi ırklara benzediğine dair detaylı açıklamalar.</p>
              </div>
            </div>
            <div className="ab-feat-item">
              <span className="ab-feat-dot" />
              <div>
                <p className="ab-feat-title">Sağlık Notları</p>
                <p className="ab-feat-desc">Olası sağlık ve davranışsal bilgiler.</p>
              </div>
            </div>
            <div className="ab-feat-item">
              <span className="ab-feat-dot" />
              <div>
                <p className="ab-feat-title">Güvenilir Kaynaklar</p>
                <p className="ab-feat-desc">TICA'dan elde edilen gerçek verilere ve veteriner hekimler tarafından desteklenen çalışmalara dayanmaktadır.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── MISSION ── */}
      <section className="ab-section">
        <div className="ab-container">
          <h2 className="ab-section-title">Misyonumuz</h2>
          <p className="ab-text">
            <span className="ab-pink">Her kedinin bir hikayesi olduğuna inanıyoruz.</span> Safkan olmasalar bile, özelliklerini ve genetik ipuçlarını içeriyor. Misyonumuz, bu ipuçlarını basit, eğlenceli ve gerçekten faydalı bir şekilde ortaya çıkarmanıza yardımcı olmaktır.
          </p>
          <div className="ab-pillars">
            <div className="ab-pillar">
              <div className="ab-pillar-icon">🔍</div>
              <p className="ab-pillar-title">Basit</p>
              <p className="ab-pillar-desc">Sadece bir fotoğraf yükleyin ve anında sonuç alın.</p>
            </div>
            <div className="ab-pillar">
              <div className="ab-pillar-icon">🎉</div>
              <p className="ab-pillar-title">Eğlence</p>
              <p className="ab-pillar-desc">Kedinizin sizi güldüren ve hararetlendirmiş karakterini keşfedin.</p>
            </div>
            <div className="ab-pillar">
              <div className="ab-pillar-icon">💡</div>
              <p className="ab-pillar-title">Kullanışlı</p>
              <p className="ab-pillar-desc">Sağlık ve davranış hakkında uygulanabilir bilgiler edinin.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── PRIVACY ── */}
      <section className="ab-section ab-privacy-section">
        <div className="ab-container">
          <h2 className="ab-section-title">Önce Gizlilik</h2>
          <p className="ab-text">
            <span className="ab-teal">Gizliliğinize önem veriyoruz</span> — fotoğrafınız sonuçlar oluşturulduktan hemen sonra silinir. Hiçbir şey saklanmaz. Kayıt olmanıza gerek yok.
          </p>
          <div className="ab-privacy-badge">
            <span className="ab-feat-dot" />
            <span>Sıfır veri depolama garantisi</span>
          </div>
        </div>
      </section>

      {/* ── FOR EVERY CAT OWNER ── */}
      <section className="ab-section">
        <div className="ab-container">
          <h2 className="ab-section-title">Her Kedi Sahibi İçin</h2>
          <p className="ab-text">
            Bu, yıllar önce sahip olmayı dilediğimiz bir araç. Şimdi, bunu merakı tüm kedi sahipleriyle paylaşıyoruz — belki de o sizsiniz.
          </p>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="ab-cta-section">
        <div className="ab-container ab-cta-inner">
          <p className="ab-cta-text">Kedinizin hikayesini keşfetmeye hazır mısınız?</p>
          <button className="btn-pink ab-cta-btn" onClick={() => navigate('/')}>
            Kedi Irkı Tanımlama Aracımızı Deneyin →
          </button>
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
              <Link to="/about">Hakkımızda</Link>
              <Link to="/privacy-policy">Gizlilik Politikası</Link>
              <Link to="/terms-of-service">Hizmet Şartları</Link>
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
};

export default AboutPage;
