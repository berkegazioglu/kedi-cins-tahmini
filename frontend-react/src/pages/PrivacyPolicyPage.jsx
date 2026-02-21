import React from 'react';
import { Link } from 'react-router-dom';
import './PrivacyPolicyPage.css';

const PrivacyPolicyPage = () => {
  return (
    <div className="pp-page">

      {/* ── NAVBAR ── */}
      <nav className="navbar">
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
      <section className="pp-hero">
        <div className="pp-container">
          <h1 className="pp-title">Gizlilik Politikası</h1>
          <p className="pp-effective">Yürürlük Tarihi: 22 Şubat 2026</p>
        </div>
      </section>

      {/* ── CONTENT ── */}
      <section className="pp-content">
        <div className="pp-container">

          <p className="pp-intro">
            Kedi.ai'yi kullandığınız için teşekkürler. Gizliliğinize önem veriyor ve verilerinizi şeffaf, saygılı ve güvenli bir şekilde işlemeyi taahhüt ediyoruz.
          </p>
          <p className="pp-intro">
            Bu Gizlilik Politikası, bilgilerinizi nasıl topladığımızı, işlediğimizi ve koruduğumuzu açıklar. Ayrıca Genel Veri Koruma Yönetmeliği (GDPR) ve diğer geçerli gizlilik yasaları kapsamındaki haklarınızı da açıklıyoruz.
          </p>

          <div className="pp-section">
            <h2 className="pp-section-title">1. Biz Kimiz</h2>
            <p>Kedi.ai, YZ destekli kedi cinsi analizi sunmaya odaklanmış bağımsız bir geliştirme ekibi tarafından sahiplenilmekte ve işletilmektedir. GDPR amaçları doğrultusunda Veri Sorumlusu olarak hareket ediyoruz.</p>
            <div className="pp-contact-box">
              <p><strong>Veri Sorumlusu İletişim Bilgileri:</strong></p>
              <p>E-posta: <a href="mailto:info@kedimhangicins.com">info@kedimhangicins.com</a></p>
              <p>Web sitesi: <a href="https://berkegazioglu-kedi-ai.hf.space" target="_blank" rel="noreferrer">berkegazioglu-kedi-ai.hf.space</a></p>
            </div>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">2. Hangi Kişisel Verileri Topluyoruz</h2>
            <p>Gönüllü olarak bize ulaşmadıkça adınız, e-posta adresiniz veya konumunuz gibi herhangi bir kişisel bilgi toplamıyor veya talep etmiyoruz.</p>
            <p>Kedi cinsi belirleme aracımızı kullanırken aşağıdaki bilgileri geçici olarak işliyoruz:</p>
            <ul className="pp-list">
              <li>Kedinizin fotoğrafı (yalnızca YZ analizi için)</li>
              <li>Tarayıcı türü, cihaz türü ve anonimleştirilmiş IP gibi temel teknik veriler (analitik için)</li>
            </ul>
            <p>Bu verileri herhangi bir kullanıcı kimliğiyle ilişkilendirmiyoruz.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">3. Verilerinizi Nasıl Kullanıyoruz</h2>
            <p>Fotoğrafınızı yalnızca şunlar için kullanıyoruz:</p>
            <ul className="pp-list">
              <li>YZ kullanarak olası kedi ırklarını tespit etmek</li>
              <li>Görüntülemeniz veya paylaşmanız için ekranda sonuçlar oluşturmak</li>
            </ul>
            <p>Teknik veriler (cihaz/tarayıcı bilgisi gibi) yalnızca şunlar için kullanılır:</p>
            <ul className="pp-list">
              <li>Site performansını izlemek</li>
              <li>Kullanılabilirliği geliştirmek</li>
              <li>Hizmeti kötüye kullanım veya istismardan korumak</li>
            </ul>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">4. Verilerinizi Ne Kadar Süre Saklıyoruz</h2>
            <p>Yüklenen fotoğraflar için katı bir saklama yok politikası uyguluyoruz:</p>
            <ul className="pp-list">
              <li>Kedi fotoğrafınız gerçek zamanlı olarak işlenir</li>
              <li>Sonuçlar oluşturulduktan hemen sonra otomatik olarak silinir</li>
              <li>Fotoğrafınızın yedeğini, arşivini veya kopyasını saklamıyoruz</li>
            </ul>
            <p>Analitik veriler (kişisel olmayan) performans takibi için sınırlı bir süre saklanabilir.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">5. İşlemenin Hukuki Dayanakları (GDPR Madde 6)</h2>
            <p>Verileri aşağıdaki hukuki dayanaklar kapsamında işliyoruz:</p>
            <ul className="pp-list">
              <li><strong>Rıza</strong> — Irk analizi için fotoğraf yükleyerek bize izin veriyorsunuz</li>
              <li><strong>Meşru Menfaat</strong> — Hizmeti verimli ve güvenli şekilde işletmek ve geliştirmek için kişisel olmayan teknik verileri işliyoruz</li>
            </ul>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">6. GDPR Kapsamındaki Haklarınız</h2>
            <p>Avrupa Ekonomik Alanı'nda (AEA) bulunuyorsanız aşağıdaki haklara sahipsiniz:</p>
            <ul className="pp-list">
              <li><strong>Erişim</strong> — Kişisel verilerinizin (varsa) bir kopyasını talep etmek</li>
              <li><strong>Düzeltme</strong> — Hatalı bilgileri düzeltmemizi istemek</li>
              <li><strong>Silme</strong> — Verilerinizin silinmesini talep etmek ("unutulma hakkı")</li>
              <li><strong>Kısıtlama</strong> — Belirli yollarla verilerinizi kullanmayı durdurmamızı istemek</li>
              <li><strong>İtiraz</strong> — Meşru menfaate dayalı veri kullanımına itiraz etmek</li>
              <li><strong>Veri Taşınabilirliği</strong> — Verilerinizi makine tarafından okunabilir formatta almak</li>
              <li><strong>Rızayı Geri Çekme</strong> — Geçmiş işlemler etkilenmeksizin her zaman</li>
            </ul>
            <p>Haklarınızı kullanmak için lütfen şu adrese ulaşın: <a href="mailto:info@kedimhangicins.com">info@kedimhangicins.com</a></p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">7. Otomatik Karar Verme</h2>
            <p>YZ aracımız, kedinizin fiziksel özelliklerini analiz etmek ve olası ırkları önermek için otomatik karar verme kullanır.</p>
            <ul className="pp-list">
              <li>Bu tamamen otomatiktir ve görsel karşılaştırmaya dayalıdır</li>
              <li>Yasal veya önemli etkilere yol açmaz</li>
              <li>Yalnızca bilgilendirme ve eğitim amaçlıdır</li>
            </ul>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">8. Çerezler ve Analitik</h2>
            <p>Aşağıdakiler için minimal çerezler kullanıyoruz:</p>
            <ul className="pp-list">
              <li>Temel web sitesi işlevselliğini etkinleştirmek</li>
              <li>IP anonimleştirmesiyle gizlilik dostu araçlar kullanarak anonim trafik istatistikleri toplamak</li>
            </ul>
            <p>Çerezleri şunlar için <strong>kullanmıyoruz</strong>:</p>
            <ul className="pp-list">
              <li>Reklam</li>
              <li>Davranışsal izleme</li>
              <li>Üçüncü taraflara veri satışı</li>
            </ul>
            <p>Çerezleri tarayıcı ayarlarınız aracılığıyla kontrol edebilirsiniz.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">9. Çocukların Gizliliği</h2>
            <p>Hizmetimiz genel kitlelere yöneliktir. 13 yaşın altındaki çocuklardan kasıtlı olarak veri toplamıyoruz. Bir çocuktan istemeden veri topladığımızın farkına varırsak, bunu derhal sileriz.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">10. Uluslararası Veri Transferleri</h2>
            <p>Sunucularımız AEA dışında bulunabilir. Bu durumda verilerin şu şekilde olmasını sağlıyoruz:</p>
            <ul className="pp-list">
              <li>Standart Sözleşme Maddeleri (SCC) kullanılarak transfer edilir</li>
              <li>Güvenli ve GDPR'a uygun şekilde işlenir</li>
              <li>Geçici kullanımın hemen ardından silinir</li>
            </ul>
            <p>Uluslararası sunucularda hiçbir kişisel veri kalıcı olarak saklanmaz.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">11. Veri Güvenliği Önlemleri</h2>
            <p>Veri güvenliğinizi ciddiye alıyoruz. Şunları kullanıyoruz:</p>
            <ul className="pp-list">
              <li>Tüm iletişim için HTTPS şifrelemesi</li>
              <li>Geçici, güvenli fotoğraf işleme</li>
              <li>Hassas bilgilerin kalıcı olarak saklanmaması</li>
              <li>Sistemlerimizin ve süreçlerimizin düzenli denetimleri</li>
            </ul>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">12. Üçüncü Taraf Hizmetleri</h2>
            <p>Verilerinizi reklam veya pazarlama amacıyla üçüncü taraflarla paylaşmıyoruz.</p>
            <p>Gizlilik düzenlemelerine uyan katı anlaşmalar kapsamında dikkatlice seçilmiş üçüncü taraf hizmetler (ör. bulut işleme veya analitik platformlar) kullanabiliriz.</p>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">13. Bu Politikadaki Değişiklikler</h2>
            <p>Hizmet güncellemelerini veya yasal değişiklikleri yansıtmak için bu politikayı zaman zaman güncelleyebiliriz. Bunu yaptığımızda:</p>
            <ul className="pp-list">
              <li>"Son Güncelleme" tarihini güncelleyeceğiz</li>
              <li>Web sitesi altbilgisi veya açılır pencere aracılığıyla kullanıcıları bildirebiliriz</li>
              <li>Hizmeti kullanmaya devam etmek, güncellenmiş politikayı kabul ettiğiniz anlamına gelir</li>
            </ul>
          </div>

          <div className="pp-section">
            <h2 className="pp-section-title">14. Bize Ulaşın</h2>
            <p>Bu Gizlilik Politikası veya verileri nasıl işlediğimiz hakkında sorularınız varsa lütfen bize ulaşın:</p>
            <div className="pp-contact-box">
              <p>E-posta: <a href="mailto:info@kedimhangicins.com">info@kedimhangicins.com</a></p>
              <p>Web sitesi: <a href="https://berkegazioglu-kedi-ai.hf.space" target="_blank" rel="noreferrer">berkegazioglu-kedi-ai.hf.space</a></p>
            </div>
            <p>GDPR ile ilgili talepler başta olmak üzere 30 gün içinde yanıt vereceğiz.</p>
            <p><strong>Kedi.ai'ye güvendiğiniz için teşekkürler. Gizliliğiniz bizim önceliğimizdir.</strong></p>
          </div>

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
};

export default PrivacyPolicyPage;
