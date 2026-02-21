import React from 'react';
import { Link } from 'react-router-dom';
import './TermsOfServicePage.css';

const TermsOfServicePage = () => {
  return (
    <div className="tos-page">

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
      <section className="tos-hero">
        <div className="tos-container">
          <h1 className="tos-title">Hizmet Şartları</h1>
          <p className="tos-effective">Yürürlük Tarihi: 22 Şubat 2026</p>
        </div>
      </section>

      {/* ── CONTENT ── */}
      <section className="tos-content">
        <div className="tos-container">

          <p className="tos-intro">
            Kedi.ai'ye hoş geldiniz. Bu Hizmet Şartları ("Şartlar"), web sitemize ve hizmetlerimize erişiminizi ve kullanımınızı düzenler. Sitemizi kullanarak bu Şartları kabul etmiş olursunuz. Kabul etmiyorsanız lütfen web sitesini kullanmayın.
          </p>
          <p className="tos-intro">Lütfen dikkatlice okuyun.</p>

          <div className="tos-section">
            <h2 className="tos-section-title">1. Kedi.ai Hakkında</h2>
            <p>Kedi.ai, kullanıcıların bir kedi fotoğrafı yüklemesine ve olası ırklarının tahmini hakkında bilgi almasına olanak tanıyan yapay zeka destekli bir web aracıdır. Araç, Uluslararası Kedi Birliği (TICA) gibi saygın kaynaklardaki verilere göre eğitilmiş bilgisayar görüşü ve makine öğrenmesi modelleri kullanır.</p>
            <p>Hizmetimiz kayıt olmadan ücretsiz olarak sunulmakta olup yalnızca bilgilendirme ve eğitim amaçlıdır.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">2. Uygunluk</h2>
            <p>Bu web sitesini şu durumlarda kullanabilirsiniz:</p>
            <ul className="tos-list">
              <li>En az 13 yaşındasınız ya da ebeveyn veya yasal vasi onayına sahipsiniz</li>
              <li>Bu Şartlara uymayı kabul ediyorsunuz</li>
              <li>Hizmeti yalnızca kişisel, ticari olmayan amaçlarla kullanıyorsunuz</li>
              <li>Bu tür hizmetleri yasaklayan bir yargı bölgesinde ikamet etmiyorsunuz</li>
            </ul>
            <p>Bu koşulların karşılanmadığına inanırsak erişimi engelleyebilir veya kaldırabiliriz.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">3. Hizmetin Açıklaması</h2>
            <p>Kedi.ai ile şunları yapabilirsiniz:</p>
            <ul className="tos-list">
              <li>Kedinizin fotoğrafını yüklemek (PNG, JPG, JPEG veya WEBP, max 10MB)</li>
              <li>Her biri şunları içeren ilk 3 olası kedi ırkını gösteren yapay zeka tarafından oluşturulmuş sonucu otomatik olarak almak:
                <ul className="tos-list tos-sublist">
                  <li>Eşleşme yüzdesi</li>
                  <li>Kısa ırk özellikleri (kişilik, fiziksel özellikler)</li>
                  <li>Genel bakım ipuçları</li>
                </ul>
              </li>
              <li>Sonucunuzu indirmek veya paylaşmak</li>
            </ul>
            <div className="tos-note">
              <p><strong>Lütfen Dikkat:</strong></p>
              <ul className="tos-list">
                <li>Irk sonuçları genetik testlere değil, görsel benzerliklere dayalı tahminlerdir.</li>
                <li>Yapay zekamız saf ırk statüsünü veya soyunu doğrulayamaz.</li>
                <li>Kesin sonuçlar için Basepaws veya Wisdom Panel gibi üçüncü taraf DNA test sağlayıcılarını öneririz.</li>
              </ul>
            </div>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">4. Web Sitesini Kullanımınız</h2>
            <p>Kedi.ai'yi yalnızca yasal ve saygılı amaçlar için kullanmayı kabul ediyorsunuz. Şunları yapmamalısınız:</p>
            <ul className="tos-list">
              <li>İnsan içeren, telif hakkıyla korunan veya başka hayvanların fotoğraflarını yüklemek</li>
              <li>Saldırgan, açık, şiddet içeren veya yasadışı içerik göndermek</li>
              <li>Yapay zeka hizmetini tersine mühendislik yapmaya, bozmaya veya istismar etmeye çalışmak</li>
              <li>Irk sonuçlarını tahrif ederek başkalarını yanıltmak</li>
            </ul>
            <p>Hizmet kötüye kullanılırsa içeriği kaldırma, kullanıcıları engelleme veya erişimi kısıtlama hakkını saklı tutarız.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">5. Yüklenen İçerik</h2>
            <p>Bir fotoğraf yükleyerek şunları kabul etmiş olursunuz:</p>
            <ul className="tos-list">
              <li>Fotoğrafın sahibisiniz ya da kullanma iznine sahipsiniz</li>
              <li>Yapay zekamızı kullanarak görüntüyü işlememiz için geçici, sınırlı izin veriyorsunuz</li>
              <li>Fotoğrafınız işlemden sonra sistemlerimizden otomatik olarak silinecektir</li>
              <li>Yüklenen fotoğrafınızı hiçbir amaçla saklamayacağız, kaydetmeyeceğiz, yeniden kullanmayacağız veya paylaşmayacağız</li>
            </ul>
            <p>Daha fazla bilgi için <Link to="/privacy-policy">Gizlilik Politikamızı</Link> inceleyin.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">6. Fikri Mülkiyet</h2>
            <p>Yapay zeka aracı, ırk açıklamaları ve kedi ırkı tablosu dahil olmak üzere bu web sitesindeki tasarım, özellikler, marka ve içerik Kedi.ai'nin fikri mülkiyetidir.</p>
            <p>Şunları yapamazsınız:</p>
            <ul className="tos-list">
              <li>Web sitesinin herhangi bir bölümünü kopyalamak, çoğaltmak veya değiştirmek</li>
              <li>Yapay zeka sonuçlarını veya ırk tablosunu satmak, yeniden satmak veya yeniden dağıtmak</li>
              <li>Marka veya varlıklarımızı yazılı izin olmadan kullanmak</li>
            </ul>
            <p>Kendi sonuçlarınızı kişisel, ticari olmayan amaçlarla (ör. sosyal medyada) paylaşabilirsiniz.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">7. Gizlilik ve Veri Koruma</h2>
            <p>Gizliliği ciddiye alıyoruz. Kısa bir özet:</p>
            <ul className="tos-list">
              <li>Yüklenen fotoğrafları saklamıyor veya kaydetmiyoruz</li>
              <li>Hesap oluşturma gerektirmiyor veya kişisel veri toplamıyoruz</li>
              <li>Web sitesi işlevselliği ve anonim analizler için temel çerezler kullanıyoruz</li>
              <li>GDPR ve diğer gizlilik düzenlemelerine uyuyoruz</li>
            </ul>
            <p>Daha fazlası için <Link to="/privacy-policy">Gizlilik Politikamızı</Link> inceleyin.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">8. Garanti Reddi</h2>
            <p>Kedi.ai "olduğu gibi" ve "mevcut olduğu şekilde" sunulmaktadır. Şunları garanti etmiyoruz:</p>
            <ul className="tos-list">
              <li>Irk tanımlama sonuçlarının doğruluğu</li>
              <li>Hizmetin hatasız, kesintisiz veya her zaman erişilebilir olacağı</li>
              <li>Sonuçların resmi DNA veya yetiştirici kayıtlarıyla eşleşeceği</li>
            </ul>
            <p>Aracı kullanmak kendi takdirinize ve riskinize bağlıdır.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">9. Sorumluluk Sınırlaması</h2>
            <p>Yasaların izin verdiği azami ölçüde, Kedi.ai şunlardan sorumlu tutulamaz:</p>
            <ul className="tos-list">
              <li>Irk tahminlerindeki hatalar</li>
              <li>Veri veya görüntü kaybı</li>
              <li>Yapay zeka sonuçlarına dayanılarak alınan kararlar</li>
              <li>Hizmeti kullanmaktan veya hizmete güvenmekten kaynaklanan zarar veya hasarlar</li>
            </ul>
            <p>Bu araç yalnızca eğitim ve eğlence amaçlıdır.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">10. Üçüncü Taraf Bağlantıları</h2>
            <p>Sitemiz zaman zaman şunlar gibi güvenilir üçüncü taraf hizmetlere bağlantı verebilir:</p>
            <ul className="tos-list">
              <li>DNA test sağlayıcıları (ör. Basepaws, Wisdom Panel)</li>
              <li>Irk standardı kaynakları (ör. TICA)</li>
            </ul>
            <p>Bu bağlantılar yalnızca kolaylık amacıyla sağlanmaktadır. İçerikleri, şartları veya gizlilik politikalarından sorumlu değiliz.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">11. Hizmette Değişiklikler</h2>
            <p>Web sitesinin veya özelliklerin bölümlerini önceden bildirimde bulunmaksızın istediğimiz zaman güncelleyebilir, duraklatabilir veya durdurabilirriz. Buna yapay zeka modeli, düzen, ırk veritabanı veya sonuç formatındaki güncellemeler dahildir.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">12. Bu Şartlardaki Değişiklikler</h2>
            <p>Bu Şartları zaman zaman gözden geçirebiliriz. Bunu yaptığımızda:</p>
            <ul className="tos-list">
              <li>Üst kısımdaki "Son Güncelleme" tarihini güncelleyeceğiz</li>
              <li>Siteyi kullanmaya devam etmek, yeni sürümü kabul ettiğiniz anlamına gelir</li>
            </ul>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">13. Geçerli Hukuk</h2>
            <p>Bu Şartlar, çatışma hukuku kuralları gözetilmeksizin Türkiye Cumhuriyeti yasalarına tabidir. Tüm uyuşmazlıklar yetkili Türk mahkemelerinde çözüme kavuşturulur.</p>
          </div>

          <div className="tos-section">
            <h2 className="tos-section-title">14. Bize Ulaşın</h2>
            <p>Bu Şartlar, hizmetimiz veya başka herhangi bir konuda sorularınız varsa lütfen bize ulaşın:</p>
            <div className="tos-contact-box">
              <p>📧 E-posta: <a href="mailto:info@kedimhangicins.com">info@kedimhangicins.com</a></p>
              <p>🌐 Web sitesi: <a href="https://berkegazioglu-kedi-ai.hf.space" target="_blank" rel="noreferrer">berkegazioglu-kedi-ai.hf.space</a></p>
            </div>
            <p><strong>Kedi.ai'yi kullandığınız için teşekkürler! Kedinizin benzersiz özelliklerinin ardındaki hikayeyi keşfetmenize yardımcı olmak için buradayız. 🐱</strong></p>
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

export default TermsOfServicePage;
