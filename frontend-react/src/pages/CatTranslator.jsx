import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import '../App.css';
import './CatTranslator.css';

/* ── INTENT LABELS ── */
const INTENT_LABELS = {
  selamlama : 'Selamlama 👋',
  aclik      : 'Açlık 🍽️',
  oyun       : 'Oyun Modu 🎾',
  kizginlik  : 'Kızgınlık 😾',
  mutluluk   : 'Mutluluk 😸',
  soru       : 'Merak ❓',
  uzuntu     : 'Üzyüntü 😿',
};

/* ── INTENT DETECTION from mapping keywords ── */
function detectIntent(text, mapping) {
  if (!mapping) return null;
  const lower = text.toLowerCase();
  const kw = mapping.intent_keywords || {};
  for (const [intent, words] of Object.entries(kw)) {
    if (words.some(w => lower.includes(w))) return intent;
  }
  // fallback: index by text length
  const intents = Object.keys(mapping.mapping || {});
  return intents.length ? intents[lower.length % intents.length] : null;
}

/* ── PICK RANDOM WAV ── */
function pickSound(intent, mapping) {
  if (!mapping || !intent) return null;
  const files = mapping.mapping[intent];
  if (!files || files.length === 0) return null;
  return files[Math.floor(Math.random() * files.length)];
}

/* ── CAT TRANSLATOR LOGIC ── */
const CAT_RESPONSES = [
  { keywords: ['seviyorum','sev','aşk','öptüm'], response: '"Mrrrow... Purrr... Miyav miyav!" 🐾\nKediniz size karşılık veriyor: Beni de seviyorsun değil mi? Çünkü ben de seni çok seviyorum! Şimdi biraz çene ovuşturabilirsin.' },
  { keywords: ['acıktım','ac','yem','mama','besle'], response: '"MİYAV! Miyav miyav MİYAV!" 🐾\nKediniz şunu söylüyor: MAMA ZAMANI! Hemen şimdi! Neden bu kadar geç kaldın? Kasem dolmalı, su tazelenmiş — şimdi!!' },
  { keywords: ['neredesin','nerede','gel','geli'], response: '"Miyav... mrrrow... miyav?" 🐾\nKediniz merak ediyor: Neredeydin? Yokluğunu hissettim. Pencerenin önünde seni bekliyordum. Bir daha bu kadar uzun gitme.' },
  { keywords: ['oyna','oyun','top','tüy'], response: '"Miyav miyav! Prrrr miyav!" 🐾\nKediniz çılgına döndü: EEVEEET! Şimdi mi oynuyoruz?! O tüylü çubuğu getir! Ben hazırım — hiçbir zaman hazır olmaktan vazgeçmem!' },
  { keywords: ['güzel','şirin','tatlı','yakışıklı'], response: '"Purrr... mrrrow purrr..." 🐾\nKediniz biliyor: Evet, biliyorum. Ben başyapıtım. Bu gerçeği nihayet kabul ettiğin için teşekkürler. Bir kez daha söyleyebilirsin.' },
  { keywords: ['hasta','iyi değil','doktor','veteriner'], response: '"Miyav... miyav..." 🐾\nKediniz endişeyle bakıyor: Neden üzgün görünüyorsun? Yanına geliyorum. Mis gibi kokuyor musun? Seni iyileştirmeye çalışacağım.' },
  { keywords: ['merhaba','selam','hey','hello'], response: '"Miyav! Prrrr miyav miyav!" 🐾\nKediniz selamlamayı kabul ediyor: Ah, döndün mü sonunda. Saat kaç olduğunun farkında mısın? Neyse, hoş geldin. Şimdi beni besle.' },
  { keywords: ['uyku','uyu','yat','gece'], response: '"Purrr... mrrrow... zzz..." 🐾\nKediniz uykusunu söylüyor: Tam zamanı. Ben senin battaniyenin tam ortasını çoktan ayırdım. Sakın hareket etme — ve beni ittirme sabaha kadar.' },
];

const DEFAULT_RESPONSES = [
  '"Miyav miyav! Prrr..." 🐾\nKediniz cevaplıyor: Seni duyuyorum ama kelimelerini anlamak için biraz daha çaba göster. Belki daha sık miyavlamayı denersen anlaşabiliriz.',
  '"Mrrrow? Miyav miau..." 🐾\nKediniz düşünüyor: İlginç. Bunu sindirmem biraz zaman alacak. Şimdilik sadece oturuyorum ve değerlendiriyorum.',
  '"Prrr miyav miyav miau!" 🐾\nKediniz heyecanlandı: Bence bu harika bir fikir! Özellikle beni içeren kısım. Devam et lütfen.',
  '"Miyav... miyav miyav..." 🐾\nKediniz umursamaz bir şekilde bakıyor: Duydum. Şimdi güneş banyosuna devam edebilir miyim lütfen?',
];

function translateToCat(text) {
  if (!text.trim()) return '';
  const lowerText = text.toLowerCase();
  for (const item of CAT_RESPONSES) {
    if (item.keywords.some(k => lowerText.includes(k))) {
      return item.response;
    }
  }
  return DEFAULT_RESPONSES[text.length % DEFAULT_RESPONSES.length];
}

/* ── FAQ DATA ── */
const FAQ_ITEMS = [
  { q: 'Bu gerçek bir kedi dili çevirmeni mi?', a: 'Kedi Çevirmeni eğlenceli ve yapay zeka destekli bir araçtır. Gerçek dilbilimsel çeviri yapmaz, ancak kedinizin miyavlama kalıplarına ve araştırmalara dayalı olarak eğlenceli ve gerçekçi tepkiler üretir. Kedinizle bağ kurmanın eğlenceli bir yolu!' },
  { q: 'Yapay zekâ destekli kedi çevirmeni nasıl çalışıyor?', a: 'Uygulamamamız, girdilerinizi analiz etmek ve kedilerin belirli durumlarda nasıl tepki verdiğine dair bilinen kalıplara dayalı miyav tepkileri oluşturmak için yapay zeka algoritmaları kullanır.' },
  { q: 'Kedi çevirmeni ücretsiz mi?', a: 'Evet! Kedi Çevirmeni tamamen ücretsizdir. Kayıt olmanıza, uygulama indirmenize veya ödeme yapmanıza gerek yoktur. Sadece mesajınızı yazın ve kedinizin tepkisini anında görün.' },
  { q: 'Kedim acı çekiyorsa ne olacak? Uygulama yardımcı olabilir mi?', a: 'Kediniz acı çekiyor görünüyorsa, lütfen bir veterinere başvurun. Uygulamamamız eğlence amaçlıdır ve tıbbi tavsiye yerine geçmez. Her zaman profesyonel veteriner bakımını ön planda tutun.' },
  { q: 'Diğer kedi sesi çeviri araçlarından nasıl farklı?', a: 'Uygulamamamız, genel metin çeviri araçlarından farklı olarak tamamen kedilere odaklanmıştır. Kedi davranışı ve iletişim araştırmalarından ilham alan özel algoritmasıyla daha özgün ve eğlenceli sonuçlar üretir.' },
  { q: 'Kedi dili tercümanı nasıl kullanılır?', a: 'Çok basit! Kedinize söylemek istediğiniz şeyi metin kutusuna yazın, ardından "Kitty Vibe\'ı Görün" düğmesine tıklayın. Kedinizin söylediklerinize nasıl tepki vereceğini anında göreceksiniz.' },
  { q: '"Seni seviyorum" kelimesini kedi dili tercümanında nasıl kullanabilirim?', a: 'Sadece metin kutusuna "Seni seviyorum" yazın ve düğmeye tıklayın! Kedinizin bu sevgi dolu mesajınıza nasıl tepki vereceğini göreceksiniz. Çoğunlukla mutlu bir mırıldama veya baş ovuşturma davranışı bekleyebilirsiniz.' },
  { q: 'Kedi çevirmenler gerçekten işe yarıyor mu?', a: 'Kedi davranışı araştırmaları, kedilerin farklı sesler ve beden dili aracılığıyla iletişim kurduğunu göstermektedir. Uygulamamız bu araştırmalardan ilham alarak eğlenceli bir çeviri deneyimi sunar.' },
  { q: 'Kedi Sesi Çevirmeni ne kadar doğru?', a: 'Uygulamamız %100 kesinlikle doğru olduğunu iddia etmez — kediler hâlâ gizemli yaratıklar! Ancak bilinen kedi davranış kalıplarına dayalı olarak mümkün olan en gerçekçi ve eğlenceli yanıtları üretmeye çalışır.' },
];

/* ── TESTIMONIALS ── */
const TESTIMONIALS = [
  { stars: 5, rating: '4.9', role: '"Pazarlama Müdürü"', quote: 'Kedimle daha iyi iletişim kurmak için Kedi Çevirici\'yi kullanmaya başladım. Miyavlamalarının ne kadar gerçekçi çıktığı inanılmaz! Kedim oyun oynadığında veya acıktığında artık anlayabiliyorum. Bu, aramızdaki bağı daha da güçlendirdi.', author: 'Emily Carter', title: 'Pazarlama Müdürü' },
  { stars: 5, rating: '5.0', role: '"Yazılım Mühendisi"', quote: 'Bu uygulamanın ne kadar doğru olacağından emin değildim, ama beni gerçekten şaşırttı! Özellikle stresli olduğunda veya ilgi istediğinde kedimin duygularını anlamama yardımcı oluyor. Artık evcil hayvanımla her zamankinden daha fazla bağ kurduğumu hissediyorum.', author: 'Jack Henderson', title: 'Yazılım Mühendisi' },
  { stars: 5, rating: '4.8', role: '"Öğretmen"', quote: 'Bu uygulamayı kedimle her gün kullanıyorum ve çok eğlenceli! Genellikle acıktığında miyavlıyor ve şimdi tam olarak ne istediğini anlıyorum. Sadece eğlenceli değil, aynı zamanda ona daha iyi bakmama da yardımcı oluyor.', author: 'Sarah Lee', title: 'Öğretmen' },
  { stars: 5, rating: '4.7', role: '"Grafik Tasarımcı"', quote: 'Bir kedi sever olarak, bu uygulamayı doğruluğu nedeniyle çok seviyorum. Özellikle kedim garip davrandığında miyavlamalarını anlamama yardımcı oluyor. Eğlenceli bir araç olmasının yanı sıra, kedinizin mutlu mu yoksa hasta mı olduğunu öğrenmek istediğinizde de oldukça pratik.', author: 'David Turner', title: 'Grafik Tasarımcı' },
  { stars: 5, rating: '4.9', role: '"Evde Kalan Ebeveyn"', quote: 'Çocuklarım ve ben Kedi Tercümanı ile çok eğleniyoruz! Kedimizin duygularını anlamak için kullanıyoruz ve bu, eğlenceli bir aile aktivitesine dönüştü. Bu kadar basit bir araçın bizi evcil hayvanımıza daha yakın hissettirmesi inanılmaz.', author: 'Jessica Adams', title: 'Evde Kalan Ebeveyn' },
  { stars: 5, rating: '4.8', role: '"Veteriner Asistanı"', quote: 'Kedilerle günlük olarak çalıştığım için bu uygulamayı merak ediyordum. Miyavlama seslerinin ne kadar gerçekçi olduğuna hayran kaldım! Farklı kedi kişiliklerini daha iyi anlamama yardımcı oldu ve ziyaretler sırasında gergin kedileri sakinleştirmek için kullanışlı bir araç haline geldi.', author: 'Michael Chen', title: 'Veteriner Asistanı' },
];

export default function CatTranslator() {
  const [inputText, setInputText]     = useState('');
  const [output, setOutput]           = useState('');
  const [openFaq, setOpenFaq]         = useState(null);
  const [navScrolled, setNavScrolled] = useState(false);

  // ─ ML ses ─
  const [soundMapping, setSoundMapping]     = useState(null);
  const [currentSound, setCurrentSound]     = useState(null);
  const [detectedIntent, setDetectedIntent] = useState(null);
  const [isPlaying, setIsPlaying]           = useState(false);
  const [isLoading, setIsLoading]           = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const audioRef = useRef(null);

  useEffect(() => {
    fetch('/cat_sound_mapping.json')
      .then(r => r.json())
      .then(data => setSoundMapping(data))
      .catch(() => {}); // ses özellik opsiyonel — hata olursa sessizce geç
  }, []);

  useEffect(() => {
    const handleScroll = () => setNavScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleTranslate = () => {
    if (!inputText.trim()) return;
    setIsLoading(true);
    setOutput('');
    setCurrentSound(null);
    setIsPlaying(false);
    setTranslatedText('');

    setTimeout(() => {
      const savedText = inputText;
      setTranslatedText(savedText);
      setOutput(translateToCat(savedText));
      setIsLoading(false);

      if (soundMapping) {
        const intent = detectIntent(savedText, soundMapping);
        const wav    = pickSound(intent, soundMapping);
        if (wav) {
          setDetectedIntent(intent);
          setCurrentSound(wav);
        }
      }
    }, 1400);
  };

  const handlePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleNewSound = () => {
    if (!soundMapping || !detectedIntent) return;
    const wav = pickSound(detectedIntent, soundMapping);
    if (wav) {
      setCurrentSound(wav);
      setIsPlaying(false);
    }
  };

  return (
    <div className="ct-page">

      {/* ── NAVBAR ── */}
      <nav className={`navbar ${navScrolled ? 'scrolled' : ''}`}>
        <div className="nav-inner">
          <Link to="/" className="nav-logo">
            <img src="/kedi-ai-logo.svg" alt="Kedi.ai" className="nav-logo-img" /> Kedi.ai
          </Link>
          <ul className="nav-links">
            <li><Link to="/">Anasayfa</Link></li>
            <li><Link to="/cat-translator" className="active">Kedi Çevirmeni</Link></li>
            <li><Link to="/#how-it-works">Nasıl Çalışır</Link></li>
            <li><Link to="/#breeds-table">Kedi Irkları Tablosu</Link></li>
            <li><Link to="/#faq">SSS</Link></li>
          </ul>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="ct-hero">
        <div className="ct-hero-inner">
          <h1 className="ct-hero-title">Kedi Tercümanı Çevrimiçi</h1>
          <p className="ct-hero-desc">
            Bir kedi sahibi, kedisinin miyavlamalarını çözmek için Kedi Tercümanımızı kullandı. Uygulama,{' '}
            <a href="#">evcil hayvanların iletişim sorunlarını çözmeye</a> ve onları birbirine daha da yakınlaştırmaya yardımcı oldu.
          </p>
          <div className="ct-hero-img-wrap">
            <img
              src="https://whatbreedismycat.app/page/cat-translator.webp"
              alt="Kedi ve sahibi"
              className="ct-hero-img"
            />
          </div>
        </div>
      </section>

      {/* ── TRANSLATOR TOOL ── */}
      <section className="ct-tool-section">
        <div className="ct-tool-wrap">
          <div className="ct-tool-card">
            <div className="ct-tool-cols">
              <div className="ct-tool-col">
                <div className="ct-col-header">
                  <span>Sözleriniz</span>
                  <span className="ct-col-icon">👤</span>
                </div>
                <textarea
                  className="ct-textarea"
                  placeholder="Kedinize söylemek istediğiniz şeyi yazın..."
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  rows={5}
                />
              </div>
              <div className="ct-tool-col">
                <div className="ct-col-header">
                  <span>Kedi Sesi</span>
                  <span className="ct-col-icon">🐱</span>
                </div>
                <div className="ct-output">
                  {isLoading ? (
                    <p className="ct-output-loading">Kedilerin frekanslarına uyum sağlanıyor...</p>
                  ) : translatedText ? (
                    <>
                      <p className="ct-output-meow">Kedilerin miyavlamalarında “{translatedText}”...</p>
                      {currentSound && (
                        <>
                          <audio
                            ref={audioRef}
                            src={`/cat-sounds/${currentSound}`}
                            onEnded={() => setIsPlaying(false)}
                            onPlay={() => setIsPlaying(true)}
                            onPause={() => setIsPlaying(false)}
                          />
                          <button className="ct-btn-hear" onClick={handlePlay}>
                            {isPlaying ? '⏸️ Durdur' : 'Miyav sesini duy! 🔊'}
                          </button>
                        </>
                      )}
                    </>
                  ) : (
                    <p className="ct-output-placeholder">Kedinizin tepkisi burada görünecek!</p>
                  )}
                </div>
              </div>
            </div>
            <div className="ct-tool-footer">
              <button className="ct-btn-pink" onClick={handleTranslate}>
                Kedi Sesine Çevirin
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── MEOW TONE SECTION ── */}
      <section className="ct-content-section">
        <div className="ct-container ct-two-col">
          <div className="ct-text-col">
            <h2>Kedinin Miyavlama Ses Tonu İfadesi</h2>
            <p>
              Kediler duygularını ifade etmek için çeşitli sesler kullanırlar ve{' '}
              <a href="#">miyavlamalarının perdesi</a>, tonu ve süresi farklı mesajlar iletebilir.
              Örneğin, tiz ve kısa bir miyavlama genellikle açlığı gösterirken,{' '}
              <a href="#">alçak perdeli ve uzun bir miyavlama</a> hoşnutsuzluğu veya hayal kırıklığını
              ifade edebilir. Bu nüansları anlamak, kedinizin ihtiyaçlarını daha
              doğru bir şekilde yorumlamanıza ve daha güçlü bir bağ kurmanıza
              yardımcı olur. <a href="#">Kedi Tercümanımızı kullandığınızda</a>, her miyavlamanın
              ardındaki duygusal tonu, ister oyunbaz, ister meraklı, isterse de
              <a href="#"> talepkar olsun</a>, kolayca anlayabilirsiniz. Bu, evcil hayvanınızın
              dünyasına uyum sağlamanın ve buna göre yanıt vermenin harika bir
              yoludur; böylece kediniz duyulduğunu ve anlaşıldığını hisseder.
            </p>
          </div>
          <div className="ct-img-col">
            <div className="ct-img-single-wrap">
              <img src="https://whatbreedismycat.app/page/meow-pitch.webp" alt="Kedinin miyavlama ses tonu" className="ct-content-img" />
            </div>
          </div>
        </div>
      </section>

      {/* ── BODY LANGUAGE SECTION ── */}
      <section className="ct-content-section ct-bg-light">
        <div className="ct-container ct-two-col ct-reverse">
          <div className="ct-img-col">
            <div className="ct-img-single-wrap">
              <img src="https://whatbreedismycat.app/page/cat-body-language.webp" alt="Kedi vücut dili" className="ct-content-img" />
            </div>
          </div>
          <div className="ct-text-col">
            <h2>Kedilerin Yaygın Vücut Dili</h2>
            <p>
              Kediler, duygularını ifade etmek için{' '}
              <a href="#">kuyrukları</a>nı, kulaklarını, gözlerini,{' '}
              kokularını ve hatta dokunma duymularını kullanarak{' '}
              <a href="#">vücut diliyle</a> çok şey iletirler. Kuyruğunu sallamak rahatsızlığı gösterebilirken,{' '}
              <a href="#">kulakları rahatlamış</a> bir kedinin mırıldanması memnuniyeti gösterir. Bir kedi
              karnını gösterdiğinde, bu güven işaretidir, ancak dikkatli olun; bu pozisyon,{' '}
              kendilerini tehdit altında hissettiklerinde savunmacı bir tavır sergilediklerini de gösterebilir.
              Benzer şekilde, kediler bölgelerini işaretlemek veya sevgi göstermek için koku kullanırlar.
              Bu ince ipuçlarını tanımak, kedinizin ne hissettiğini daha iyi anlamanıza ve{' '}
              aranızdaki bağı güçlendirmenize yardımcı olabilir.
            </p>
          </div>
        </div>
      </section>

      {/* ── HOW TO USE ── */}
      <section className="ct-hiw-section">
        <div className="ct-container">
          <h2 className="ct-section-title">Kedi Çevirmen Uygulaması Nasıl Kullanılır?</h2>
          <p className="ct-section-desc">
            Kedi Tercümanımızı kullanmak çok kolay! Kedinizin miyavlamalarını anlamaya başlamak ve tüylü dostunuzla
            aranızdaki bağı güçlendirmek için bu kolay adımları izleyin. Eğlenceli, ücretsiz ve kullanımı kolay!
          </p>
          <div className="ct-hiw-grid">
            <div className="ct-steps">
              {[
                { n: 'Adım 1 :', icon: '📩', title: 'Mesajınızı Girin', desc: 'Kedinize söylemek istediğiniz şeyi giriş kutusuna yazın. "Seni seviyorum" veya "Aç mısın?" gibi basit bir şey olabilir. Mesaj ne kadar net olursa, miyavlama sesine çevirisi de o kadar doğru olur.' },
                { n: 'Adım 2 :', icon: '🎵', title: 'Çeviriyi Dinleyin', desc: '"Çevir" düğmesine basın ve uygulama, girdiğiniz metne göre anında gerçekçi bir miyavlama sesi oluşturacaktır. İster neşeli, ister sevgi dolu, ister meraklı olsun, yazdığınız duyguya uygun bir ses duyacaksınız.' },
                { n: '3. Adım :', icon: '📤', title: 'Miyavınızı Paylaşın', desc: 'Arkadaşlarınızı veya diğer kedi severlerle etkilemek mi istiyorsunuz? Miyav çevirinizi anında başkalarıyla paylaşabilirsiniz. Paylaş düğmesine dokunarak sosyal medyada yayınlayın veya özel birine gönderin.' },
                { n: 'Adım 4 :', icon: '📥', title: 'İndirin ve Kaydedin', desc: 'Eğlencenin devam etmesini istiyorsanız, çevrilmiş miyav sesini indirmeniz yeterli. Sesleri daha sonra kullanmak üzere kaydedebilir, tekrar paylaşabilir veya kedinizle geçirdiğiniz eğlenceli miyav anlarından oluşan bir koleksiyon oluşturabilirsiniz.' },
              ].map(step => (
                <div className="ct-step-row" key={step.n}>
                  <div className="ct-step-icon">{step.icon}</div>
                  <div className="ct-step-body">
                    <h3><span className="ct-step-num">{step.n}</span> {step.title}</h3>
                    <p>{step.desc}</p>
                  </div>
                </div>
              ))}
              <button className="ct-btn-pink" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
                Kedi Çevirici'yi Şimdi Deneyin 🐾
              </button>
            </div>
            <div className="ct-hiw-mockup">
              <div className="ct-hiw-mock-card">
                <div className="ct-hiw-mock-cols">
                  <div className="ct-hiw-mock-col">
                    <div className="ct-hiw-mock-header">
                      <span>Sözleriniz</span>
                      <span className="ct-hiw-mock-icon">👤</span>
                    </div>
                    <div className="ct-hiw-mock-input">Selam</div>
                  </div>
                  <div className="ct-hiw-mock-col">
                    <div className="ct-hiw-mock-header">
                      <span>Kedi Sesi</span>
                      <span className="ct-hiw-mock-icon">🐱</span>
                    </div>
                    <div className="ct-hiw-mock-output">
                      Kedilerin miyavlamalarında “Selam”...
                    </div>
                    <button className="ct-hiw-mock-hear">Miyav sesini duy! 🔊</button>
                  </div>
                </div>
                <div className="ct-hiw-mock-footer">
                  <button className="ct-hiw-mock-btn-pink">Kedi Sesine Çevirin</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── FEATURE SECTIONS HEADER ── */}
      <section className="ct-features-header">
        <div className="ct-container">
          <h2 className="ct-section-title">Mutlu Anlarda Çevrimiçi Kedi Çevirmenini Kullanın</h2>
          <p className="ct-section-desc">
            <a href="#">Kedi çeviri</a> uygulamamızın tüylü dostunuzla geçirdiğiniz özel anlara nasıl neşe ve bağ kattığını keşfedin.
          </p>
        </div>
      </section>

      {/* ── FEATURE 1 ── */}
      <section className="ct-feat-section">
        <div className="ct-container ct-two-col">
          <div className="ct-text-col">
            <h2>Oyun Zamanında Kedinizle Bağ Kurun</h2>
            <p>
              Kedinizle eğlenceli bir oyun seansı geçirdiğinizi hayal edin, ancak
              artık ne istediğini tahmin etmek yerine biliyorsunuz. İster oyuncak
              kovalamak, ister saklambaç oynamak, isterse de sadece kanepede
              rahatlamak olsun, kedinizin miyavlamalarını anlamak,{' '}
              <a href="#">etkileşiminize ekstra neşe ve derinlik katıyor</a>.{' '}
              <a href="#">Kedi Tercümanımızla</a>, oyunbaz, meraklı veya hatta sinirli miyavlamalarını hassas bir şekilde duyacaksınız.
              Buna göre yanıt verebilir, oyun zamanını daha heyecanlı ve etkileşimli hale getirebilirsiniz.
              Kedinizin oyunbaz miyavlamalarını{' '}
              <a href="#">artık daha bağlantılı hissettirmesine</a> ve onlarla daha anlamlı bir şekilde etkileşim kurabilmeme bayılıyorum,
              hepsi bu araç sayesinde.
            </p>
            <button className="ct-btn-pink-outline">Kedi Çevirici'yi Şimdi Deneyin →</button>
          </div>
          <div className="ct-img-col">
            <img src="https://whatbreedismycat.app/cat/siamese.webp" alt="Oyun zamanı" className="ct-feat-img" />
          </div>
        </div>
      </section>

      {/* ── FEATURE 2 ── */}
      <section className="ct-feat-section ct-bg-white">
        <div className="ct-container ct-two-col ct-reverse">
          <div className="ct-img-col">
            <img src="https://whatbreedismycat.app/cat/persian.webp" alt="Kaygı" className="ct-feat-img" />
          </div>
          <div className="ct-text-col">
            <h2>Kedinizin Kaygısını Yatıştırmak</h2>
            <p>
              Kediniz işe giderken veya geceleyin sürekli miyavlıyor mu? Kedi
              Tercümanı ile, <a href="#">sürekli miyavlamalarının ardındaki nedeni</a> nihayet
              anlayabilirsiniz. Yalnız mı, aç mı yoksa sadece meraklı mı? Uygulama,
              duygularını ve ihtiyaçlarını çözmenize yardımcı olarak davranışlara
              dair fikir veriyor. Bu, endişelerini gidermeyi kolaylaştırarak hem sizin
              hem de kediniz için stresi azaltıyor. Kedim huzursuz olduğunda,
              sadece ilgiye mi ihtiyacı olduğunu yoksa bir sorun mu olduğunu
              anlayabilmek bana huzur veriyor. <a href="#">Ayrı olsak bile bizi birbirimize
              yaklaştıran bir köprü gibi</a>.
            </p>
            <button className="ct-btn-pink-outline">Kedi Çevirici'yi Şimdi Deneyin →</button>
          </div>
        </div>
      </section>

      {/* ── FEATURE 3 ── */}
      <section className="ct-feat-section">
        <div className="ct-container ct-two-col">
          <div className="ct-text-col">
            <h2>Arkadaşlarla Eğlenceli Miyavlamalar Paylaşmak</h2>
            <p>
              Kedinizin komik tepkilerini sergilemek mi istiyorsunuz? Kedi
              Tercümanı ile komik miyavlamalar yapabilir ve bunları
              arkadaşlarınızla, <a href="#">ailenizle veya sosyal medyada paylaşabilirsiniz</a>. Bu
              uygulama, ister aptalca bir "merhaba" ister dramatik bir "lütfen beni
              besle!" olsun, gününüze mizah ve eğlence katmanın harika bir
              yoludur. Kedinizin miyavlamalarını kaydedebilir, indirebilir ve anında{' '}
              <a href="#">iyi bir kahkahayı seven herkesle paylaşabilirsiniz</a>. Özellikle kedilerin ne
              kadar aptalca ve eğlenceli olabileceğine hepimiz birlikte
              güldüğümüzde, bu anları diğer kedi severlerle paylaşmaktan
              gerçekten keyif alıyorum. Anında moral yükseltici!
            </p>
            <button className="ct-btn-pink-outline">Kedi Çevirici'yi Şimdi Deneyin →</button>
          </div>
          <div className="ct-img-col">
            <img src="https://whatbreedismycat.app/cat/british-shorthair.webp" alt="Paylaşım" className="ct-feat-img" />
          </div>
        </div>
      </section>

      {/* ── WHY CHOOSE US ── */}
      <section className="ct-trust-section">
        <div className="ct-container">
          <h2 className="ct-section-title">Kedi Tercümanımızı Neden Seçmelisiniz?</h2>
          <div className="ct-trust-grid">
            {[
              { icon: '🎁', title: 'Kullanımı Basit ve Ücretsiz', desc: 'Kedi Çeviricimiz tamamen ücretsizdir ve hiçbir gizli ücret içermez. Eğlencenin tadını çıkarmak için kayıt olmanıza veya herhangi bir şey indirmenize gerek yok! Sadece mesajınızı yazın ve çevirisini anında dinleyin. Basit, hızlı ve her zaman erişilebilir.' },
              { icon: '🗄️', title: 'Doğru Veri Çevirisi', desc: 'Yapay zekâ destekli sistemimiz mesajınızı analiz eder ve amaçlanan duyguyu yansıtan doğru miyavlamalara dönüştürür. İster oyunbaz ister ciddi olsun, kedinizin duygularına uygun miyavlamalar duyacaksınız, bu da deneyimi gerçekçi ve keyifli hale getirecektir.' },
              { icon: '🛡️', title: 'Gizlilik ve Veri Güvenliği', desc: 'Gizliliğinizi ciddiye alıyoruz. Bilgileriniz asla saklanmaz veya üçüncü taraflarla paylaşılmaz. Veri güvenliğinize öncelik vererek, Kedi Çevirmeni uygulamasını gönül rahatlığıyla kullanabilmenizi sağlıyoruz.' },
              { icon: '⚡', title: 'Gerçekçi Ses Deneyimi', desc: 'Gelişmiş ses sentezi teknolojisiyle çalışan Kedi Tercümanımız, yapay zeka algoritmalarına dayalı olarak en otantik kedi miyavlamalarını üretir. Kedinizin duygularının özünü gerçeğe yakın seslerle deneyimleyin ve evcil hayvanınızla etkileşimi daha anlamlı hale getirin.' },
            ].map((item, i) => (
              <div className="ct-trust-card" key={i}>
                <div className="ct-trust-icon">{item.icon}</div>
                <h3>{item.title}</h3>
                <p>{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── TESTIMONIALS ── */}
      <section className="ct-testi-section">
        <div className="ct-container">
          <h2 className="ct-section-title">Kullanıcılar Kedi Çevirmeni Online Hakkında Ne Diyor?</h2>
          <div className="ct-testi-grid">
            {TESTIMONIALS.map((t, i) => (
              <div className="ct-testi-card" key={i}>
                <div className="ct-testi-stars">
                  {'★'.repeat(t.stars)} <span className="ct-testi-rating">{t.rating}</span>
                </div>
                <p className="ct-testi-role">{t.role}</p>
                <p className="ct-testi-quote">{t.quote}</p>
                <p className="ct-testi-author">{t.author}</p>
                <p className="ct-testi-title">{t.title}</p>
              </div>
            ))}
          </div>
          <div className="ct-center">
            <button className="ct-btn-pink" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
              Kedi Çevirici'yi Şimdi Deneyin 🐾
            </button>
          </div>
        </div>
      </section>

      {/* ── FAQ ── */}
      <section className="ct-faq-section">
        <div className="ct-container">
          <h2 className="ct-section-title">Kedi Sesi Çevirmeni için Sıkça Sorulan Sorular</h2>
          <div className="ct-faq-list">
            {FAQ_ITEMS.map((item, i) => (
              <div className={`ct-faq-item ${openFaq === i ? 'open' : ''}`} key={i}>
                <button className="ct-faq-q" onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                  <span>{item.q}</span>
                  <span className="ct-faq-icon">{openFaq === i ? '−' : '+'}</span>
                </button>
                {openFaq === i && <div className="ct-faq-a">{item.a}</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="ct-cta-section">
        <div className="ct-container ct-cta-inner">
          <h2>Kedinizin miyavlamalarını bugünden anlamaya başlayın!</h2>
          <p>
            Ücretsiz çevrimiçi{' '}
            <a href="#">Kedi Tercümanımızla</a>, kedinizin miyavlamalarını nihayet
            anlayabilir ve aranızdaki bağı güçlendirebilirsiniz. Kayıt olmaya gerek
            kalmadan miyavlamaları anında tercüme edin!
          </p>
          <button className="ct-btn-pink ct-btn-lg" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            Kedi Çevirici'yi Şimdi Deneyin ↑
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
              <Link to="/cat-translator">Kedi Çevirmeni</Link>
            </div>
            <div className="footer-col">
              <h4>Hakkında</h4>
              <Link to="/about">Hakkımızda</Link>
              <Link to="/privacy-policy">Gizlilik Politikası</Link>
              <Link to="/terms-of-service">Hizmet Şartları</Link>
            </div>
            <div className="footer-col">
              <h4>Temas etmek</h4>
              <a href="mailto:merhaba@kedimincins.ne">merhaba@kedimincins.ne</a>
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
