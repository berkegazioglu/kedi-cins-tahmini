"""
app_resnet50.py

Streamlit web uygulaması - ResNet-50 ile kedi cinsi tahmini
"""

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import numpy as np
from scipy.stats import entropy
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# Vahşi kedi türleri listesi
WILD_CATS = [
    "aslan", "kaplan", "leopar", "çita", "jaguar",
    "puma", "vaşak", "çöl kedisi", "serval", "oselot",
    "kara ayak kedi", "pallas kedisi", "balıkçı kedisi",
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "puma", "lynx", "serval", "ocelot", "caracal",
    "sand cat", "fishing cat", "pallas cat", "black-footed cat"
]

# Entropi eşik değeri - bu değerin üzerinde vahşi kedi olabilir
ENTROPY_THRESHOLD = 2.5  # Yüksek belirsizlik eşiği

# Kedi cinsi bilgi kartları yolu
BREED_INFO_PATH = 'cat_breed_info.json'

@st.cache_resource
def load_breed_info():
    """Load cat breed information from JSON file"""
    try:
        with open(BREED_INFO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Cins bilgileri yüklenemedi: {e}")
        return {}

# Sayfa yapılandırması
st.set_page_config(
    page_title="🐱 Kedi Cinsi Tahmin Sistemi - ResNet-50",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "ResNet-50 ile Kedi Cinsi Tanıma - AI Powered"
    }
)

# Dark mode zorlaması ve CSS ile stil ayarları
st.markdown("""
    <style>
    /* Dark mode zorlama */
    .stApp {
        background-color: #0E1117 !important;
    }
    body {
        color: #FAFAFA !important;
        background-color: #0E1117 !important;
    }
    .main {
        background-color: #0E1117 !important;
    }
    /* Tüm text elemanları beyaz */
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #FAFAFA !important;
    }
    .main {
        background-color: #0E1117 !important;
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 30px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    .prediction-box {
        background-color: #262730 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin: 10px 0;
        color: #FAFAFA !important;
    }
    .accuracy-bar {
        background-color: #1E1E1E !important;
        border-radius: 5px;
        height: 25px;
        margin: 5px 0;
    }
    .accuracy-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #262730 !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        color: #FAFAFA !important;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #FF6B6B;
    }
    .metric-label {
        font-size: 14px;
        color: #AAAAAA !important;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Model yolu
MODEL_PATH = 'runs/resnet50_v2/weights/best.pth'  # Updated to v2 model (Epoch 23, 64.67% val acc)
YOLO_MODEL_PATH = 'yolo11n.pt'  # Pre-trained YOLO for object detection

@st.cache_resource
def load_yolo_detector():
    """Load YOLO model for cat detection"""
    if not YOLO_AVAILABLE:
        return None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except:
        return None

def detect_cat(image, yolo_model):
    """Detect if image contains a cat using YOLO"""
    if yolo_model is None:
        return True, 1.0, "YOLO not available - skipping detection"  # Skip detection if YOLO not available
    
    try:
        results = yolo_model(image, verbose=False)
        
        if len(results) == 0:
            return False, 0.0, "No objects detected"
        
        # COCO dataset: class 15 is cat
        detected_objects = []
        cat_found = False
        max_cat_conf = 0.0
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Get class names from model
            names = yolo_model.names if hasattr(yolo_model, 'names') else {}
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = names.get(cls, f"class_{cls}")
                detected_objects.append((cls, class_name, conf))
                
                # Class 15 = cat in COCO
                if cls == 15 and conf > 0.15:  # Lowered threshold to 0.15
                    cat_found = True
                    max_cat_conf = max(max_cat_conf, conf)
        
        # If cat was found, return success
        if cat_found:
            return True, max_cat_conf, f"Cat detected (class 15, conf {max_cat_conf:.2f})"
        
        # Debug: show what was detected
        if len(detected_objects) > 0:
            debug_info = ", ".join([f"{name}({cls}):{conf:.2f}" for cls, name, conf in detected_objects[:3]])
            debug_msg = f"No cat found. Detected: {debug_info}"
        else:
            debug_msg = "No objects detected"
            
        return False, 0.0, debug_msg
    except Exception as e:
        # If YOLO fails, allow the prediction to continue
        return True, 1.0, f"Detection error (proceeding anyway): {str(e)[:100]}"

@st.cache_resource
def load_resnet50_model(model_path):
    """Load trained ResNet-50 model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        num_classes = len(checkpoint['class_names'])
        class_names = checkpoint['class_names']
        
        # Create model with same architecture as train_resnet50_v2.py
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, class_names, device, checkpoint.get('val_loss', None)
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None, None

def preprocess_image(image):
    """Preprocess image for ResNet-50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution"""
    # probabilities: torch tensor of shape (num_classes,)
    probs = probabilities.cpu().numpy()
    # Normalize to ensure sum = 1
    probs = probs / probs.sum()
    # Add small epsilon to avoid log(0)
    probs = probs + 1e-10
    return entropy(probs, base=2)

def detect_wild_cat(probabilities, entropy_value, top_predictions, threshold=ENTROPY_THRESHOLD):
    """Detect if image might be a wild cat based on entropy and predictions"""
    # Yüksek entropi = yüksek belirsizlik = muhtemelen vahşi kedi
    is_high_entropy = entropy_value > threshold
    
    # En yüksek tahmin güveni düşükse şüpheli
    max_confidence = top_predictions[0] * 100
    is_low_confidence = max_confidence < 40
    
    # Tahmin edilen cinsler arasında vahşi kedi benzeri isimler var mı kontrol et
    # (bazı kullanıcılar vahşi kedi resimlerini yükleyebilir)
    
    # Vahşi kedi tespiti
    is_wild = is_high_entropy and is_low_confidence
    
    warning_msg = None
    if is_wild:
        warning_msg = (
            f"⚠️ **Vahşi Kedi Uyarısı**: Bu görsel bir ev kedisi değil, "
            f"vahşi bir kedi türü (aslan, kaplan, vaşak, leopar vb.) olabilir. "
            f"Sistem sadece ev kedisi cinsleri için eğitilmiştir.\n\n"
            f"📊 Entropi: {entropy_value:.3f} (Eşik: {threshold})\n"
            f"📉 Maksimum Güven: %{max_confidence:.2f}"
        )
    elif is_high_entropy:
        warning_msg = (
            f"⚠️ **Yüksek Belirsizlik**: Model bu görselde kararsız. "
            f"Bu bir vahşi kedi, kedi olmayan bir hayvan veya belirsiz bir görsel olabilir.\n\n"
            f"📊 Entropi: {entropy_value:.3f} (Eşik: {threshold})"
        )
    
    return is_wild, warning_msg, entropy_value

def predict_breed(model, image, class_names, device, top_k=5):
    """Predict cat breed with top-k results and entropy-based wild cat detection"""
    try:
        # Preprocess
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Calculate entropy for wild cat detection
        entropy_value = calculate_entropy(probabilities[0])
        
        # Detect wild cat
        is_wild, wild_warning, _ = detect_wild_cat(
            probabilities[0], 
            entropy_value, 
            top_probs[0]
        )
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'breed': class_names[idx],
                'confidence': prob.item() * 100
            })
        
        return results, entropy_value, is_wild, wild_warning
    except Exception as e:
        st.error(f"Tahmin yapılırken hata oluştu: {e}")
        return None, None, None, None

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #FF6B6B;'>🐱 Kedi Cinsi Tahmin Sistemi</h1>
            <p style='font-size: 18px; color: #666;'>ResNet-50 ile Derin Öğrenme Tabanlı Kedi Cinsi Tanıma</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Bilgileri")
        
        # Load models
        model, class_names, device, val_loss = load_resnet50_model(MODEL_PATH)
        yolo_model = load_yolo_detector()
        
        if model is not None:
            st.success("✅ Model başarıyla yüklendi!")
            
            # Model metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-value'>59</div>
                        <div class='metric-label'>Kedi Cinsi</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                device_icon = "🚀" if str(device) == "cuda" else "💻"
                device_text = "GPU (CUDA)" if str(device) == "cuda" else "CPU"
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{device_icon}</div>
                        <div class='metric-label'>{device_text}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            if val_loss:
                st.info(f"📉 Validation Loss: {val_loss:.4f}")
            
            st.markdown("---")
            st.markdown("### 🎯 Model Performansı")
            st.markdown("""
                **Sample Evaluation (2000 görüntü):**
                - Top-1 Accuracy: 56.95%
                - Top-3 Accuracy: 75.05%
                - Top-5 Accuracy: 83.35%
            """)
            
            st.markdown("---")
            st.markdown("### 🏆 En İyi Sınıflar")
            st.markdown("""
                1. Domestic Short Hair (97%)
                2. Persian (89%)
                3. Siamese (44%)
            """)
            
            # Detection status
            if yolo_model is not None:
                st.success("✅ Kedi Tespiti Aktif")
            else:
                st.warning("⚠️ Kedi Tespiti Devre Dışı")
            
            st.markdown("---")
            
            # Skip detection option
            skip_detection = st.checkbox("🔧 Kedi Tespitini Atla (Debug)", 
                                        help="Kedi tespitini devre dışı bırakır, doğrudan cins tahminine geçer")
            
            st.markdown("---")
            st.markdown("### ℹ️ Nasıl Kullanılır?")
            st.markdown("""
                1. Bir kedi fotoğrafı yükleyin
                2. "Tahmin Et" butonuna tıklayın
                3. Sonuçları görüntüleyin
                
                💡 **İpucu:** Daha iyi sonuçlar için:
                - Net, iyi aydınlatılmış fotoğraflar
                - Kedinin tüm vücudu görünür
                - Tek kedi olmalı
                
                ⚠️ **Önemli:** 
                - Sistem önce kedi tespiti yapar
                - Kedi olmayan görseller reddedilir
                - Vahşi kediler (aslan, kaplan, vaşak vb.) otomatik tespit edilir
                
                🦁 **Vahşi Kedi Tespiti:**
                - Entropi analizi ile vahşi kedi türleri tespit edilir
                - Sistem sadece ev kedisi cinsleri için eğitilmiştir
            """)
        else:
            st.error("❌ Model yüklenemedi!")
            st.info("Model yolu: " + MODEL_PATH)
            return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Fotoğraf Yükle")
        uploaded_file = st.file_uploader(
            "Kedi fotoğrafı seçin (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Yükleyeceğiniz fotoğrafta bir kedi bulunmalıdır"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Yüklenen Fotoğraf')
            
            # Predict button
            if st.button("🎯 Tahmin Et", key="predict_btn"):
                with st.spinner('Tahmin yapılıyor...'):
                    # Check if detection should be skipped
                    if skip_detection:
                        st.info("🔧 Kedi tespiti atlandı, doğrudan cins tahmini yapılıyor...")
                        is_cat = True
                        cat_confidence = 1.0
                        detection_msg = "Detection skipped"
                    else:
                        # First, detect if there's a cat
                        is_cat, cat_confidence, detection_msg = detect_cat(image, yolo_model)
                    
                    # Debug info
                    with st.expander("🔍 Debug Bilgisi"):
                        st.write(f"**Detection Result**: {is_cat}")
                        st.write(f"**Confidence**: {cat_confidence:.3f}")
                        st.write(f"**Message**: {detection_msg}")
                    
                    if not is_cat:
                        st.error("⚠️ Bu görselde kedi tespit edilemedi!")
                        st.warning(f"Detay: {detection_msg}")
                        st.info("💡 İpucu: Sol menüden 'Kedi Tespitini Atla' seçeneğini işaretleyerek doğrudan tahmin yapabilirsiniz.")
                        if 'results' in st.session_state:
                            del st.session_state['results']
                    else:
                        # Proceed with breed classification
                        if cat_confidence < 0.5 and yolo_model is not None and not skip_detection:
                            st.warning(f"⚠️ Düşük güvenle kedi tespit edildi (%{cat_confidence*100:.1f}). Sonuçlar yanıltıcı olabilir.")
                        
                        results, entropy_val, is_wild_cat, wild_warning = predict_breed(model, image, class_names, device, top_k=5)
                        
                        if results:
                            st.session_state['results'] = results
                            st.session_state['cat_confidence'] = cat_confidence
                            st.session_state['detection_msg'] = detection_msg
                            st.session_state['entropy_value'] = entropy_val
                            st.session_state['is_wild_cat'] = is_wild_cat
                            st.session_state['wild_warning'] = wild_warning
    
    with col2:
        st.markdown("### 🎯 Tahmin Sonuçları")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            cat_conf = st.session_state.get('cat_confidence', 1.0)
            entropy_val = st.session_state.get('entropy_value', 0.0)
            is_wild_cat = st.session_state.get('is_wild_cat', False)
            wild_warning = st.session_state.get('wild_warning', None)
            
            # Show wild cat warning if detected
            if wild_warning:
                st.error(wild_warning)
                st.markdown("""<div class='prediction-box' style='border-left: 5px solid #FF6B6B; background-color: #3D1F1F !important;'>
                    <h3 style='color: #FF6B6B;'>🦁 Vahşi Kedi Tespiti</h3>
                    <p>Bu görsel muhtemelen bir <strong>vahşi kedi türü</strong> (aslan, kaplan, leopar, vaşak, çita vb.) içeriyor.</p>
                    <p>Sistem yalnızca <strong>59 ev kedisi cinsi</strong> için eğitilmiştir ve vahşi kedileri tanıyamaz.</p>
                    <p><strong>Önerilen Aksiyon:</strong> Lütfen bir ev kedisi fotoğrafı yükleyin.</p>
                </div>""", unsafe_allow_html=True)
            
            # Show cat detection confidence if available
            if cat_conf < 1.0:
                st.info(f"🔍 Kedi Tespit Güveni: %{cat_conf*100:.1f}")
            
            # Show entropy value
            if entropy_val > 0:
                entropy_color = "🔴" if entropy_val > ENTROPY_THRESHOLD else "🟢"
                st.info(f"{entropy_color} Entropi Değeri: {entropy_val:.3f} (Eşik: {ENTROPY_THRESHOLD})")
            
            # Top prediction
            top_result = results[0]
            st.markdown(f"""
                <div class='prediction-box' style='border-left: 5px solid #FF6B6B;'>
                    <h2 style='color: #FF6B6B; margin: 0;'>{top_result['breed']}</h2>
                    <p style='font-size: 24px; color: #4CAF50; margin: 10px 0;'>
                        %{top_result['confidence']:.2f} güven
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Diğer Olası Cinler")
            
            for i, result in enumerate(results[1:], 2):
                confidence = result['confidence']
                breed_name = result['breed']
                
                # Use Streamlit columns for better compatibility
                col_name, col_conf = st.columns([3, 1])
                with col_name:
                    st.markdown(f"**{i}. {breed_name}**")
                with col_conf:
                    st.markdown(f"**%{confidence:.2f}**")
                
                # Progress bar for confidence
                st.progress(confidence / 100.0)
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence interpretation
            top_confidence = results[0]['confidence']
            if top_confidence > 80:
                st.success("✅ Yüksek güvenle tahmin edildi!")
            elif top_confidence > 60:
                st.info("ℹ️ Orta düzey güvenle tahmin edildi.")
            else:
                st.warning("⚠️ Düşük güven - Bu cins için daha fazla eğitim gerekebilir.")
            
            # Detailed breed information card
            st.markdown("---")
            st.markdown("### 📖 Detaylı Cins Bilgileri")
            
            breed_info = load_breed_info()
            top_breed = results[0]['breed']
            
            if top_breed in breed_info:
                info = breed_info[top_breed]
                
                # Create tabs for different information categories
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Genel Bilgi", "🏥 Sağlık", "🍽️ Beslenme", "✨ Bakım", "💝 Karakter"])
                
                with tab1:
                    st.markdown(f"**Türkçe İsim:** {info.get('name_tr', 'N/A')}")
                    st.markdown(f"**Menşei:** {info.get('origin', 'N/A')}")
                    st.markdown(f"**Boyut:** {info.get('size', 'N/A')}")
                    st.markdown(f"**Ağırlık:** {info.get('weight', 'N/A')}")
                    st.markdown(f"**Yaşam Süresi:** {info.get('lifespan', 'N/A')}")
                    
                    # Activity and grooming indicators
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Aktivite Seviyesi", info.get('activity_level', 'N/A'))
                    with col2:
                        st.metric("Bakım İhtiyacı", info.get('grooming', 'N/A'))
                    with col3:
                        st.metric("Tüy Dökme", info.get('shedding', 'N/A'))
                    
                    # Compatibility indicators
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Çocuk Dostu", info.get('child_friendly', 'N/A'))
                    with col5:
                        st.metric("Diğer Hayvanlarla", info.get('pet_friendly', 'N/A'))
                
                with tab2:
                    st.markdown("#### 🏥 Sağlık Özellikleri")
                    st.write(info.get('health', 'Bilgi mevcut değil.'))
                
                with tab3:
                    st.markdown("#### 🍽️ Beslenme Önerileri")
                    st.write(info.get('nutrition', 'Bilgi mevcut değil.'))
                
                with tab4:
                    st.markdown("#### ✨ Bakım Gereksinimleri")
                    st.write(info.get('care', 'Bilgi mevcut değil.'))
                
                with tab5:
                    st.markdown("#### 💝 Karakter Özellikleri")
                    st.write(info.get('character', 'Bilgi mevcut değil.'))
            else:
                st.info(f"🔍 {top_breed} için detaylı bilgi henüz eklenmemiş.")
                
        else:
            st.info("👆 Bir fotoğraf yükleyin ve 'Tahmin Et' butonuna tıklayın.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 ResNet-50 ile güçlendirilmiştir | PyTorch & Streamlit</p>
            <p>Model: Transfer Learning (ImageNet → Cat Breeds)</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
