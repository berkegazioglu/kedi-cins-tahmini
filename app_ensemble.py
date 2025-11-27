"""
Streamlit Web App for Ensemble Model
ResNet50 + EfficientNetB3 + ConvNeXt + Meta-Learner
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import json
import os
from ensemble_model import StackingEnsemble

# Page config
st.set_page_config(
    page_title="🐱 Kedi Cinsi Tahmin - Ensemble AI",
    page_icon="🐱",
    layout="wide"
)

# Paths
MODEL_PATH = 'runs/ensemble/ensemble_finetuned_best.pth'
CLASSES_FILE = 'runs/ensemble/training_summary.json'

@st.cache_resource
def load_model():
    """Load ensemble model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load classes
    with open(CLASSES_FILE, 'r') as f:
        summary = json.load(f)
    classes = summary['classes']
    num_classes = len(classes)
    
    # Load model
    model = StackingEnsemble(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    
    return model, classes, device, summary

def preprocess_image(image):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

def predict_ensemble(model, image_tensor, classes, device):
    """Get predictions from ensemble and individual models"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get detailed predictions
        predictions = model.predict_with_details(image_tensor)
        
        results = {}
        for model_name, pred in predictions.items():
            probs = pred[0].cpu().numpy()
            top5_idx = np.argsort(probs)[-5:][::-1]
            
            results[model_name] = {
                'classes': [classes[i] for i in top5_idx],
                'confidences': [float(probs[i]) * 100 for i in top5_idx]
            }
    
    return results

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;
    }
    .ensemble-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 25px;
        margin: 5px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🐱 Kedi Cinsi Tahmin Sistemi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🤖 Ensemble AI: ResNet50 + EfficientNetB3 + ConvNeXt</div>', unsafe_allow_html=True)
    
    # Load model
    try:
        model, classes, device, summary = load_model()
        
        # Sidebar - Model info
        with st.sidebar:
            st.header("📊 Model Bilgileri")
            st.write(f"**Cihaz:** {device.upper()}")
            st.write(f"**Sınıf Sayısı:** {len(classes)}")
            st.write(f"**Eğitim Tarihi:** {summary.get('training_date', 'N/A')}")
            
            st.subheader("🎯 Base Model Performansları")
            for model_name, acc in summary['base_models'].items():
                if isinstance(acc, (int, float)):
                    st.metric(model_name, f"{acc:.2f}%")
                else:
                    st.write(f"{model_name}: {acc}")
            
            st.subheader("🏆 Ensemble Performans")
            st.metric("Meta-Learner", f"{summary.get('meta_learner_accuracy', 0):.2f}%")
            st.metric("Final Ensemble", f"{summary.get('final_ensemble_accuracy', 0):.2f}%", 
                     delta=f"+{summary.get('final_ensemble_accuracy', 0) - summary.get('meta_learner_accuracy', 0):.2f}%")
            
            st.markdown("---")
            st.subheader("ℹ️ Nasıl Çalışır?")
            st.write("""
            1. **3 Farklı Model** bağımsız tahmin yapar
            2. **Meta-Learner** tahminleri birleştirir
            3. **Ensemble** sonucu en güvenilir tahmini verir
            """)
    
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {str(e)}")
        st.info("Lütfen önce modeli eğitin: `python train_ensemble.py`")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Kedi Fotoğrafı Yükleyin")
        uploaded_file = st.file_uploader("Bir resim seçin...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Yüklenen Görüntü', use_container_width=True)
            
            if st.button("🔮 Tahmin Et", type="primary", use_container_width=True):
                with st.spinner('🤖 AI modelleri analiz ediyor...'):
                    # Preprocess
                    img_tensor = preprocess_image(image)
                    
                    # Predict
                    results = predict_ensemble(model, img_tensor, classes, device)
                    
                    # Display results
                    with col2:
                        st.subheader("🎯 Tahmin Sonuçları")
                        
                        # Ensemble result (main)
                        ensemble_result = results['Ensemble']
                        st.markdown('<div class="ensemble-card">', unsafe_allow_html=True)
                        st.markdown("### 🏆 ENSEMBLE TAHMİN")
                        st.markdown(f"## {ensemble_result['classes'][0]}")
                        st.markdown(f"### Güven: {ensemble_result['confidences'][0]:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Top 5 from ensemble
                        st.markdown("#### 📊 En Olası 5 Cins (Ensemble)")
                        for i, (cls, conf) in enumerate(zip(ensemble_result['classes'], 
                                                            ensemble_result['confidences'])):
                            st.markdown(f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf}%;">
                                    {i+1}. {cls}: {conf:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Individual model predictions
                        st.markdown("#### 🔍 Model Detayları")
                        
                        tabs = st.tabs(["ResNet50", "EfficientNetB3", "ConvNeXt"])
                        
                        for tab, model_name in zip(tabs, ['ResNet50', 'EfficientNetB3', 'ConvNeXt']):
                            with tab:
                                model_result = results[model_name]
                                st.markdown(f"**En İyi Tahmin:** {model_result['classes'][0]}")
                                st.markdown(f"**Güven:** {model_result['confidences'][0]:.2f}%")
                                
                                with st.expander("Top 5 Detay"):
                                    for i, (cls, conf) in enumerate(zip(model_result['classes'][:5], 
                                                                        model_result['confidences'][:5])):
                                        st.progress(conf/100, text=f"{i+1}. {cls}: {conf:.1f}%")
        else:
            with col2:
                st.info("👈 Lütfen sol taraftan bir kedi fotoğrafı yükleyin")
                st.image("https://via.placeholder.com/400x300.png?text=Kedi+Fotografi+Yukleyin", 
                        use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by PyTorch | Ensemble Learning | Stacking Method</p>
        <p>Made with ❤️ by Berke Gazioğlu</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
