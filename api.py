"""
Flask RESTful API for Cat Breed Prediction
Backend API for React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import base64
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    import requests
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Model paths
MODEL_PATH = 'runs/resnet50_v2/weights/best.pth'
YOLO_MODEL_PATH = 'yolo11n.pt'

# Global variables for loaded models
model = None
class_names = None
device = None
yolo_model = None


def load_models():
    """Load YOLO and ResNet50 models"""
    global model, class_names, device, yolo_model
    
    # Load YOLO model
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✅ YOLO model loaded")
        except Exception as e:
            print(f"⚠️ YOLO model could not be loaded: {e}")
            yolo_model = None
    
    # Load ResNet50 model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        num_classes = len(checkpoint['class_names'])
        class_names = checkpoint['class_names']
        
        # Create model architecture
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ ResNet50 model loaded ({num_classes} classes)")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def detect_cat(image, yolo_model):
    """Detect if image contains a cat using YOLO"""
    if yolo_model is None:
        return True, 1.0, "YOLO mevcut değil - tespit atlanıyor"
    
    try:
        results = yolo_model(image, verbose=False)
        
        if len(results) == 0:
            return False, 0.0, "Hiçbir nesne tespit edilemedi"
        
        cat_found = False
        max_cat_conf = 0.0
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Class 15 = cat in COCO dataset
                if cls == 15 and conf > 0.15:
                    cat_found = True
                    max_cat_conf = max(max_cat_conf, conf)
        
        if cat_found:
            return True, max_cat_conf, f"Kedi tespit edildi (güven: {max_cat_conf:.2f})"
        
        return False, 0.0, "Bu görselde kedi tespit edilemedi"
    except Exception as e:
        return True, 1.0, f"Tespit hatası: {str(e)[:100]}"


def preprocess_image(image):
    """Preprocess image for ResNet-50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_breed(image, top_k=5):
    """Predict cat breed with top-k results"""
    global model, class_names, device
    
    if model is None or class_names is None:
        return None
    
    try:
        # Preprocess
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'breed': class_names[idx],
                'confidence': round(prob.item() * 100, 2)
            })
        
        return results
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'yolo_loaded': yolo_model is not None,
        'device': str(device) if device else None,
        'num_classes': len(class_names) if class_names else 0
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict cat breed from uploaded image"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'Görsel dosyası sağlanmadı'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Görsel dosyası seçilmedi'}), 400
        
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        
        # Optional: skip detection flag
        skip_detection = request.form.get('skip_detection', 'false').lower() == 'true'
        
        # Detect cat (optional)
        cat_detected = True
        cat_confidence = 1.0
        detection_message = "Tespit atlandı"
        
        if not skip_detection:
            cat_detected, cat_confidence, detection_message = detect_cat(image, yolo_model)
        
        if not cat_detected:
            return jsonify({
                'error': 'Bu görselde kedi tespit edilemedi',
                'detection_message': detection_message,
                'cat_detected': False
            }), 400
        
        # Predict breed
        results = predict_breed(image, top_k=5)
        
        if results is None:
            return jsonify({'error': 'Tahmin başarısız oldu'}), 500
        
        # Convert image to base64 for Gemini Vision analysis
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Analyze cat image with Gemini Vision
        cat_analysis = None
        if GEMINI_AVAILABLE and results:
            try:
                top_breed = results[0]['breed'] if results else None
                print(f"🔍 Görsel analizi başlatılıyor (Cins: {top_breed})...")
                cat_analysis = analyze_cat_image_with_gemini(image_base64, top_breed)
                if cat_analysis:
                    print(f"✅ Görsel analizi tamamlandı ({len(cat_analysis)} karakter)")
                else:
                    print("⚠️ Görsel analizi sonuç döndürmedi")
            except Exception as e:
                print(f"❌ Görsel analizi hatası: {e}")
                import traceback
                traceback.print_exc()
                cat_analysis = None
        
        response_data = {
            'success': True,
            'predictions': results,
            'cat_detection': {
                'detected': cat_detected,
                'confidence': round(cat_confidence * 100, 2),
                'message': detection_message
            }
        }
        
        if cat_analysis:
            response_data['cat_analysis'] = cat_analysis
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all cat breed classes"""
    if class_names is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    return jsonify({
        'classes': class_names,
        'total': len(class_names)
    })


def get_breed_info_from_gemini(breed_name):
    """Get cat breed information from Gemini AI using REST API"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        # Get API key from environment variable or use default
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyC7fP0h3TI82wRzdmGvnBvA4ys2UuLUVvs')
        if not api_key:
            print("⚠️ GEMINI_API_KEY environment variable not set")
            return None
        
        # Create prompt
        prompt = f"""Lütfen {breed_name} kedi cinsi hakkında kedi sahipleri için pratik ve kısa bilgiler ver. 
Aşağıdaki bilgileri Türkçe olarak, kısa ve öz şekilde ver (her bölüm 2-3 cümle):
1. **Karakter:** Bu kedi nasıl bir karaktere sahip? (sakin/aktif, sosyal/bağımsız, çocuklarla uyumlu mu?)
2. **Bakım:** Günlük bakımda nelere dikkat edilmeli? (tüy bakımı, egzersiz ihtiyacı)
3. **Sağlık:** Bilinen sağlık sorunları neler? (dikkat edilmesi gerekenler)
4. **Mama:** Bu cins için önerilen mama markaları ve beslenme ipuçları (kuru/yaş mama önerileri)
5. **Yaşam:** Hangi ortamlarda mutlu olur? (apartman/ev, çocuklu aile, tek kişi)

Toplam maksimum 200 kelime. Pratik ve kedi sahipleri için faydalı bilgiler ver."""

        # Call Gemini REST API
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            print("⚠️ Gemini API quota aşıldı (429). Breed info atlanıyor.")
            return None
        elif response.status_code == 401 or response.status_code == 403:
            error_msg = "API key geçersiz veya süresi dolmuş. Lütfen yeni bir API key oluşturun."
            print(f"❌ Gemini API yetkilendirme hatası ({response.status_code}): {error_msg}")
            return None
        
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0]:
                if 'parts' in result['candidates'][0]['content']:
                    if len(result['candidates'][0]['content']['parts']) > 0:
                        return result['candidates'][0]['content']['parts'][0].get('text', '')
        
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            print(f"❌ Gemini API yetkilendirme hatası ({e.response.status_code}): API key geçersiz veya süresi dolmuş.")
        else:
            print(f"❌ Gemini AI HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Gemini AI error: {e}")
        return None


@app.route('/api/breed-info', methods=['POST'])
def get_breed_info():
    """Get cat breed information from Gemini AI"""
    try:
        data = request.get_json()
        breed_name = data.get('breed')
        
        if not breed_name:
            return jsonify({'error': 'Kedi cinsi adı sağlanmadı'}), 400
        
        info = get_breed_info_from_gemini(breed_name)
        
        if info is None:
            return jsonify({
                'error': 'Gemini AI bilgi alınamadı. GEMINI_API_KEY kontrol edin.',
                'breed': breed_name
            }), 500
        
        return jsonify({
            'success': True,
            'breed': breed_name,
            'info': info
        })
        
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'}), 500


def analyze_cat_image_with_gemini(image_base64, breed_name=None):
    """Analyze cat image using Gemini AI vision model"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyC7fP0h3TI82wRzdmGvnBvA4ys2UuLUVvs')
        if not api_key:
            return None
        
        # Create prompt for image analysis
        breed_context = f" Tahmin edilen cins: {breed_name}." if breed_name else ""
        prompt = f"""Bu kedi fotoğrafını analiz et ve aşağıdaki bilgileri Türkçe olarak, kısa ve pratik şekilde ver:

**1. Yaş Tahmini:** Yavru mu, genç mi (1-2 yaş), yetişkin mi (3-7 yaş), yaşlı mı (8+ yaş)? Gözler, vücut yapısı ve tüy durumuna bakarak tahmin et.

**2. Sağlık Durumu:** Genel görünümü sağlıklı görünüyor mu? Tüy kalitesi, göz parlaklığı, vücut kondisyonu (zayıf/normal/kilolu) nasıl?

**3. Fiziksel Özellikler:** Vücut yapısı, tüy durumu, genel görünüm hakkında kısa notlar.

**4. Bakım Önerileri:** Bu kedinin görünümüne göre özel bakım önerileri (varsa).

{breed_context}

Her bölüm 1-2 cümle, toplam maksimum 150 kelime."""

        # Prepare image data
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Call Gemini Vision API
        # Use gemini-2.0-flash (same as text generation)
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            error_msg = response.json().get('error', {}).get('message', 'Quota aşıldı')
            print(f"⚠️ Gemini API quota aşıldı (429): {error_msg}")
            return "⚠️ Görsel analiz şu anda kullanılamıyor (API quota aşıldı). Lütfen daha sonra tekrar deneyin."
        
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0]:
                if 'parts' in result['candidates'][0]['content']:
                    if len(result['candidates'][0]['content']['parts']) > 0:
                        return result['candidates'][0]['content']['parts'][0].get('text', '')
        
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("⚠️ Gemini API quota aşıldı (429). Görsel analizi atlanıyor.")
            return None
        elif e.response.status_code == 401 or e.response.status_code == 403:
            error_msg = "API key geçersiz veya süresi dolmuş. Lütfen yeni bir API key oluşturun."
            print(f"❌ Gemini API yetkilendirme hatası ({e.response.status_code}): {error_msg}")
            return None
        else:
            print(f"❌ Gemini Vision API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Gemini Vision AI error: {e}")
        return None


@app.route('/api/analyze-cat', methods=['POST'])
def analyze_cat():
    """Analyze cat image for health, age, and other characteristics"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            # Try to get base64 image from JSON
            data = request.get_json()
            if data and 'image_base64' in data:
                image_base64 = data['image_base64']
                breed_name = data.get('breed', None)
            else:
                return jsonify({'error': 'Görsel dosyası veya base64 görsel sağlanmadı'}), 400
        else:
            # Get image from file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Görsel dosyası seçilmedi'}), 400
            
            # Read and convert to base64
            image = Image.open(io.BytesIO(file.read()))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            breed_name = request.form.get('breed', None)
        
        # Analyze with Gemini Vision
        analysis = analyze_cat_image_with_gemini(image_base64, breed_name)
        
        if analysis is None:
            return jsonify({
                'error': 'Görsel analizi yapılamadı. Gemini AI kontrol edin.',
            }), 500
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'breed': breed_name
        })
        
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    print("📦 Loading models...")
    
    if load_models():
        print("✅ All models loaded successfully!")
        print("🌐 Starting server on http://localhost:5001")
        print("📡 API endpoints:")
        print("   - GET  /api/health")
        print("   - POST /api/predict")
        print("   - GET  /api/classes")
        print("   - POST /api/breed-info (Gemini AI - Kedi cinsi bilgisi)")
        print("   - POST /api/analyze-cat (Gemini Vision - Fotoğraf analizi)")
        if GEMINI_AVAILABLE:
            print("   ✅ Gemini AI hazır (GEMINI_API_KEY gerekli)")
        else:
            print("   ⚠️  Gemini AI yüklü değil")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("❌ Failed to load models. Exiting...")

