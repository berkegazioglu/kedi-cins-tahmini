"""
Test script for multi-model backend
"""
import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        data = response.json()
        print(f"✅ Health check passed")
        print(f"   Status: {data['status']}")
        print(f"   Models loaded: {len(data.get('models_status', {}))}")
        for model, status in data.get('models_status', {}).items():
            print(f"   - {model}: {status}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\n📋 Testing /models endpoint...")
    try:
        response = requests.get(f"{API_BASE}/models")
        data = response.json()
        print(f"✅ Models endpoint passed")
        print(f"   Available models: {data['models']}")
        print(f"   Default model: {data['default_model']}")
        return data['models']
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return []

def test_breeds():
    """Test breeds endpoint"""
    print("\n🐱 Testing /breeds endpoint...")
    try:
        response = requests.get(f"{API_BASE}/breeds")
        data = response.json()
        print(f"✅ Breeds endpoint passed")
        print(f"   Total breeds: {data['count']}")
        print(f"   Sample breeds: {', '.join(data['breeds'][:5])}...")
        return True
    except Exception as e:
        print(f"❌ Breeds endpoint failed: {e}")
        return False

def test_predict(image_path, model='resnet50'):
    """Test predict endpoint with a specific model"""
    print(f"\n🔮 Testing /predict endpoint with model: {model}...")
    
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'model': model,
                'skip_detection': 'true',
                'top_k': 3
            }
            response = requests.post(
                f"{API_BASE}/predict",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            print(f"   Model used: {data.get('model_used', model)}")
            print(f"   Top breed: {data['top_breed']} ({data['top_confidence']*100:.1f}%)")
            print(f"   Is wild cat: {data['is_wild_cat']}")
            print(f"   Entropy: {data['entropy']:.3f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_predict_all(image_path):
    """Test predict-all endpoint"""
    print(f"\n🎯 Testing /predict-all endpoint...")
    
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'skip_detection': 'true',
                'top_k': 3
            }
            response = requests.post(
                f"{API_BASE}/predict-all",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ All-models prediction successful")
            print(f"   Consensus: {data.get('consensus', 'N/A')}")
            print(f"   Models used: {len(data['models_used'])}")
            
            for model_name in data['models_used']:
                pred = data['predictions'][model_name]
                if 'error' in pred:
                    print(f"   ❌ {model_name}: {pred['error']}")
                else:
                    print(f"   ✅ {model_name}: {pred['top_breed']} "
                          f"({pred['top_confidence']*100:.1f}%) "
                          f"[{pred['inference_time']}s]")
            return True
        else:
            print(f"❌ All-models prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ All-models prediction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Multi-Model Backend Test Suite")
    print("=" * 60)
    
    # Test basic endpoints
    if not test_health():
        print("\n❌ Backend not running or unhealthy!")
        print("   Start backend with: uvicorn backend.api.main:app --reload")
        return
    
    models = test_models()
    test_breeds()
    
    # Test prediction with image if available
    print("\n" + "=" * 60)
    print("🖼️  Testing with sample images...")
    print("=" * 60)
    
    # Try to find a test image
    test_images = [
        "test_cat.jpg",
        "uploads/test.jpg",
        "sample.jpg"
    ]
    
    test_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break
    
    if test_image:
        print(f"\n📸 Using test image: {test_image}")
        
        # Test each model
        for model in models:
            test_predict(test_image, model)
        
        # Test all models together
        test_predict_all(test_image)
    else:
        print("\n⚠️  No test image found. Skipping prediction tests.")
        print("   To test predictions, add a cat image as 'test_cat.jpg'")
    
    print("\n" + "=" * 60)
    print("✨ Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
