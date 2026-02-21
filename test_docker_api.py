"""
Test script for Docker API
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8002"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Pipeline loaded: {data.get('pipeline_loaded')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is Docker running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print("\nℹ️  Testing info endpoint...")
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Info retrieved!")
            print(f"   Architecture: {data.get('architecture')}")
            print(f"   Device: {data.get('device')}")
            if 'stage1' in data:
                print(f"   Stage 1: {data['stage1'].get('model_type')}")
            if 'stage2' in data:
                print(f"   Stage 2: {data['stage2'].get('models')}")
            return True
        else:
            print(f"❌ Info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict(image_path=None):
    """Test prediction endpoint"""
    print("\n🔮 Testing prediction endpoint...")
    
    # Try to find a test image
    test_images = [
        image_path,
        "uploads/test.jpg",
        "uploads/sample.jpg",
        "test.jpg",
        "sample.jpg"
    ]
    
    image_file = None
    for img in test_images:
        if img and Path(img).exists():
            image_file = img
            break
    
    if not image_file:
        print("⚠️  No test image found. Skipping prediction test.")
        print(f"   Tried: {', '.join(str(p) for p in test_images if p)}")
        return None
    
    print(f"   Using image: {image_file}")
    
    try:
        with open(image_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                data={'top_k': 5},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            
            # Stage 1 results
            if data.get('stage1'):
                stage1 = data['stage1']
                print(f"\n   📷 Stage 1 (Detection):")
                print(f"      Cat detected: {stage1.get('cat_detected')}")
                print(f"      Confidence: {stage1.get('detection_confidence'):.4f}" if stage1.get('detection_confidence') else "      Confidence: N/A")
                print(f"      Time: {stage1.get('detection_time'):.3f}s" if stage1.get('detection_time') else "      Time: N/A")
            
            # Stage 2 results
            if data.get('stage2'):
                stage2 = data['stage2']
                print(f"\n   🎯 Stage 2 (Classification):")
                print(f"      Top breed: {stage2.get('top_breed')}")
                print(f"      Confidence: {stage2.get('confidence'):.2%}" if stage2.get('confidence') else "      Confidence: N/A")
                print(f"      Wild cat: {stage2.get('is_wild_cat')}")
                print(f"      Entropy: {stage2.get('entropy'):.4f}" if stage2.get('entropy') else "      Entropy: N/A")
                print(f"      Time: {stage2.get('classification_time'):.3f}s" if stage2.get('classification_time') else "      Time: N/A")
                
                if stage2.get('predictions'):
                    print(f"\n   📊 Top 5 Predictions:")
                    for i, pred in enumerate(stage2['predictions'][:5], 1):
                        print(f"      {i}. {pred['breed']:30s} {pred['confidence']:6.2%}")
            
            print(f"\n   ⏱️  Total time: {data.get('total_time'):.3f}s" if data.get('total_time') else "")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🐱 Cat Breed Classification API - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health()))
    
    # Test 2: Info
    results.append(("Info Endpoint", test_info()))
    
    # Test 3: Prediction
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    pred_result = test_predict(image_path)
    if pred_result is not None:
        results.append(("Prediction", pred_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20s} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
