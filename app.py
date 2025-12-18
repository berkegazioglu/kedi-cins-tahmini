"""
PatiPedia - Kedi Cinsi Tanıma ve Bilgi Platformu
Hugging Face Spaces Deployment
"""

import os
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from api import app as api_app, predict_breed, yolo_model, model, class_names, device

# Flask app
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'yolo_loaded': yolo_model is not None,
        'model_loaded': model is not None,
        'device': str(device),
        'classes': len(class_names)
    })

# API endpoints (proxy to api.py)
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_proxy(path):
    """Proxy API requests to api.py"""
    from flask import request
    with api_app.test_request_context(path='/' + path, method=request.method, 
                                       data=request.data, headers=request.headers):
        try:
            response = api_app.full_dispatch_request()
            return response
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React build or index.html"""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # HF Spaces uses port 7860
    print(f"🚀 PatiPedia starting on port {port}")
    print(f"🐱 YOLO loaded: {yolo_model is not None}")
    print(f"🧠 Model loaded: {model is not None}")
    print(f"💻 Device: {device}")
    print(f"📊 Classes: {len(class_names)}")
    app.run(host='0.0.0.0', port=port, debug=False)
