// API Service for Cat Breed Classification
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

class ApiService {
  /**
   * Predict cat breed from image
   */
  async predictBreed(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams();
    if (options.skipDetection) params.append('skip_detection', 'true');
    if (options.topK) params.append('top_k', options.topK.toString());
    
    const url = `${API_BASE_URL}/predict${params.toString() ? '?' + params.toString() : ''}`;
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }
    
    return await response.json();
  }

  /**
   * Detect if image contains a cat
   */
  async detectCat(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Detection failed');
    }
    
    return await response.json();
  }

  /**
   * Get list of all breeds
   */
  async getBreeds() {
    const response = await fetch(`${API_BASE_URL}/breeds`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch breeds');
    }
    
    return await response.json();
  }

  /**
   * Get detailed information about a breed
   */
  async getBreedInfo(breedName) {
    const response = await fetch(`${API_BASE_URL}/breeds/${encodeURIComponent(breedName)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Breed not found');
      }
      throw new Error('Failed to fetch breed info');
    }
    
    return await response.json();
  }

  /**
   * Health check
   */
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error('API health check failed');
    }
    
    return await response.json();
  }
}

export default new ApiService();
