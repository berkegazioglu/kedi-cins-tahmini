const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

class ApiService {
  /**
   * Get available models
   */
  async getModels() {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching models:', error);
      throw error;
    }
  }

  /**
   * Predict cat breed from image
   * @param {File} imageFile - Image file to analyze
   * @param {string} model - Model to use (resnet50, efficientnet, mobilenet, ensemble)
   * @param {boolean} skipDetection - Skip YOLO cat detection
   * @param {number} topK - Number of top predictions
   */
  async predictBreed(imageFile, model = 'resnet50', skipDetection = false, topK = 5) {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const params = new URLSearchParams({
        model: model,
        skip_detection: skipDetection.toString(),
        top_k: topK.toString()
      });

      const response = await fetch(`${API_BASE_URL}/predict?${params}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting breed:', error);
      throw error;
    }
  }

  /**
   * Predict cat breed using all available models
   * @param {File} imageFile - Image file to analyze
   * @param {boolean} skipDetection - Skip YOLO cat detection
   * @param {number} topK - Number of top predictions per model
   */
  async predictAllModels(imageFile, skipDetection = false, topK = 5) {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const params = new URLSearchParams({
        skip_detection: skipDetection.toString(),
        top_k: topK.toString()
      });

      const response = await fetch(`${API_BASE_URL}/predict-all?${params}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting with all models:', error);
      throw error;
    }
  }

  /**
   * Detect if image contains a cat
   */
  async detectCat(imageFile) {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error detecting cat:', error);
      throw error;
    }
  }

  /**
   * Get all available breeds
   */
  async getBreeds() {
    try {
      const response = await fetch(`${API_BASE_URL}/breeds`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching breeds:', error);
      throw error;
    }
  }

  /**
   * Get information about a specific breed
   */
  async getBreedInfo(breedName) {
    try {
      const response = await fetch(`${API_BASE_URL}/breeds/${encodeURIComponent(breedName)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching breed info:', error);
      throw error;
    }
  }

  /**
   * Health check endpoint
   */
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }
}

export default new ApiService();
