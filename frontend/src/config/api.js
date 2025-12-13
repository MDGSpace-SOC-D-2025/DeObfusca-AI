/**
 * API Configuration
 * Centralized configuration for all API calls
 */

// Get API base URL from environment or use default
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
  // Auth endpoints
  auth: {
    register: `${API_BASE_URL}/api/auth/register`,
    login: `${API_BASE_URL}/api/auth/login`,
    logout: `${API_BASE_URL}/api/auth/logout`,
  },
  
  // Project endpoints
  projects: {
    list: `${API_BASE_URL}/api/projects`,
    create: `${API_BASE_URL}/api/projects`,
    get: (id) => `${API_BASE_URL}/api/projects/${id}`,
    update: (id) => `${API_BASE_URL}/api/projects/${id}`,
    delete: (id) => `${API_BASE_URL}/api/projects/${id}`,
  },
  
  // Job endpoints
  jobs: {
    list: `${API_BASE_URL}/api/jobs`,
    create: `${API_BASE_URL}/api/jobs`,
    get: (id) => `${API_BASE_URL}/api/jobs/${id}`,
    update: (id) => `${API_BASE_URL}/api/jobs/${id}`,
    delete: (id) => `${API_BASE_URL}/api/jobs/${id}`,
  },
  
  // Upload endpoints
  upload: {
    single: `${API_BASE_URL}/api/upload`,
    batch: `${API_BASE_URL}/api/upload/batch`,
  },
  
  // History endpoints
  history: {
    list: `${API_BASE_URL}/api/history`,
    get: (id) => `${API_BASE_URL}/api/history/${id}`,
  },
  
  // User/Profile endpoints
  user: {
    profile: `${API_BASE_URL}/api/user/profile`,
    update: `${API_BASE_URL}/api/user/profile`,
  },
  
  // Orchestrator (AI Pipeline) endpoints
  orchestrator: {
    health: 'http://localhost:5000/health',
    sanitize: 'http://localhost:5000/sanitize',
  }
};

/**
 * Helper function to make authenticated API calls
 * @param {string} url - API endpoint URL
 * @param {object} options - Fetch options
 * @param {string} token - Firebase ID token
 */
export async function authenticatedFetch(url, options = {}, token) {
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(url, {
    ...options,
    headers,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }
  
  return response.json();
}

/**
 * Helper function for file uploads
 * @param {string} url - API endpoint URL
 * @param {FormData} formData - Form data with files
 * @param {string} token - Firebase ID token
 */
export async function uploadFiles(url, formData, token) {
  const headers = {};
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Upload failed' }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }
  
  return response.json();
}

export default {
  API_BASE_URL,
  API_ENDPOINTS,
  authenticatedFetch,
  uploadFiles,
};
