import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { AuthProvider } from './context/AuthContext'
import App from './App'
import './styles.css'

// Debug: Log when main.jsx executes
console.log('main.jsx: Starting app initialization');

try {
  const rootElement = document.getElementById('root');
  console.log('main.jsx: Root element found:', !!rootElement);
  
  const root = createRoot(rootElement);
  console.log('main.jsx: React root created');
  
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <ThemeProvider>
          <AuthProvider>
            <App />
          </AuthProvider>
        </ThemeProvider>
      </BrowserRouter>
    </React.StrictMode>
  );
  
  console.log('main.jsx: Render called successfully');
} catch (error) {
  console.error('main.jsx: Fatal error during initialization:', error);
  document.body.innerHTML = `
    <div style="padding: 50px; background: #2d1f1f; color: #ff8888; min-height: 100vh; font-family: monospace;">
      <h1>Fatal Initialization Error</h1>
      <pre style="white-space: pre-wrap; margin-top: 20px;">${error.stack || error.message || error}</pre>
    </div>
  `;
}
