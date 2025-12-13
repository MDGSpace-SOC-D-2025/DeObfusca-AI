import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import GlobalStyles from './styles/GlobalStyles'
import Navbar from './components/Navbar'
import PrivateRoute from './components/PrivateRoute'
import Home from './pages/Home'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/Dashboard'
import ProjectDetail from './pages/ProjectDetail'
import JobDetail from './pages/JobDetail'
import History from './pages/History'
import Profile from './pages/Profile'

// Error boundary to catch rendering errors
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '50px', 
          background: '#1a1a2e', 
          color: '#ff6b6b', 
          minHeight: '100vh',
          fontFamily: 'monospace'
        }}>
          <h1>Something went wrong!</h1>
          <pre style={{ whiteSpace: 'pre-wrap', marginTop: '20px' }}>
            {this.state.error?.toString()}
          </pre>
          <button 
            onClick={() => window.location.reload()} 
            style={{ marginTop: '20px', padding: '10px 20px' }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App(){
  return (
    <ErrorBoundary>
      <GlobalStyles />
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
            <Route path="/projects/:projectId" element={<PrivateRoute><ProjectDetail /></PrivateRoute>} />
            <Route path="/jobs/:jobId" element={<PrivateRoute><JobDetail /></PrivateRoute>} />
            <Route path="/history" element={<PrivateRoute><History /></PrivateRoute>} />
            <Route path="/profile" element={<PrivateRoute><Profile /></PrivateRoute>} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </ErrorBoundary>
  )
}
