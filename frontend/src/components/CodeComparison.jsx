// Code Comparison Component
import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { API_ENDPOINTS, authenticatedFetch } from '../config/api';
import Card from './Card';
import Button from './Button';
import './CodeComparison.css';

export default function CodeComparison({ jobId1, jobId2 }) {
  const { idToken } = useAuth();
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side' or 'unified'

  useEffect(() => {
    if (jobId1 && jobId2) {
      loadComparison();
    }
  }, [jobId1, jobId2]);

  const loadComparison = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await authenticatedFetch(
        API_ENDPOINTS.analytics.compare,
        idToken,
        {
          method: 'POST',
          body: JSON.stringify({ jobId1, jobId2 })
        }
      );
      
      const data = await response.json();
      setComparison(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading comparison...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!comparison) {
    return (
      <Card>
        <h3>Code Comparison</h3>
        <p>Select two jobs to compare their decompilation results.</p>
      </Card>
    );
  }

  const { job1, job2, differences } = comparison;

  return (
    <div className="code-comparison">
      <Card>
        <div className="comparison-header">
          <h3>Code Comparison</h3>
          <div className="view-mode-toggle">
            <button
              className={viewMode === 'side-by-side' ? 'active' : ''}
              onClick={() => setViewMode('side-by-side')}
            >
              Side by Side
            </button>
            <button
              className={viewMode === 'unified' ? 'active' : ''}
              onClick={() => setViewMode('unified')}
            >
              Unified
            </button>
          </div>
        </div>

        <div className="comparison-stats">
          <div className="stat-group">
            <h4>Job 1: {job1.filename}</h4>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Lines:</span>
                <span className="stat-value">{job1.linesOfCode}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Functions:</span>
                <span className="stat-value">{job1.functionsCount}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Time:</span>
                <span className="stat-value">{job1.processingTime}s</span>
              </div>
            </div>
          </div>

          <div className="comparison-arrows">
            <div className="arrow-indicator">
              {differences.linesOfCodeDiff > 0 ? '→' : differences.linesOfCodeDiff < 0 ? '←' : '≈'}
            </div>
            <div className="diff-summary">
              <div className={`diff-value ${differences.linesOfCodeDiff > 0 ? 'positive' : differences.linesOfCodeDiff < 0 ? 'negative' : ''}`}>
                {differences.linesOfCodeDiff > 0 ? '+' : ''}{differences.linesOfCodeDiff} lines
              </div>
              <div className={`diff-value ${differences.functionsDiff > 0 ? 'positive' : differences.functionsDiff < 0 ? 'negative' : ''}`}>
                {differences.functionsDiff > 0 ? '+' : ''}{differences.functionsDiff} functions
              </div>
            </div>
          </div>

          <div className="stat-group">
            <h4>Job 2: {job2.filename}</h4>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Lines:</span>
                <span className="stat-value">{job2.linesOfCode}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Functions:</span>
                <span className="stat-value">{job2.functionsCount}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Time:</span>
                <span className="stat-value">{job2.processingTime}s</span>
              </div>
            </div>
          </div>
        </div>

        <div className={`code-view code-view-${viewMode}`}>
          {viewMode === 'side-by-side' ? (
            <>
              <div className="code-panel">
                <div className="panel-header">Job 1</div>
                <pre className="code-content">{job1.code || '// Code not available'}</pre>
              </div>
              <div className="code-panel">
                <div className="panel-header">Job 2</div>
                <pre className="code-content">{job2.code || '// Code not available'}</pre>
              </div>
            </>
          ) : (
            <div className="code-panel unified">
              <div className="panel-header">Unified View</div>
              <pre className="code-content">
                {`// Job 1: ${job1.filename}\n${job1.code || '// Code not available'}\n\n`}
                {`// Job 2: ${job2.filename}\n${job2.code || '// Code not available'}`}
              </pre>
            </div>
          )}
        </div>

        <div className="comparison-actions">
          <Button onClick={loadComparison}>Refresh Comparison</Button>
        </div>
      </Card>
    </div>
  );
}
