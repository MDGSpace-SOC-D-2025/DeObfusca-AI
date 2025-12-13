import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useParams, Link } from 'react-router-dom';

export default function JobDetail() {
  const { jobId } = useParams();
  const { idToken } = useAuth();
  const [job, setJob] = useState(null);

  useEffect(() => {
    loadJob();
    // Poll for updates if job is processing
    const interval = setInterval(() => {
      if (job && (job.status === 'pending' || job.status === 'processing')) {
        loadJob();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [jobId, idToken]);

  async function loadJob() {
    if (!idToken) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}`, {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      const data = await res.json();
      setJob(data);
    } catch (err) {
      console.error('Failed to load job:', err);
    }
  }

  if (!job) {
    return <div className="loading">Loading job...</div>;
  }

  return (
    <div className="job-detail">
      <div className="job-header">
        <h1>Job: {job.filename}</h1>
        <span className={`status-badge status-${job.status}`}>{job.status}</span>
      </div>

      <div className="card">
        <h2>Job Information</h2>
        <div className="job-info">
          <div className="info-row">
            <label>Job ID:</label>
            <span className="monospace">{job.id}</span>
          </div>
          <div className="info-row">
            <label>Filename:</label>
            <span className="monospace">{job.filename}</span>
          </div>
          <div className="info-row">
            <label>Project:</label>
            <Link to={`/projects/${job.project_id}`}>View Project</Link>
          </div>
          <div className="info-row">
            <label>Created:</label>
            <span>{new Date(job.created_at).toLocaleString()}</span>
          </div>
          {job.completed_at && (
            <div className="info-row">
              <label>Completed:</label>
              <span>{new Date(job.completed_at).toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>

      {job.error_message && (
        <div className="card error-card">
          <h2>Error</h2>
          <pre className="error-message">{job.error_message}</pre>
        </div>
      )}

      {job.sanitized_features && (
        <div className="card">
          <h2>Sanitized Features</h2>
          <pre className="features">{JSON.stringify(job.sanitized_features, null, 2)}</pre>
        </div>
      )}

      {job.decompiled_source && (
        <div className="card">
          <h2>Decompiled Source Code</h2>
          <pre className="source">{job.decompiled_source}</pre>
          <button
            onClick={() => {
              const blob = new Blob([job.decompiled_source], { type: 'text/plain' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `${job.filename}.c`;
              a.click();
            }}
            className="btn-primary"
          >
            Download Source
          </button>
        </div>
      )}
    </div>
  );
}
