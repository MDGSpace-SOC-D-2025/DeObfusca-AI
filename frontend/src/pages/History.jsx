import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Link } from 'react-router-dom';

export default function History() {
  const { idToken } = useAuth();
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, [idToken]);

  async function loadHistory() {
    if (!idToken) return;
    try {
      const res = await fetch('/api/history', {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      const data = await res.json();
      setJobs(data.jobs || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
    setLoading(false);
  }

  if (loading) {
    return <div className="loading">Loading history...</div>;
  }

  return (
    <div className="history">
      <h1>Decompilation History</h1>
      <div className="card">
        {jobs.length === 0 ? (
          <div className="empty-state">No decompilation history yet.</div>
        ) : (
          <div className="jobs-table">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Completed</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(job => (
                  <tr key={job.id}>
                    <td className="monospace">{job.filename}</td>
                    <td>
                      <span className={`status-badge status-${job.status}`}>
                        {job.status}
                      </span>
                    </td>
                    <td>{new Date(job.created_at).toLocaleString()}</td>
                    <td>
                      {job.completed_at 
                        ? new Date(job.completed_at).toLocaleString() 
                        : '-'}
                    </td>
                    <td>
                      <Link to={`/jobs/${job.id}`} className="btn-link">View</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
