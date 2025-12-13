import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useParams, useNavigate } from 'react-router-dom';

export default function ProjectDetail() {
  const { projectId } = useParams();
  const { idToken } = useAuth();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [zipFile, setZipFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    loadProject();
    loadJobs();
  }, [projectId, idToken]);

  async function loadProject() {
    if (!idToken) return;
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      const data = await res.json();
      setProject(data);
    } catch (err) {
      console.error('Failed to load project:', err);
    }
  }

  async function loadJobs() {
    if (!idToken) return;
    try {
      const res = await fetch(`/api/projects/${projectId}/jobs`, {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      const data = await res.json();
      setJobs(data.jobs || []);
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  }

  async function handleBatchUpload(e) {
    e.preventDefault();
    if (selectedFiles.length === 0 && !zipFile) return;

    setUploading(true);
    const formData = new FormData();

    if (zipFile) {
      // Upload zip
      formData.append('file', zipFile);
      try {
        const res = await fetch(`/api/batch-upload-zip?project_id=${projectId}`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${idToken}` },
          body: formData
        });
        const data = await res.json();
        alert(`Uploaded ${data.count} files from ZIP`);
        loadJobs();
        loadProject();
        setZipFile(null);
      } catch (err) {
        console.error('Failed to upload zip:', err);
      }
    } else {
      // Upload individual files
      for (let file of selectedFiles) {
        formData.append('files', file);
      }
      try {
        const res = await fetch(`/api/batch-upload?project_id=${projectId}`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${idToken}` },
          body: formData
        });
        const data = await res.json();
        alert(`Uploaded ${data.count} files`);
        loadJobs();
        loadProject();
        setSelectedFiles([]);
      } catch (err) {
        console.error('Failed to upload files:', err);
      }
    }
    setUploading(false);
  }

  async function deleteProject() {
    if (!confirm('Delete this project and all its files?')) return;
    try {
      await fetch(`/api/projects/${projectId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      navigate('/dashboard');
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  }

  if (!project) {
    return <div className="loading">Loading project...</div>;
  }

  return (
    <div className="project-detail">
      <div className="project-header">
        <div>
          <h1>{project.name}</h1>
          {project.description && <p className="project-desc">{project.description}</p>}
        </div>
        <button onClick={deleteProject} className="btn-danger">Delete Project</button>
      </div>

      <div className="card">
        <h2>Upload Files</h2>
        <form onSubmit={handleBatchUpload}>
          <div className="upload-section">
            <div className="form-group">
              <label>Upload Multiple Binary Files</label>
              <input
                type="file"
                multiple
                onChange={(e) => setSelectedFiles([...e.target.files])}
                disabled={zipFile !== null}
              />
              {selectedFiles.length > 0 && (
                <div className="file-list">
                  {selectedFiles.length} file(s) selected
                </div>
              )}
            </div>
            <div className="divider-text">OR</div>
            <div className="form-group">
              <label>Upload ZIP Archive</label>
              <input
                type="file"
                accept=".zip"
                onChange={(e) => setZipFile(e.target.files[0])}
                disabled={selectedFiles.length > 0}
              />
            </div>
          </div>
          <button type="submit" disabled={uploading} className="btn-primary">
            {uploading ? 'Uploading...' : 'Upload & Process'}
          </button>
        </form>
      </div>

      <div className="card">
        <h2>Jobs ({jobs.length})</h2>
        {jobs.length === 0 ? (
          <div className="empty-state">No files uploaded yet.</div>
        ) : (
          <div className="jobs-table">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Status</th>
                  <th>Created</th>
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
                      <a href={`/jobs/${job.id}`} className="btn-link">View</a>
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
