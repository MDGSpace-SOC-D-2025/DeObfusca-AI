import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const { idToken } = useAuth();
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');

  useEffect(() => {
    loadProjects();
  }, [idToken]);

  async function loadProjects() {
    if (!idToken) return;
    try {
      const res = await fetch('/api/projects', {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      const data = await res.json();
      setProjects(data.projects || []);
    } catch (err) {
      console.error('Failed to load projects:', err);
    }
    setLoading(false);
  }

  async function createProject(e) {
    e.preventDefault();
    if (!newProjectName.trim()) return;

    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${idToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: newProjectName,
          description: newProjectDesc
        })
      });
      const data = await res.json();
      if (data.project) {
        setProjects([data.project, ...projects]);
        setNewProjectName('');
        setNewProjectDesc('');
        setShowNewProject(false);
      }
    } catch (err) {
      console.error('Failed to create project:', err);
    }
  }

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Your Projects</h1>
        <button onClick={() => setShowNewProject(!showNewProject)} className="btn-primary">
          + New Project
        </button>
      </div>

      {showNewProject && (
        <div className="card new-project-form">
          <h3>Create New Project</h3>
          <form onSubmit={createProject}>
            <div className="form-group">
              <label>Project Name</label>
              <input
                type="text"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="My Decompilation Project"
                required
              />
            </div>
            <div className="form-group">
              <label>Description (optional)</label>
              <textarea
                value={newProjectDesc}
                onChange={(e) => setNewProjectDesc(e.target.value)}
                placeholder="Project description..."
                rows="3"
              />
            </div>
            <div className="form-actions">
              <button type="submit" className="btn-primary">Create</button>
              <button type="button" onClick={() => setShowNewProject(false)} className="btn-secondary">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="projects-grid">
        {projects.length === 0 ? (
          <div className="empty-state">
            <p>No projects yet. Create your first project to get started!</p>
          </div>
        ) : (
          projects.map(project => (
            <Link to={`/projects/${project.id}`} key={project.id} className="project-card">
              <h3>{project.name}</h3>
              {project.description && <p className="project-desc">{project.description}</p>}
              <div className="project-meta">
                <span>{project.file_count} files</span>
                <span className={`status-badge status-${project.status}`}>
                  {project.status}
                </span>
              </div>
              <div className="project-date">
                Updated {new Date(project.updated_at).toLocaleDateString()}
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}
