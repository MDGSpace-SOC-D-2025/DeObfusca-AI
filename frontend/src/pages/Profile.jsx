import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Profile() {
  const { currentUser, idToken, logout } = useAuth();
  const [profile, setProfile] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (idToken) {
      fetch('/api/profile', {
        headers: { 'Authorization': `Bearer ${idToken}` }
      })
        .then(res => res.json())
        .then(data => setProfile(data))
        .catch(err => console.error('Failed to load profile:', err));
    }
  }, [idToken]);

  async function handleLogout() {
    await logout();
    navigate('/login');
  }

  if (!profile) {
    return <div className="loading">Loading profile...</div>;
  }

  return (
    <div className="profile-container">
      <div className="card">
        <h2>Profile</h2>
        <div className="profile-info">
          <div className="profile-field">
            <label>Email:</label>
            <span>{currentUser?.email}</span>
          </div>
          <div className="profile-field">
            <label>User ID:</label>
            <span className="monospace">{currentUser?.uid}</span>
          </div>
          <div className="profile-field">
            <label>Display Name:</label>
            <span>{profile.display_name || 'Not set'}</span>
          </div>
          <div className="profile-field">
            <label>Member Since:</label>
            <span>{new Date(profile.created_at).toLocaleDateString()}</span>
          </div>
        </div>
        <button onClick={handleLogout} className="btn-secondary">Logout</button>
      </div>
    </div>
  );
}
