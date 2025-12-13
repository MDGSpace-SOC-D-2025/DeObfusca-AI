// Real-time Job Monitor Component
import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from '../context/AuthContext';
import Card from './Card';
import './RealTimeMonitor.css';

export default function RealTimeMonitor({ jobId }) {
  const { user } = useAuth();
  const [socket, setSocket] = useState(null);
  const [jobStatus, setJobStatus] = useState({
    status: 'pending',
    progress: 0,
    currentStep: '',
    message: ''
  });
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // Connect to WebSocket
    const newSocket = io('http://localhost:8000');
    
    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      newSocket.emit('join', user.uid);
    });
    
    newSocket.on('job_status', (data) => {
      if (data.jobId === jobId) {
        setJobStatus(prev => ({
          ...prev,
          status: data.status,
          message: data.message,
          currentStep: data.step || prev.currentStep
        }));
        
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString(),
          message: data.message,
          type: 'info'
        }]);
      }
    });
    
    newSocket.on('job_progress', (data) => {
      if (data.jobId === jobId) {
        setJobStatus(prev => ({
          ...prev,
          progress: data.progress,
          currentStep: data.step
        }));
      }
    });
    
    newSocket.on('job_complete', (data) => {
      if (data.jobId === jobId) {
        setJobStatus({
          status: 'completed',
          progress: 100,
          currentStep: 'Completed',
          message: data.message,
          linesOfCode: data.linesOfCode,
          functionsCount: data.functionsCount
        });
        
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString(),
          message: data.message,
          type: 'success'
        }]);
      }
    });
    
    newSocket.on('job_failed', (data) => {
      if (data.jobId === jobId) {
        setJobStatus({
          status: 'failed',
          progress: 0,
          message: data.error,
          currentStep: 'Failed'
        });
        
        setLogs(prev => [...prev, {
          timestamp: new Date().toISOString(),
          message: data.error,
          type: 'error'
        }]);
      }
    });
    
    setSocket(newSocket);
    
    return () => {
      newSocket.close();
    };
  }, [user.uid, jobId]);

  return (
    <div className="real-time-monitor">
      <Card>
        <h3>Job Monitor - {jobId.substring(0, 8)}</h3>
        
        <div className="status-display">
          <div className={`status-badge status-${jobStatus.status}`}>
            {jobStatus.status.toUpperCase()}
          </div>
          <p className="current-step">{jobStatus.currentStep}</p>
        </div>
        
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${jobStatus.progress}%` }}
            ></div>
          </div>
          <span className="progress-text">{jobStatus.progress}%</span>
        </div>
        
        {jobStatus.message && (
          <div className="status-message">
            {jobStatus.message}
          </div>
        )}
        
        {jobStatus.status === 'completed' && (
          <div className="completion-stats">
            <div className="stat">
              <span className="stat-label">Lines of Code:</span>
              <span className="stat-value">{jobStatus.linesOfCode}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Functions:</span>
              <span className="stat-value">{jobStatus.functionsCount}</span>
            </div>
          </div>
        )}
        
        <div className="logs-section">
          <h4>Activity Log</h4>
          <div className="logs-container">
            {logs.map((log, idx) => (
              <div key={idx} className={`log-entry log-${log.type}`}>
                <span className="log-timestamp">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
