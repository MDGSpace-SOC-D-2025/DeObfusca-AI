const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const AdmZip = require('adm-zip');
const authMiddleware = require('../middleware/auth');

const UPLOAD_DIR = '/data/uploads';
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://ai-service:5000';

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Batch upload files
router.post('/batch-upload', authMiddleware, async (req, res) => {
  try {
    const { project_id } = req.query;
    
    if (!project_id) {
      return res.status(400).json({ error: 'project_id required' });
    }
    
    const db = admin.firestore();
    const projectDoc = await db.collection('projects').doc(project_id).get();
    
    if (!projectDoc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = projectDoc.data();
    if (project.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    if (!req.files || !req.files.files) {
      return res.status(400).json({ error: 'No files uploaded' });
    }
    
    const files = Array.isArray(req.files.files) ? req.files.files : [req.files.files];
    const jobIds = [];
    
    for (const file of files) {
      const jobId = uuidv4();
      const destPath = path.join(UPLOAD_DIR, `${project_id}_${file.name}`);
      
      await file.mv(destPath);
      
      const jobData = {
        id: jobId,
        project_id,
        user_id: req.user.uid,
        filename: file.name,
        original_path: destPath,
        status: 'pending',
        created_at: admin.firestore.FieldValue.serverTimestamp()
      };
      
      await db.collection('jobs').doc(jobId).set(jobData);
      jobIds.push(jobId);
      
      // Queue processing
      processJob(jobId, destPath).catch(err => console.error('Job processing error:', err));
    }
    
    // Update project file count
    await db.collection('projects').doc(project_id).update({
      file_count: admin.firestore.FieldValue.increment(files.length),
      updated_at: admin.firestore.FieldValue.serverTimestamp()
    });
    
    res.json({ status: 'ok', job_ids: jobIds, count: jobIds.length });
  } catch (error) {
    console.error('Batch upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Batch upload ZIP
router.post('/batch-upload-zip', authMiddleware, async (req, res) => {
  try {
    const { project_id } = req.query;
    
    if (!project_id) {
      return res.status(400).json({ error: 'project_id required' });
    }
    
    const db = admin.firestore();
    const projectDoc = await db.collection('projects').doc(project_id).get();
    
    if (!projectDoc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = projectDoc.data();
    if (project.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    if (!req.files || !req.files.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const zipFile = req.files.file;
    const tempZipPath = path.join(UPLOAD_DIR, `temp_${Date.now()}.zip`);
    
    await zipFile.mv(tempZipPath);
    
    const zip = new AdmZip(tempZipPath);
    const zipEntries = zip.getEntries();
    const jobIds = [];
    
    for (const entry of zipEntries) {
      if (entry.isDirectory || entry.entryName.endsWith('.zip')) {
        continue;
      }
      
      const jobId = uuidv4();
      const filename = path.basename(entry.entryName);
      const destPath = path.join(UPLOAD_DIR, `${project_id}_${filename}`);
      
      zip.extractEntryTo(entry, UPLOAD_DIR, false, true, false, filename);
      fs.renameSync(path.join(UPLOAD_DIR, filename), destPath);
      
      const jobData = {
        id: jobId,
        project_id,
        user_id: req.user.uid,
        filename,
        original_path: destPath,
        status: 'pending',
        created_at: admin.firestore.FieldValue.serverTimestamp()
      };
      
      await db.collection('jobs').doc(jobId).set(jobData);
      jobIds.push(jobId);
      
      // Queue processing
      processJob(jobId, destPath).catch(err => console.error('Job processing error:', err));
    }
    
    // Clean up temp zip
    fs.unlinkSync(tempZipPath);
    
    // Update project file count
    await db.collection('projects').doc(project_id).update({
      file_count: admin.firestore.FieldValue.increment(jobIds.length),
      updated_at: admin.firestore.FieldValue.serverTimestamp()
    });
    
    res.json({ status: 'ok', job_ids: jobIds, count: jobIds.length });
  } catch (error) {
    console.error('ZIP upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Process job (background task)
async function processJob(jobId, filePath) {
  try {
    const db = admin.firestore();
    const jobRef = db.collection('jobs').doc(jobId);
    
    // Update status to processing
    await jobRef.update({ status: 'processing' });
    
    // Call AI service for sanitization
    const sanitizeResponse = await axios.post(`${AI_SERVICE_URL}/sanitize`, {
      file_path: filePath
    });
    
    const features = sanitizeResponse.data.features;
    await jobRef.update({ sanitized_features: features });
    
    // Call AI service for decompilation
    const decompileResponse = await axios.post(`${AI_SERVICE_URL}/decompile`, {
      features
    });
    
    const source = decompileResponse.data.source;
    
    // Update job with results
    await jobRef.update({
      status: 'completed',
      decompiled_source: source,
      completed_at: admin.firestore.FieldValue.serverTimestamp()
    });
  } catch (error) {
    console.error('Process job error:', error);
    const db = admin.firestore();
    await db.collection('jobs').doc(jobId).update({
      status: 'failed',
      error_message: error.message,
      completed_at: admin.firestore.FieldValue.serverTimestamp()
    });
  }
}

module.exports = router;
