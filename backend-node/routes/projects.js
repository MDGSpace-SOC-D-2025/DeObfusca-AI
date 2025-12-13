const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const { v4: uuidv4 } = require('uuid');
const authMiddleware = require('../middleware/auth');

// Create project
router.post('/', authMiddleware, async (req, res) => {
  try {
    const { name, description } = req.body;
    
    if (!name) {
      return res.status(400).json({ error: 'name required' });
    }
    
    const db = admin.firestore();
    const projectId = uuidv4();
    
    const projectData = {
      id: projectId,
      user_id: req.user.uid,
      name,
      description: description || null,
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
      file_count: 0,
      status: 'pending'
    };
    
    await db.collection('projects').doc(projectId).set(projectData);
    
    res.json({ status: 'ok', project: projectData });
  } catch (error) {
    console.error('Create project error:', error);
    res.status(500).json({ error: error.message });
  }
});

// List projects
router.get('/', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const snapshot = await db.collection('projects')
      .where('user_id', '==', req.user.uid)
      .orderBy('updated_at', 'desc')
      .limit(50)
      .get();
    
    const projects = [];
    snapshot.forEach(doc => projects.push(doc.data()));
    
    res.json({ projects });
  } catch (error) {
    console.error('List projects error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get project
router.get('/:id', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const doc = await db.collection('projects').doc(req.params.id).get();
    
    if (!doc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = doc.data();
    if (project.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    res.json(project);
  } catch (error) {
    console.error('Get project error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Delete project
router.delete('/:id', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const projectRef = db.collection('projects').doc(req.params.id);
    const doc = await projectRef.get();
    
    if (!doc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = doc.data();
    if (project.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    // Delete all jobs
    const jobsSnapshot = await db.collection('jobs')
      .where('project_id', '==', req.params.id)
      .get();
    
    const batch = db.batch();
    jobsSnapshot.forEach(doc => batch.delete(doc.ref));
    batch.delete(projectRef);
    
    await batch.commit();
    
    res.json({ status: 'deleted' });
  } catch (error) {
    console.error('Delete project error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get project jobs
router.get('/:id/jobs', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const projectDoc = await db.collection('projects').doc(req.params.id).get();
    
    if (!projectDoc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = projectDoc.data();
    if (project.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    const snapshot = await db.collection('jobs')
      .where('project_id', '==', req.params.id)
      .orderBy('created_at', 'desc')
      .limit(100)
      .get();
    
    const jobs = [];
    snapshot.forEach(doc => jobs.push(doc.data()));
    
    res.json({ jobs });
  } catch (error) {
    console.error('Get project jobs error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
