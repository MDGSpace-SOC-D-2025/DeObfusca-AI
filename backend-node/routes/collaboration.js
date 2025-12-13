// Collaboration routes for team features
const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const authMiddleware = require('../middleware/auth');

// Share project with another user
router.post('/share', authMiddleware, async (req, res) => {
  try {
    const { projectId, targetEmail, permission } = req.body;
    const db = admin.firestore();
    
    // Verify project ownership
    const projectDoc = await db.collection('projects').doc(projectId).get();
    if (!projectDoc.exists) {
      return res.status(404).json({ error: 'Project not found' });
    }
    
    const project = projectDoc.data();
    if (project.userId !== req.user.uid) {
      return res.status(403).json({ error: 'Not authorized to share this project' });
    }
    
    // Find target user by email
    const targetUser = await admin.auth().getUserByEmail(targetEmail);
    
    // Create share record
    await db.collection('shares').add({
      projectId,
      ownerId: req.user.uid,
      targetUserId: targetUser.uid,
      permission: permission || 'view', // 'view' or 'edit'
      createdAt: admin.firestore.FieldValue.serverTimestamp()
    });
    
    // Notify via WebSocket
    const io = req.app.get('io');
    io.to(`user_${targetUser.uid}`).emit('project_shared', {
      projectId,
      projectName: project.name,
      sharedBy: req.user.email
    });
    
    res.json({ success: true, message: 'Project shared successfully' });
  } catch (error) {
    console.error('Error sharing project:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get shared projects
router.get('/shared', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    
    // Get projects shared with this user
    const sharesSnapshot = await db.collection('shares')
      .where('targetUserId', '==', req.user.uid)
      .get();
    
    const sharedProjects = await Promise.all(sharesSnapshot.docs.map(async (doc) => {
      const share = doc.data();
      const projectDoc = await db.collection('projects').doc(share.projectId).get();
      const ownerDoc = await db.collection('users').doc(share.ownerId).get();
      
      return {
        id: share.projectId,
        ...projectDoc.data(),
        sharedBy: ownerDoc.data()?.email || 'Unknown',
        permission: share.permission,
        sharedAt: share.createdAt
      };
    }));
    
    res.json({ sharedProjects });
  } catch (error) {
    console.error('Error fetching shared projects:', error);
    res.status(500).json({ error: error.message });
  }
});

// Add comment to a job
router.post('/comment', authMiddleware, async (req, res) => {
  try {
    const { jobId, comment } = req.body;
    const db = admin.firestore();
    
    const commentDoc = await db.collection('comments').add({
      jobId,
      userId: req.user.uid,
      userEmail: req.user.email,
      comment,
      createdAt: admin.firestore.FieldValue.serverTimestamp()
    });
    
    // Notify collaborators via WebSocket
    const io = req.app.get('io');
    const jobDoc = await db.collection('jobs').doc(jobId).get();
    if (jobDoc.exists) {
      const job = jobDoc.data();
      io.to(`project_${job.projectId}`).emit('new_comment', {
        jobId,
        comment,
        user: req.user.email,
        timestamp: new Date()
      });
    }
    
    res.json({ success: true, commentId: commentDoc.id });
  } catch (error) {
    console.error('Error adding comment:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get comments for a job
router.get('/comments/:jobId', authMiddleware, async (req, res) => {
  try {
    const { jobId } = req.params;
    const db = admin.firestore();
    
    const commentsSnapshot = await db.collection('comments')
      .where('jobId', '==', jobId)
      .orderBy('createdAt', 'desc')
      .get();
    
    const comments = commentsSnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
    
    res.json({ comments });
  } catch (error) {
    console.error('Error fetching comments:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
