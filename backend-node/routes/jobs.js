const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const authMiddleware = require('../middleware/auth');

// Get job
router.get('/:id', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const doc = await db.collection('jobs').doc(req.params.id).get();
    
    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    const job = doc.data();
    if (job.user_id !== req.user.uid) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    res.json(job);
  } catch (error) {
    console.error('Get job error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
