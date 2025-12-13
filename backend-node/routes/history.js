const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const authMiddleware = require('../middleware/auth');

// Get user history
router.get('/', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const snapshot = await db.collection('jobs')
      .where('user_id', '==', req.user.uid)
      .orderBy('created_at', 'desc')
      .limit(50)
      .get();
    
    const jobs = [];
    snapshot.forEach(doc => jobs.push(doc.data()));
    
    res.json({ jobs, total: jobs.length });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
