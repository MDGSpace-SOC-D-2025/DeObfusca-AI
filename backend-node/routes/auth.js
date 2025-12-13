const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');

// Register or update user
router.post('/register', async (req, res) => {
  try {
    const { uid, email, displayName } = req.body;
    
    if (!uid || !email) {
      return res.status(400).json({ error: 'uid and email required' });
    }
    
    const db = admin.firestore();
    const userRef = db.collection('users').doc(uid);
    
    const userData = {
      uid,
      email,
      display_name: displayName || null,
      created_at: admin.firestore.FieldValue.serverTimestamp()
    };
    
    await userRef.set(userData, { merge: true });
    
    res.json({ status: 'ok', user: userData });
  } catch (error) {
    console.error('Register error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
