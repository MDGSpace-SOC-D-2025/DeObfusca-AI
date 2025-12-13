const express = require('express');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const admin = require('firebase-admin');
const axios = require('axios');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const projectRoutes = require('./routes/projects');
const jobRoutes = require('./routes/jobs');
const uploadRoutes = require('./routes/upload');
const historyRoutes = require('./routes/history');

const app = express();
const PORT = process.env.PORT || 8000;

// Initialize Firebase Admin
const initFirebase = () => {
  const emulatorHost = process.env.FIRESTORE_EMULATOR_HOST;
  
  if (emulatorHost) {
    // Use emulator
    if (!admin.apps.length) {
      admin.initializeApp({
        projectId: 'demo-project'
      });
    }
    console.log(`Firebase initialized with emulator: ${emulatorHost}`);
  } else {
    // Production: use service account
    const serviceAccount = require(process.env.FIREBASE_CREDENTIALS || './firebase-credentials.json');
    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert(serviceAccount)
      });
    }
    console.log('Firebase initialized with credentials');
  }
};

initFirebase();

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload({
  createParentPath: true,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB max
}));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/projects', projectRoutes);
app.use('/api/jobs', jobRoutes);
app.use('/api', uploadRoutes);
app.use('/api/history', historyRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Profile endpoint
app.get('/api/profile', require('./middleware/auth'), async (req, res) => {
  try {
    const db = admin.firestore();
    const userDoc = await db.collection('users').doc(req.user.uid).get();
    
    if (!userDoc.exists) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(userDoc.data());
  } catch (error) {
    console.error('Error getting profile:', error);
    res.status(500).json({ error: error.message });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;
