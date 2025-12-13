// Enhanced backend server with WebSocket support, caching, and metrics
const express = require('express');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const admin = require('firebase-admin');
const axios = require('axios');
const { createServer } = require('http');
const { Server } = require('socket.io');
const Redis = require('ioredis');
const compression = require('compression');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const projectRoutes = require('./routes/projects');
const jobRoutes = require('./routes/jobs');
const uploadRoutes = require('./routes/upload');
const historyRoutes = require('./routes/history');
const analyticsRoutes = require('./routes/analytics');
const collaborationRoutes = require('./routes/collaboration');

const app = express();
const httpServer = createServer(app);
const PORT = process.env.PORT || 8000;

// Initialize Redis for caching and pub/sub
let redisClient;
try {
  redisClient = new Redis({
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    retryStrategy: (times) => {
      const delay = Math.min(times * 50, 2000);
      return delay;
    }
  });
  console.log('Redis connected for caching');
} catch (err) {
  console.warn('Redis not available, using in-memory cache');
  redisClient = null;
}

// Initialize Socket.IO for real-time updates
const io = new Server(httpServer, {
  cors: {
    origin: ['http://localhost:3000', 'http://localhost:5173'],
    credentials: true
  }
});

// Initialize Firebase Admin
const initFirebase = () => {
  const emulatorHost = process.env.FIRESTORE_EMULATOR_HOST;
  
  if (emulatorHost) {
    if (!admin.apps.length) {
      admin.initializeApp({
        projectId: 'demo-project'
      });
    }
    console.log(`Firebase initialized with emulator: ${emulatorHost}`);
  } else {
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

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginEmbedderPolicy: false
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

// Compression
app.use(compression());

// CORS
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload({
  createParentPath: true,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB max
  useTempFiles: true,
  tempFileDir: '/tmp/'
}));

// Make io and redis available to routes
app.set('io', io);
app.set('redis', redisClient);

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/projects', projectRoutes);
app.use('/api/jobs', jobRoutes);
app.use('/api', uploadRoutes);
app.use('/api/history', historyRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/collaboration', collaborationRoutes);

// Health check with service status
app.get('/api/health', async (req, res) => {
  const services = {};
  
  // Check orchestrator
  try {
    const resp = await axios.get(`${process.env.ORCHESTRATOR_URL || 'http://orchestrator:5000'}/health`, { timeout: 5000 });
    services.orchestrator = resp.data;
  } catch (err) {
    services.orchestrator = { status: 'error', message: err.message };
  }
  
  // Check Redis
  if (redisClient) {
    try {
      await redisClient.ping();
      services.redis = { status: 'ok' };
    } catch (err) {
      services.redis = { status: 'error' };
    }
  }
  
  res.json({ 
    status: 'ok',
    timestamp: new Date().toISOString(),
    services
  });
});

// Metrics endpoint
app.get('/api/metrics', require('./middleware/auth'), async (req, res) => {
  try {
    const db = admin.firestore();
    
    // Get user's job statistics
    const jobsSnapshot = await db.collection('jobs')
      .where('userId', '==', req.user.uid)
      .get();
    
    const jobs = jobsSnapshot.docs.map(doc => doc.data());
    
    const metrics = {
      totalJobs: jobs.length,
      completedJobs: jobs.filter(j => j.status === 'completed').length,
      failedJobs: jobs.filter(j => j.status === 'failed').length,
      averageProcessingTime: jobs
        .filter(j => j.completedAt && j.createdAt)
        .reduce((acc, j) => acc + (j.completedAt.toMillis() - j.createdAt.toMillis()), 0) / jobs.length || 0,
      successRate: jobs.length > 0 ? (jobs.filter(j => j.status === 'completed').length / jobs.length * 100).toFixed(2) : 0
    };
    
    res.json(metrics);
  } catch (error) {
    console.error('Error getting metrics:', error);
    res.status(500).json({ error: error.message });
  }
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

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Join user-specific room
  socket.on('join', (userId) => {
    socket.join(`user_${userId}`);
    console.log(`User ${userId} joined their room`);
  });
  
  // Join project room for collaboration
  socket.on('join_project', (projectId) => {
    socket.join(`project_${projectId}`);
    console.log(`Socket ${socket.id} joined project ${projectId}`);
  });
  
  // Handle AI chat messages
  socket.on('ai_chat', async (data) => {
    try {
      const { message, context, userId } = data;
      
      // Forward to AI chat service
      const response = await axios.post(`${process.env.ORCHESTRATOR_URL}/chat`, {
        message,
        context
      });
      
      socket.emit('ai_response', response.data);
    } catch (error) {
      socket.emit('ai_error', { error: error.message });
    }
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`Enhanced server running on port ${PORT}`);
  console.log('Features: WebSocket, Redis caching, Rate limiting, Compression');
});

module.exports = { app, io, redisClient };
