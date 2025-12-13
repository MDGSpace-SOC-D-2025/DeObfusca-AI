// Analytics routes for advanced metrics and insights
const express = require('express');
const router = express.Router();
const admin = require('firebase-admin');
const authMiddleware = require('../middleware/auth');

// Get user analytics dashboard
router.get('/dashboard', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const userId = req.user.uid;
    
    // Get all user jobs
    const jobsSnapshot = await db.collection('jobs')
      .where('userId', '==', userId)
      .orderBy('createdAt', 'desc')
      .get();
    
    const jobs = jobsSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    
    // Calculate analytics
    const analytics = {
      overview: {
        totalJobs: jobs.length,
        completedJobs: jobs.filter(j => j.status === 'completed').length,
        failedJobs: jobs.filter(j => j.status === 'failed').length,
        inProgressJobs: jobs.filter(j => j.status === 'processing').length
      },
      performance: {
        averageTime: calculateAverageTime(jobs),
        successRate: calculateSuccessRate(jobs),
        totalBinariesProcessed: jobs.length,
        totalLinesDecompiled: jobs.reduce((acc, j) => acc + (j.linesOfCode || 0), 0)
      },
      trends: {
        jobsByDay: groupJobsByDay(jobs),
        successRateByDay: calculateSuccessRateByDay(jobs),
        averageTimeByDay: calculateAverageTimeByDay(jobs)
      },
      topProjects: await getTopProjects(db, userId),
      recentActivity: jobs.slice(0, 10).map(j => ({
        id: j.id,
        filename: j.filename,
        status: j.status,
        createdAt: j.createdAt,
        completedAt: j.completedAt
      }))
    };
    
    res.json(analytics);
  } catch (error) {
    console.error('Error fetching analytics:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get AI model performance metrics
router.get('/ai-performance', authMiddleware, async (req, res) => {
  try {
    const db = admin.firestore();
    const userId = req.user.uid;
    
    // Get jobs with AI metrics
    const jobsSnapshot = await db.collection('jobs')
      .where('userId', '==', userId)
      .where('status', '==', 'completed')
      .get();
    
    const jobs = jobsSnapshot.docs.map(doc => doc.data());
    
    const aiMetrics = {
      modelAccuracy: {
        gnnSanitizer: calculateModelAccuracy(jobs, 'gnn'),
        llmDecompiler: calculateModelAccuracy(jobs, 'llm'),
        rlVerifier: calculateModelAccuracy(jobs, 'rl')
      },
      refinementStats: {
        averageIterations: jobs.reduce((acc, j) => acc + (j.refinementIterations || 1), 0) / jobs.length,
        successfulRefinements: jobs.filter(j => (j.refinementIterations || 0) > 1).length,
        maxIterationsReached: jobs.filter(j => j.refinementIterations >= 3).length
      },
      codeQuality: {
        averageComplexity: jobs.reduce((acc, j) => acc + (j.cyclomaticComplexity || 0), 0) / jobs.length,
        averageFunctions: jobs.reduce((acc, j) => acc + (j.functionsCount || 0), 0) / jobs.length,
        syntaxErrorRate: jobs.filter(j => j.hasSyntaxErrors).length / jobs.length * 100
      }
    };
    
    res.json(aiMetrics);
  } catch (error) {
    console.error('Error fetching AI metrics:', error);
    res.status(500).json({ error: error.message });
  }
});

// Compare two decompilations
router.post('/compare', authMiddleware, async (req, res) => {
  try {
    const { jobId1, jobId2 } = req.body;
    const db = admin.firestore();
    
    const [job1Doc, job2Doc] = await Promise.all([
      db.collection('jobs').doc(jobId1).get(),
      db.collection('jobs').doc(jobId2).get()
    ]);
    
    if (!job1Doc.exists || !job2Doc.exists) {
      return res.status(404).json({ error: 'One or both jobs not found' });
    }
    
    const job1 = job1Doc.data();
    const job2 = job2Doc.data();
    
    // Calculate differences
    const comparison = {
      job1: {
        id: jobId1,
        filename: job1.filename,
        linesOfCode: job1.linesOfCode || 0,
        functionsCount: job1.functionsCount || 0,
        processingTime: job1.processingTime || 0
      },
      job2: {
        id: jobId2,
        filename: job2.filename,
        linesOfCode: job2.linesOfCode || 0,
        functionsCount: job2.functionsCount || 0,
        processingTime: job2.processingTime || 0
      },
      differences: {
        linesOfCodeDiff: (job2.linesOfCode || 0) - (job1.linesOfCode || 0),
        functionsDiff: (job2.functionsCount || 0) - (job1.functionsCount || 0),
        timeDiff: (job2.processingTime || 0) - (job1.processingTime || 0)
      }
    };
    
    res.json(comparison);
  } catch (error) {
    console.error('Error comparing jobs:', error);
    res.status(500).json({ error: error.message });
  }
});

// Helper functions
function calculateAverageTime(jobs) {
  const completedJobs = jobs.filter(j => j.status === 'completed' && j.processingTime);
  if (completedJobs.length === 0) return 0;
  return completedJobs.reduce((acc, j) => acc + j.processingTime, 0) / completedJobs.length;
}

function calculateSuccessRate(jobs) {
  if (jobs.length === 0) return 0;
  return (jobs.filter(j => j.status === 'completed').length / jobs.length * 100).toFixed(2);
}

function groupJobsByDay(jobs) {
  const grouped = {};
  jobs.forEach(job => {
    if (job.createdAt) {
      const date = job.createdAt.toDate().toISOString().split('T')[0];
      grouped[date] = (grouped[date] || 0) + 1;
    }
  });
  return grouped;
}

function calculateSuccessRateByDay(jobs) {
  const grouped = {};
  jobs.forEach(job => {
    if (job.createdAt) {
      const date = job.createdAt.toDate().toISOString().split('T')[0];
      if (!grouped[date]) grouped[date] = { total: 0, completed: 0 };
      grouped[date].total++;
      if (job.status === 'completed') grouped[date].completed++;
    }
  });
  
  Object.keys(grouped).forEach(date => {
    grouped[date].rate = (grouped[date].completed / grouped[date].total * 100).toFixed(2);
  });
  
  return grouped;
}

function calculateAverageTimeByDay(jobs) {
  const grouped = {};
  jobs.forEach(job => {
    if (job.createdAt && job.processingTime) {
      const date = job.createdAt.toDate().toISOString().split('T')[0];
      if (!grouped[date]) grouped[date] = { total: 0, count: 0 };
      grouped[date].total += job.processingTime;
      grouped[date].count++;
    }
  });
  
  Object.keys(grouped).forEach(date => {
    grouped[date].average = (grouped[date].total / grouped[date].count).toFixed(2);
  });
  
  return grouped;
}

async function getTopProjects(db, userId) {
  const projectsSnapshot = await db.collection('projects')
    .where('userId', '==', userId)
    .get();
  
  const projects = await Promise.all(projectsSnapshot.docs.map(async (doc) => {
    const jobsSnapshot = await db.collection('jobs')
      .where('projectId', '==', doc.id)
      .get();
    
    return {
      id: doc.id,
      name: doc.data().name,
      jobCount: jobsSnapshot.size
    };
  }));
  
  return projects.sort((a, b) => b.jobCount - a.jobCount).slice(0, 5);
}

function calculateModelAccuracy(jobs, modelType) {
  const relevant = jobs.filter(j => j.aiMetrics && j.aiMetrics[modelType]);
  if (relevant.length === 0) return 0;
  return (relevant.reduce((acc, j) => acc + (j.aiMetrics[modelType].accuracy || 0), 0) / relevant.length * 100).toFixed(2);
}

module.exports = router;
