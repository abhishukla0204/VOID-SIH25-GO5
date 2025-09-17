import React, { useState, useEffect } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  PhotoCamera as CameraIcon,
  Assessment as AssessmentIcon,
  WaterDrop as WaterDropIcon,
  Thermostat as ThermostatIcon,
  Terrain as TerrainIcon,
  Timeline as VibrationIcon,
  Shield as ShieldIcon
} from '@mui/icons-material'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { motion } from 'framer-motion'

const Dashboard = ({ systemStatus, connectionStatus, lastMessage }) => {
  const [recentActivities, setRecentActivities] = useState([])
  const [riskTrends, setRiskTrends] = useState([])
  const [detectionStats, setDetectionStats] = useState({
    totalDetections: 0,
    averageConfidence: 0,
    processedImages: 0
  })
  const [environmentalData, setEnvironmentalData] = useState({
    rainfall: 0,
    temperature: 0,
    fractureDensity: 0,
    seismicActivity: 0,
    currentRisk: 0, // Now a number (percentage)
    riskLevel: 'LOW', // String for risk level
    riskScore: 0
  })
  
  // Mock data for demonstration
  useEffect(() => {
    const mockRiskData = Array.from({ length: 24 }, (_, i) => ({
      hour: `${i}:00`,
      risk: Math.random() * 0.8,
      detections: Math.floor(Math.random() * 5)
    }))
    setRiskTrends(mockRiskData)
    
    const mockActivities = [
      { time: '14:30', type: 'detection', message: 'Rock detected with 95% confidence', severity: 'info' },
      { time: '14:15', type: 'risk', message: 'Risk level increased to MEDIUM', severity: 'warning' },
      { time: '13:45', type: 'system', message: 'All models loaded successfully', severity: 'success' },
      { time: '13:30', type: 'detection', message: '3 rocks detected in sector A', severity: 'info' },
    ]
    setRecentActivities(mockActivities)
    
    // Simulate environmental data updates
    const updateEnvironmentalData = () => {
      const baseRainfall = 15 + Math.random() * 20
      const baseTemp = 18 + Math.random() * 12
      const baseFracture = 1.5 + Math.random() * 2
      const baseSeismic = Math.random() * 3
      const riskScore = (baseRainfall / 35 + baseFracture / 3.5 + baseSeismic / 3) / 3
      
      setEnvironmentalData({
        rainfall: baseRainfall,
        temperature: baseTemp,
        fractureDensity: baseFracture,
        seismicActivity: baseSeismic,
        currentRisk: riskScore * 100, // Convert to percentage
        riskLevel: riskScore > 0.7 ? 'HIGH' : riskScore > 0.4 ? 'MEDIUM' : 'LOW',
        riskScore: riskScore
      })
    }
    
    updateEnvironmentalData()
    const interval = setInterval(updateEnvironmentalData, 10000) // Update every 10 seconds
    
    return () => clearInterval(interval)
  }, [])
  
  // Update stats based on real WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage)
        
        if (data.type === 'detection_update') {
          setDetectionStats(prev => ({
            totalDetections: prev.totalDetections + data.data.total_detections,
            averageConfidence: data.data.detections.length > 0 
              ? data.data.detections.reduce((sum, det) => sum + det.confidence, 0) / data.data.detections.length
              : prev.averageConfidence,
            processedImages: prev.processedImages + 1
          }))
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error)
      }
    }
  }, [lastMessage])
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
        return <CheckCircleIcon sx={{ color: '#10b981' }} />
      case 'warning':
        return <WarningIcon sx={{ color: '#f59e0b' }} />
      case 'error':
        return <ErrorIcon sx={{ color: '#ef4444' }} />
      default:
        return <WarningIcon sx={{ color: '#6b7280' }} />
    }
  }
  
  const getActivityIcon = (type) => {
    switch (type) {
      case 'detection':
        return <CameraIcon sx={{ color: '#8b5cf6' }} />
      case 'risk':
        return <AssessmentIcon sx={{ color: '#f59e0b' }} />
      case 'system':
        return <SecurityIcon sx={{ color: '#10b981' }} />
      default:
        return <CheckCircleIcon sx={{ color: '#6b7280' }} />
    }
  }
  
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'success': return '#10b981'
      case 'warning': return '#f59e0b'
      case 'error': return '#ef4444'
      default: return '#3b82f6'
    }
  }
  
  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 1 }}>
          System Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time monitoring of rockfall detection and prediction system
        </Typography>
      </Box>
      
      {/* Status Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SecurityIcon sx={{ color: '#3b82f6', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    System Status
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(systemStatus.status)}
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {systemStatus.status || 'Loading'}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  All systems operational
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <MemoryIcon sx={{ color: '#10b981', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Models Loaded
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {Object.values(systemStatus.models_loaded || {}).filter(Boolean).length} / {Object.keys(systemStatus.models_loaded || {}).length}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Object.keys(systemStatus.models_loaded || {}).length > 0 
                    ? (Object.values(systemStatus.models_loaded || {}).filter(Boolean).length / Object.keys(systemStatus.models_loaded || {}).length) * 100 
                    : 0
                  }
                  sx={{ mt: 1, height: 6, borderRadius: 3 }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUpIcon sx={{ color: '#8b5cf6', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Detections Today
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {detectionStats.totalDetections}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {detectionStats.processedImages} images processed
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SpeedIcon sx={{ color: '#f59e0b', mr: 1 }} />
                  <Typography variant="h6" component="div">
                    Avg Confidence
                  </Typography>
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {(detectionStats.averageConfidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Detection accuracy
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      {/* Environmental Monitoring Section */}
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600, color: '#1976d2' }}>
        Environmental & Sensor Data
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #2196F3, #1976D2)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <WaterDropIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Rainfall
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.rainfall.toFixed(1)} mm
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Last 24 hours
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #FF9800, #F57C00)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <ThermostatIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Temperature
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.temperature.toFixed(1)}°C
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Current ambient
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #795548, #5D4037)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TerrainIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Fracture Density
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.fractureDensity.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Fractures/m²
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55 }}
          >
            <Card 
              sx={{ 
                background: 'linear-gradient(135deg, #E91E63, #C2185B)',
                color: 'white',
                height: '100%',
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateY(-4px)' }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <VibrationIcon sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Seismic Activity
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {environmentalData.seismicActivity.toFixed(1)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Magnitude (Richter)
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      {/* Current Risk Assessment */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card 
              sx={{ 
                background: `linear-gradient(135deg, ${
                  environmentalData.currentRisk > 70 ? '#D32F2F, #B71C1C' :
                  environmentalData.currentRisk > 40 ? '#F57C00, #E65100' :
                  '#388E3C, #2E7D32'
                })`,
                color: 'white',
                textAlign: 'center',
                py: 3,
                transition: 'transform 0.3s',
                '&:hover': { transform: 'scale(1.02)' }
              }}
            >
              <CardContent>
                <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
                  Current Risk Level
                </Typography>
                <Typography variant="h2" sx={{ fontWeight: 700, mb: 2 }}>
                  {environmentalData.currentRisk.toFixed(1)}%
                </Typography>
                <Typography variant="h6" sx={{ opacity: 0.9 }}>
                  {environmentalData.riskLevel} RISK
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                  Risk Trends (24 Hours)
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={riskTrends}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="hour" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1e293b', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="risk" 
                        stroke="#3b82f6" 
                        fill="rgba(59, 130, 246, 0.2)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card className="glass-card" sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                  Connection Status
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Chip 
                    label={connectionStatus}
                    color={connectionStatus === 'Connected' ? 'success' : 'warning'}
                    sx={{ mb: 2 }}
                  />
                  
                  <Typography variant="body2" color="text.secondary">
                    Active connections: {systemStatus.active_connections || 0}
                  </Typography>
                </Box>
                
                <Typography variant="subtitle2" sx={{ mb: 2 }}>
                  Model Status
                </Typography>
                
                {Object.entries(systemStatus.models_loaded || {}).map(([model, loaded]) => (
                  <Box key={model} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    {loaded ? (
                      <CheckCircleIcon sx={{ color: '#10b981', fontSize: 16, mr: 1 }} />
                    ) : (
                      <ErrorIcon sx={{ color: '#ef4444', fontSize: 16, mr: 1 }} />
                    )}
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {model.replace('_', ' ')}
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      
      {/* Recent Activities */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <Card className="glass-card">
          <CardContent>
            <Typography variant="h6" component="div" sx={{ mb: 2 }}>
              Recent Activities
            </Typography>
            
            <List>
              {recentActivities.map((activity, index) => (
                <React.Fragment key={index}>
                  <ListItem>
                    <ListItemIcon>
                      {getActivityIcon(activity.type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={activity.message}
                      secondary={activity.time}
                      secondaryTypographyProps={{
                        color: getSeverityColor(activity.severity)
                      }}
                    />
                  </ListItem>
                  {index < recentActivities.length - 1 && (
                    <Divider variant="inset" component="li" sx={{ borderColor: '#334155' }} />
                  )}
                </React.Fragment>
              ))}
            </List>
          </CardContent>
        </Card>
      </motion.div>
    </Box>
  )
}

export default Dashboard