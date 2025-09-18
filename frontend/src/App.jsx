import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import config, { apiRequest, getCurrentBackendInfo } from './config/api'
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box, 
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  IconButton,
  Badge,
  Chip,
  useMediaQuery,
  useTheme
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  PhotoCamera as CameraIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Videocam as VideocamIcon,
  Terrain as TerrainIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'

// Import page components
import Dashboard from './pages/Dashboard'
import Detection from './pages/Detection'
import RiskAssessment from './pages/RiskAssessment'
import Settings from './pages/Settings'
import LiveMonitoring from './pages/LiveMonitoring'
import DEMAnalysis from './pages/DEMAnalysis'

// WebSocket hook for real-time updates
import useWebSocket from './hooks/useWebSocket'

const drawerWidth = 280

function App() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState({
    status: 'loading',
    models_loaded: {},
    active_connections: 0
  })
  
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  
  // WebSocket connection for real-time updates
  const { connectionStatus, lastMessage, currentUrl, reconnect, isConnected } = useWebSocket('/ws')
  
  // Handle drawer toggle
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen)
  }
  
  // Navigation items
  const navigationItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: 'dashboard',
      color: '#3b82f6'
    },
    {
      text: 'Live Monitoring',
      icon: <VideocamIcon />,
      path: 'live-monitoring',
      color: '#ef4444'
    },
    {
      text: 'DEM Analysis',
      icon: <TerrainIcon />,
      path: 'dem-analysis',
      color: '#10b981'
    },
    {
      text: 'Rock Detection',
      icon: <CameraIcon />,
      path: 'detection',
      color: '#8b5cf6'
    },
    {
      text: 'Risk Assessment',
      icon: <AssessmentIcon />,
      path: 'risk-assessment',
      color: '#f59e0b'
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: 'settings',
      color: '#6b7280'
    }
  ]
  
  // Fetch system status
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const data = await apiRequest('/api/status')
        console.log('🔄 System status received:', data)
        console.log('📊 Models loaded:', data.models_loaded)
        setSystemStatus(data)
      } catch (error) {
        console.error('Failed to fetch system status:', error)
        // Don't show toast error for status checks as they happen frequently
      }
    }
    
    fetchSystemStatus()
    const interval = setInterval(fetchSystemStatus, 30000) // Update every 30s
    
    return () => clearInterval(interval)
  }, [])
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage)
        
        switch (data.type) {
          case 'risk_update':
            if (data.data.risk_level === 'HIGH') {
              toast.error(`🚨 HIGH RISK DETECTED: ${data.data.risk_score.toFixed(3)}`, {
                duration: 8000,
              })
            }
            break
            
          case 'detection_update':
            if (data.data.total_detections > 0) {
              toast.success(`🏔️ Detected ${data.data.total_detections} rocks`, {
                duration: 4000,
              })
            }
            break
            
          case 'heartbeat':
            // Connection is alive
            break
            
          default:
            console.log('Unknown message type:', data.type)
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }
  }, [lastMessage])
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational': return 'success'
      case 'loading': return 'warning'
      case 'error': return 'error'
      default: return 'default'
    }
  }
  
  // Get connection status icon
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'Connected':
        return <CheckCircleIcon sx={{ color: '#10b981', fontSize: 20 }} />
      case 'Connecting':
        return <WarningIcon sx={{ color: '#f59e0b', fontSize: 20 }} />
      default:
        return <WarningIcon sx={{ color: '#ef4444', fontSize: 20 }} />
    }
  }
  
  // Drawer content
  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 3, borderBottom: '1px solid #334155' }}>
        <Typography variant="h5" component="div" sx={{ 
          fontWeight: 700,
          background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          mb: 1
        }}>
          🏔️ Rockfall AI
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Detection & Prediction System
        </Typography>
      </Box>
      
      {/* System Status */}
      <Box sx={{ p: 2, borderBottom: '1px solid #334155' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {getConnectionIcon()}
          <Typography variant="body2" color="text.secondary">
            {connectionStatus}
          </Typography>
        </Box>
        <Chip 
          label={systemStatus.status}
          color={getStatusColor(systemStatus.status)}
          size="small"
          sx={{ fontSize: '0.75rem' }}
        />
      </Box>
      
      {/* Navigation */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              selected={currentPage === item.path}
              onClick={() => {
                setCurrentPage(item.path)
                if (isMobile) setMobileOpen(false)
              }}
              sx={{
                mx: 1,
                borderRadius: 2,
                mb: 0.5,
                '&.Mui-selected': {
                  backgroundColor: `${item.color}20`,
                  '&:hover': {
                    backgroundColor: `${item.color}30`,
                  }
                }
              }}
            >
              <ListItemIcon sx={{ 
                color: currentPage === item.path ? item.color : 'text.secondary',
                minWidth: 40 
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: currentPage === item.path ? 600 : 400,
                  color: currentPage === item.path ? item.color : 'text.primary'
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      {/* Footer */}
      <Box sx={{ p: 2, borderTop: '1px solid #334155' }}>
        <Typography variant="caption" color="text.secondary" display="block">
          v1.0.0 • Models Ready
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {Object.values(systemStatus.models_loaded || {}).filter(Boolean).length} / {Object.keys(systemStatus.models_loaded || {}).length} models loaded
        </Typography>
      </Box>
    </Box>
  )
  
  // Render current page component
  const renderCurrentPage = () => {
    const pageProps = {
      systemStatus,
      connectionStatus,
      lastMessage
    }
    
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...pageProps} />
      case 'live-monitoring':
        return <LiveMonitoring {...pageProps} />
      case 'dem-analysis':
        return <DEMAnalysis {...pageProps} />
      case 'detection':
        return <Detection {...pageProps} />
      case 'risk-assessment':
        return <RiskAssessment {...pageProps} />
      case 'settings':
        return <Settings {...pageProps} />
      default:
        return <Dashboard {...pageProps} />
    }
  }
  
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find(item => item.path === currentPage)?.text || 'Dashboard'}
          </Typography>
          
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <Badge badgeContent={systemStatus.active_connections} color="secondary">
              <NotificationsIcon />
            </Badge>
          </IconButton>
          
          <Chip 
            label={connectionStatus}
            color={connectionStatus === 'Connected' ? 'success' : 'warning'}
            variant="outlined"
            size="small"
            sx={{ 
              borderColor: 'rgba(255,255,255,0.3)',
              color: 'white',
              fontSize: '0.75rem'
            }}
          />
        </Toolbar>
      </AppBar>
      
      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: '#1e293b',
              borderRight: '1px solid #334155'
            },
          }}
        >
          {drawerContent}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: '#1e293b',
              borderRight: '1px solid #334155'
            },
          }}
          open
        >
          {drawerContent}
        </Drawer>
      </Box>
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: '#0f172a',
        }}
      >
        <Toolbar />
        <Container maxWidth="xl" sx={{ py: 3 }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentPage}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderCurrentPage()}
            </motion.div>
          </AnimatePresence>
        </Container>
      </Box>
    </Box>
  )
}

export default App