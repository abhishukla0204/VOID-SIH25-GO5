import React, { useState, useEffect } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Divider,
  Stack,
  LinearProgress
} from '@mui/material'
import {
  Terrain as TerrainIcon,
  Map as MapIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Info as InfoIcon,
  Palette as PaletteIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

const DEMAnalysis = () => {
  const [selectedDEM, setSelectedDEM] = useState('')
  const [demData, setDemData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [imageLoaded, setImageLoaded] = useState(false)

  // Available DEM files
  const demFiles = [
    {
      id: 'bingham_canyon',
      name: 'Bingham Canyon Mine',
      location: 'Utah, USA',
      description: 'Large open-pit copper mine with significant terrain variations'
    },
    {
      id: 'chuquicamata',
      name: 'Chuquicamata Copper Mine',
      location: 'Chile',
      description: 'One of the largest open-pit mines in the world'
    },
    {
      id: 'grasberg',
      name: 'Grasberg Mine',
      location: 'Papua, Indonesia',
      description: 'High-altitude mining operation in mountainous terrain'
    }
  ]

  // Color scale information
  const colorScale = [
    { color: '#2D5016', elevation: 'Low', description: 'Valley floors, water bodies' },
    { color: '#4F7942', elevation: 'Low-Medium', description: 'Gentle slopes, plains' },
    { color: '#8FBC8F', elevation: 'Medium', description: 'Rolling hills, moderate terrain' },
    { color: '#DAA520', elevation: 'Medium-High', description: 'Steep slopes, ridges' },
    { color: '#CD853F', elevation: 'High', description: 'Mountain slopes, cliffs' },
    { color: '#A0522D', elevation: 'Very High', description: 'Rocky outcrops, peaks' },
    { color: '#FFFFFF', elevation: 'Highest', description: 'Mountain peaks, extreme elevation' }
  ]

  // Fetch DEM data when selection changes
  useEffect(() => {
    if (selectedDEM) {
      fetchDEMData()
    }
  }, [selectedDEM])

  const fetchDEMData = async () => {
    setLoading(true)
    setError(null)
    setImageLoaded(false)
    
    try {
      const response = await fetch(`/api/dem/analyze/${selectedDEM}`)
      if (!response.ok) {
        throw new Error('Failed to fetch DEM data')
      }
      
      const data = await response.json()
      setDemData(data)
    } catch (err) {
      setError(err.message)
      setDemData(null)
    } finally {
      setLoading(false)
    }
  }

  const handleDEMChange = (event) => {
    setSelectedDEM(event.target.value)
  }

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 3))
  }

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.5))
  }

  const handleDownload = () => {
    if (demData && demData.image_url) {
      const link = document.createElement('a')
      link.href = demData.image_url
      link.download = `${selectedDEM}_elevation_map.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const StatisticCard = ({ title, value, unit, color, icon }) => (
    <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}>
      <CardContent sx={{ textAlign: 'center', py: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
          {icon}
        </Box>
        <Typography variant="h5" sx={{ color: color, fontWeight: 700, mb: 0.5 }}>
          {value}
        </Typography>
        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
          {unit}
        </Typography>
        <Typography variant="body2" sx={{ color: 'white', mt: 1, fontWeight: 500 }}>
          {title}
        </Typography>
      </CardContent>
    </Card>
  )

  const ColorScaleLegend = () => (
    <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <PaletteIcon sx={{ color: '#3b82f6', mr: 1 }} />
          <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
            Elevation Color Scale
          </Typography>
        </Box>
        <Stack spacing={1}>
          {colorScale.map((item, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  width: 20,
                  height: 20,
                  backgroundColor: item.color,
                  borderRadius: 1,
                  border: '1px solid #334155'
                }}
              />
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ color: 'white', fontWeight: 500 }}>
                  {item.elevation}
                </Typography>
                <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                  {item.description}
                </Typography>
              </Box>
            </Box>
          ))}
        </Stack>
      </CardContent>
    </Card>
  )

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: '#0f172a', color: 'white', p: 3 }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 2, color: '#3b82f6' }}>
            🗺️ Digital Elevation Model (DEM) Analysis
          </Typography>
          <Typography variant="body1" sx={{ color: '#94a3b8', mb: 1 }}>
            Interactive elevation mapping and terrain analysis for rockfall risk assessment
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981', mb: 3, fontWeight: 600 }}>
            🎨 Color-coded elevation visualization from .tif to PNG conversion
          </Typography>
        </Box>
      </motion.div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155', mb: 3 }}>
          <CardContent>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel sx={{ color: '#94a3b8' }}>Select DEM File</InputLabel>
                  <Select
                    value={selectedDEM}
                    onChange={handleDEMChange}
                    sx={{
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#334155'
                      },
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#3b82f6'
                      }
                    }}
                  >
                    {demFiles.map((file) => (
                      <MenuItem key={file.id} value={file.id}>
                        <Box>
                          <Typography variant="body1" sx={{ fontWeight: 600 }}>
                            {file.name}
                          </Typography>
                          <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                            {file.location}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={fetchDEMData}
                    disabled={!selectedDEM || loading}
                    sx={{ color: '#3b82f6', borderColor: '#3b82f6' }}
                  >
                    Refresh
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={handleDownload}
                    disabled={!demData}
                    sx={{ color: '#10b981', borderColor: '#10b981' }}
                  >
                    Download
                  </Button>
                </Box>
              </Grid>

              <Grid item xs={12} md={4}>
                {selectedDEM && (
                  <Box>
                    <Typography variant="body2" sx={{ color: '#94a3b8', mb: 1 }}>
                      Selected: {demFiles.find(f => f.id === selectedDEM)?.name}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#64748b' }}>
                      {demFiles.find(f => f.id === selectedDEM)?.description}
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </motion.div>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* DEM Visualization */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155', minHeight: 600 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                    Elevation Map Visualization
                  </Typography>
                  {demData && (
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={handleZoomOut}
                        sx={{ minWidth: 40, color: '#64748b', borderColor: '#334155' }}
                      >
                        <ZoomOutIcon fontSize="small" />
                      </Button>
                      <Chip 
                        label={`${Math.round(zoomLevel * 100)}%`} 
                        size="small"
                        sx={{ backgroundColor: '#334155', color: 'white' }}
                      />
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={handleZoomIn}
                        sx={{ minWidth: 40, color: '#64748b', borderColor: '#334155' }}
                      >
                        <ZoomInIcon fontSize="small" />
                      </Button>
                    </Box>
                  )}
                </Box>

                <Box
                  sx={{
                    width: '100%',
                    height: 500,
                    backgroundColor: '#0f172a',
                    borderRadius: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  {loading && (
                    <Box sx={{ textAlign: 'center' }}>
                      <CircularProgress sx={{ color: '#3b82f6', mb: 2 }} />
                      <Typography variant="body1" sx={{ color: '#94a3b8' }}>
                        Processing DEM data...
                      </Typography>
                    </Box>
                  )}

                  {error && (
                    <Alert severity="error" sx={{ maxWidth: 400 }}>
                      {error}
                    </Alert>
                  )}

                  {!selectedDEM && !loading && (
                    <Box sx={{ textAlign: 'center' }}>
                      <TerrainIcon sx={{ fontSize: 64, color: '#64748b', mb: 2 }} />
                      <Typography variant="h6" sx={{ color: '#94a3b8', mb: 1 }}>
                        Select a DEM file to view elevation map
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#64748b' }}>
                        Choose from Bingham Canyon, Chuquicamata, or Grasberg mines
                      </Typography>
                    </Box>
                  )}

                  {demData && !loading && (
                    <Box
                      sx={{
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transform: `scale(${zoomLevel})`,
                        transition: 'transform 0.3s ease'
                      }}
                    >
                      <img
                        src={demData.image_url}
                        alt={`${selectedDEM} elevation map`}
                        style={{
                          maxWidth: '100%',
                          maxHeight: '100%',
                          borderRadius: 8,
                          opacity: imageLoaded ? 1 : 0,
                          transition: 'opacity 0.3s ease'
                        }}
                        onLoad={() => setImageLoaded(true)}
                        onError={() => setError('Failed to load elevation map image')}
                      />
                      {!imageLoaded && demData && (
                        <Box
                          sx={{
                            position: 'absolute',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: '100%',
                            height: '100%'
                          }}
                        >
                          <CircularProgress sx={{ color: '#3b82f6' }} />
                        </Box>
                      )}
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Statistics and Color Scale */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            {/* Statistics */}
            {demData && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <InfoIcon sx={{ color: '#3b82f6', mr: 1 }} />
                      <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                        Elevation Statistics
                      </Typography>
                    </Box>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <StatisticCard
                          title="Minimum"
                          value={demData.statistics?.min_elevation || 'N/A'}
                          unit="meters"
                          color="#22c55e"
                          icon={<TerrainIcon sx={{ color: '#22c55e' }} />}
                        />
                      </Grid>
                      <Grid item xs={6}>
                        <StatisticCard
                          title="Maximum"
                          value={demData.statistics?.max_elevation || 'N/A'}
                          unit="meters"
                          color="#f44336"
                          icon={<TerrainIcon sx={{ color: '#f44336' }} />}
                        />
                      </Grid>
                      <Grid item xs={6}>
                        <StatisticCard
                          title="Average"
                          value={demData.statistics?.mean_elevation || 'N/A'}
                          unit="meters"
                          color="#3b82f6"
                          icon={<TerrainIcon sx={{ color: '#3b82f6' }} />}
                        />
                      </Grid>
                      <Grid item xs={6}>
                        <StatisticCard
                          title="Std Dev"
                          value={demData.statistics?.std_elevation || 'N/A'}
                          unit="meters"
                          color="#8b5cf6"
                          icon={<TerrainIcon sx={{ color: '#8b5cf6' }} />}
                        />
                      </Grid>
                    </Grid>

                    <Divider sx={{ my: 2, borderColor: '#334155' }} />
                    
                    <Box>
                      <Typography variant="body2" sx={{ color: '#94a3b8', mb: 1 }}>
                        Terrain Analysis
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
                        Elevation Range: {demData.statistics?.elevation_range || 'N/A'} meters
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
                        Terrain Type: {demData.statistics?.terrain_type || 'Complex'}
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'white' }}>
                        Risk Assessment: {demData.statistics?.risk_level || 'Moderate to High'}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Color Scale Legend */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5, duration: 0.5 }}
            >
              <ColorScaleLegend />
            </motion.div>
          </Stack>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DEMAnalysis