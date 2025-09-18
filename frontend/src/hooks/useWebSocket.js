import { useState, useEffect, useRef } from 'react'
import { getWsUrl } from '../config/api'

const useWebSocket = (endpoint) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  const [lastMessage, setLastMessage] = useState(null)
  const [currentUrl, setCurrentUrl] = useState(null)
  const ws = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5
  
  useEffect(() => {
    let reconnectTimeout
    
    const connect = () => {
      try {
        // Get the WebSocket URL from configuration
        const url = getWsUrl(endpoint)
        setCurrentUrl(url)
        
        console.log(`🔌 Attempting WebSocket connection to: ${url}`)
        ws.current = new WebSocket(url)
        
        ws.current.onopen = () => {
          console.log(`✅ WebSocket connected to: ${url}`)
          setConnectionStatus('Connected')
          reconnectAttempts.current = 0 // Reset on successful connection
        }
        
        ws.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
          } catch (error) {
            console.warn('Failed to parse WebSocket message:', event.data)
            setLastMessage(event.data)
          }
        }
        
        ws.current.onclose = () => {
          console.log(`🔌 WebSocket disconnected from: ${url}`)
          setConnectionStatus('Disconnected')
          
          // Attempt to reconnect with exponential backoff
          if (reconnectAttempts.current < maxReconnectAttempts) {
            const delay = Math.pow(2, reconnectAttempts.current) * 1000 // 1s, 2s, 4s, 8s, 16s
            console.log(`🔄 Reconnecting in ${delay}ms... (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`)
            
            reconnectTimeout = setTimeout(() => {
              reconnectAttempts.current++
              
              // Simple reconnect attempt with exponential backoff
              console.log(`🔄 WebSocket reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`)
              
              connect()
            }, delay)
          } else {
            console.error('❌ Max WebSocket reconnection attempts reached')
            setConnectionStatus('Failed')
          }
        }
        
        ws.current.onerror = (error) => {
          console.error('🚨 WebSocket error:', error)
          setConnectionStatus('Error')
        }
      } catch (error) {
        console.error('🚨 Failed to create WebSocket connection:', error)
        setConnectionStatus('Error')
        
        // Retry connection after delay
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.pow(2, reconnectAttempts.current) * 1000
          reconnectTimeout = setTimeout(() => {
            reconnectAttempts.current++
            console.log(`🔄 WebSocket error reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`)
            connect()
          }, delay)
        }
      }
    }
    
    connect()
    
    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [endpoint])

  // Manual reconnect function
  const reconnect = () => {
    if (ws.current) {
      ws.current.close()
    }
    reconnectAttempts.current = 0
    setConnectionStatus('Connecting')
  }

  // Send message function
  const sendMessage = (message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
      return true
    }
    console.warn('WebSocket is not connected. Cannot send message:', message)
    return false
  }

  return { 
    connectionStatus, 
    lastMessage, 
    currentUrl,
    reconnect,
    sendMessage,
    isConnected: connectionStatus === 'Connected'
  }
}

export default useWebSocket