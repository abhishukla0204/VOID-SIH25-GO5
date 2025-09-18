import { useState, useEffect, useRef } from 'react'
import { getWsUrl, getApiUrl } from '../config/api'
import useServerSentEvents from './useServerSentEvents'

const useWebSocket = (endpoint) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  const [lastMessage, setLastMessage] = useState(null)
  const [currentUrl, setCurrentUrl] = useState(null)
  const [usingFallback, setUsingFallback] = useState(false)
  const ws = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 3 // Reduced for faster fallback
  
  // SSE fallback hook
  const sseEndpoint = endpoint.replace('/ws', '/api/events/stream')
  const sseHook = useServerSentEvents(sseEndpoint)
  
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
          setUsingFallback(false)
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
            const delay = Math.pow(2, reconnectAttempts.current) * 1000 // 1s, 2s, 4s
            console.log(`🔄 Reconnecting in ${delay}ms... (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`)
            
            reconnectTimeout = setTimeout(() => {
              reconnectAttempts.current++
              console.log(`🔄 WebSocket reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`)
              connect()
            }, delay)
          } else {
            console.warn('❌ Max WebSocket reconnection attempts reached, falling back to SSE')
            setConnectionStatus('Failed')
            setUsingFallback(true)
          }
        }
        
        ws.current.onerror = (error) => {
          console.error('🚨 WebSocket error:', error)
          setConnectionStatus('Error')
          
          // For immediate errors (like connection refused), fallback faster
          if (reconnectAttempts.current === 0) {
            console.warn('🔄 WebSocket connection failed immediately, trying SSE fallback')
            setUsingFallback(true)
            setConnectionStatus('Failed')
          }
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
        } else {
          console.warn('❌ WebSocket creation failed, falling back to SSE')
          setUsingFallback(true)
        }
      }
    }
    
    // Only try WebSocket if not already using fallback
    if (!usingFallback) {
      connect()
    }
    
    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [endpoint, usingFallback])

  // Manual reconnect function
  const reconnect = () => {
    if (usingFallback) {
      sseHook.reconnect()
    } else {
      if (ws.current) {
        ws.current.close()
      }
      reconnectAttempts.current = 0
      setConnectionStatus('Connecting')
    }
  }

  // Send message function
  const sendMessage = (message) => {
    if (usingFallback) {
      // Use SSE sendMessage method
      return sseHook.sendMessage(message)
    } else if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
      return true
    }
    console.warn('No active connection. Cannot send message:', message)
    return false
  }

  // Return appropriate values based on connection type
  if (usingFallback) {
    return {
      connectionStatus: sseHook.connectionStatus,
      lastMessage: sseHook.lastMessage,
      currentUrl: sseHook.currentUrl,
      reconnect,
      sendMessage,
      isConnected: sseHook.isConnected,
      usingFallback: true,
      connectionType: 'SSE'
    }
  }

  return { 
    connectionStatus, 
    lastMessage, 
    currentUrl,
    reconnect,
    sendMessage,
    isConnected: connectionStatus === 'Connected',
    usingFallback: false,
    connectionType: 'WebSocket'
  }
}

export default useWebSocket