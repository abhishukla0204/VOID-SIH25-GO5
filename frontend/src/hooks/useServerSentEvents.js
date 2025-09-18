import { useState, useEffect, useRef } from 'react'
import { getApiUrl } from '../config/api'

const useServerSentEvents = (endpoint) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  const [lastMessage, setLastMessage] = useState(null)
  const [currentUrl, setCurrentUrl] = useState(null)
  const eventSource = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  useEffect(() => {
    let reconnectTimeout

    const connect = () => {
      try {
        // Get the SSE URL from configuration
        const url = getApiUrl(endpoint)
        setCurrentUrl(url)

        console.log(`📡 Attempting SSE connection to: ${url}`)
        eventSource.current = new EventSource(url)

        eventSource.current.onopen = () => {
          console.log(`✅ SSE connected to: ${url}`)
          setConnectionStatus('Connected')
          reconnectAttempts.current = 0 // Reset on successful connection
        }

        eventSource.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
          } catch (error) {
            console.warn('Failed to parse SSE message:', event.data)
            setLastMessage({ raw: event.data })
          }
        }

        eventSource.current.onerror = (error) => {
          console.error('🚨 SSE error:', error)
          setConnectionStatus('Error')
          
          // Close the connection and attempt to reconnect
          eventSource.current.close()
          
          if (reconnectAttempts.current < maxReconnectAttempts) {
            const delay = Math.pow(2, reconnectAttempts.current) * 1000 // 1s, 2s, 4s, 8s, 16s
            console.log(`🔄 SSE Reconnecting in ${delay}ms... (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`)
            
            reconnectTimeout = setTimeout(() => {
              reconnectAttempts.current++
              console.log(`🔄 SSE reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`)
              connect()
            }, delay)
          } else {
            console.error('❌ Max SSE reconnection attempts reached')
            setConnectionStatus('Failed')
          }
        }
      } catch (error) {
        console.error('🚨 Failed to create SSE connection:', error)
        setConnectionStatus('Error')
        
        // Retry connection after delay
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.pow(2, reconnectAttempts.current) * 1000
          reconnectTimeout = setTimeout(() => {
            reconnectAttempts.current++
            console.log(`🔄 SSE error reconnection attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`)
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
      if (eventSource.current) {
        eventSource.current.close()
      }
    }
  }, [endpoint])

  // Manual reconnect function
  const reconnect = () => {
    if (eventSource.current) {
      eventSource.current.close()
    }
    reconnectAttempts.current = 0
    setConnectionStatus('Connecting')
  }

  // Send data function (SSE is one-way, so this would need HTTP POST)
  const sendMessage = async (message) => {
    try {
      const response = await fetch(getApiUrl('/api/events/message'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(message),
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Failed to send message via SSE:', error)
      throw error
    }
  }

  return {
    connectionStatus,
    lastMessage,
    currentUrl,
    reconnect,
    sendMessage,
    isConnected: connectionStatus === 'Connected',
    isConnecting: connectionStatus === 'Connecting',
    isFailed: connectionStatus === 'Failed'
  }
}

export default useServerSentEvents