import { useState, useEffect, useRef } from 'react'

const useWebSocket = (url) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting')
  const [lastMessage, setLastMessage] = useState(null)
  const ws = useRef(null)
  
  useEffect(() => {
    const connect = () => {
      try {
        ws.current = new WebSocket(url)
        
        ws.current.onopen = () => {
          console.log('WebSocket connected')
          setConnectionStatus('Connected')
        }
        
        ws.current.onmessage = (event) => {
          setLastMessage(event.data)
        }
        
        ws.current.onclose = () => {
          console.log('WebSocket disconnected')
          setConnectionStatus('Disconnected')
          
          // Attempt to reconnect after 3 seconds
          setTimeout(connect, 3000)
        }
        
        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error)
          setConnectionStatus('Error')
        }
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error)
        setConnectionStatus('Error')
        
        // Attempt to reconnect after 5 seconds
        setTimeout(connect, 5000)
      }
    }
    
    connect()
    
    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [url])
  
  const sendMessage = (message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(message)
    } else {
      console.warn('WebSocket is not connected')
    }
  }
  
  return {
    connectionStatus,
    lastMessage,
    sendMessage
  }
}

export default useWebSocket