import React from 'react'

const ConnectionStatus = ({ connectionStatus, connectionType, usingFallback, onReconnect }) => {
  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'Connected':
        return usingFallback ? 'bg-yellow-100 text-yellow-800 border-yellow-200' : 'bg-green-100 text-green-800 border-green-200'
      case 'Connecting':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'Error':
      case 'Failed':
        return 'bg-red-100 text-red-800 border-red-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'Connected':
        return usingFallback ? '📡' : '🔌'
      case 'Connecting':
        return '🔄'
      case 'Error':
      case 'Failed':
        return '❌'
      default:
        return '⚪'
    }
  }

  const getStatusMessage = () => {
    if (connectionStatus === 'Connected') {
      if (usingFallback) {
        return `Connected via ${connectionType} (WebSocket fallback)`
      }
      return `Connected via ${connectionType}`
    }
    return connectionStatus
  }

  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor()}`}>
      <span className="mr-2">{getStatusIcon()}</span>
      <span>{getStatusMessage()}</span>
      {(connectionStatus === 'Error' || connectionStatus === 'Failed') && (
        <button
          onClick={onReconnect}
          className="ml-2 text-xs underline hover:no-underline"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export default ConnectionStatus