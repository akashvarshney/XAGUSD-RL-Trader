import { useEffect, useRef, useCallback } from 'react'
import { useTradingStore } from '../store/tradingStore'

const WS_URL = `ws://${window.location.hostname}:8000/ws/live`

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()

  const {
    setConnected,
    setAgentStatus,
    setTrainingStatus,
    addCandle,
    addPrediction,
    addLog,
  } = useTradingStore()

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        addLog('Connected to server')

        // Request initial status
        ws.send(JSON.stringify({ type: 'get_status' }))

        // Subscribe to channels
        ws.send(JSON.stringify({ type: 'subscribe', channel: 'trading' }))
        ws.send(JSON.stringify({ type: 'subscribe', channel: 'training' }))
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleMessage(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setConnected(false)
        addLog('Disconnected from server')

        // Reconnect after delay
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...')
          connect()
        }, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        addLog('Connection error')
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      reconnectTimeoutRef.current = setTimeout(connect, 3000)
    }
  }, [setConnected, addLog])

  const handleMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'connected':
        addLog(`Connected with ID: ${data.client_id}`)
        break

      case 'status':
        if (data.agent) {
          setAgentStatus(data.agent)
        }
        if (data.training) {
          setTrainingStatus(data.training)
        }
        break

      case 'step':
        if (data.candle) {
          addCandle(data.candle)
        }
        if (data.prediction) {
          addPrediction({
            timestamp: new Date().toISOString(),
            open: data.prediction[0],
            high: data.prediction[1],
            low: data.prediction[2],
            close: data.prediction[3],
            volume: data.prediction[4],
          })
        }
        addLog(`Step: Action=${data.action}, Reward=${data.reward?.toFixed(4)}`)
        break

      case 'training_progress':
        setTrainingStatus({
          timesteps: data.timesteps,
          episodes: data.episodes,
          best_reward: data.best_reward,
          progress: data.progress,
        })
        break

      case 'trade':
        addLog(`Trade: ${data.side} @ ${data.price}, PnL: ${data.pnl?.toFixed(2)}`)
        break

      case 'pong':
        // Heartbeat response
        break

      default:
        console.log('Unknown message type:', data.type)
    }
  }, [setAgentStatus, setTrainingStatus, addCandle, addPrediction, addLog])

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  useEffect(() => {
    connect()

    // Heartbeat
    const heartbeatInterval = setInterval(() => {
      sendMessage({ type: 'ping' })
    }, 30000)

    return () => {
      clearInterval(heartbeatInterval)
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect, sendMessage])

  return { sendMessage }
}

