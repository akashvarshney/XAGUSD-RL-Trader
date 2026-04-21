import { useState, useEffect } from 'react'
import { Play, Square, Pause, FileText } from 'lucide-react'
import { useTradingStore } from '../store/tradingStore'
import { trainingApi, dataApi } from '../api/client'
import { cn } from '../lib/utils'

interface CsvFile {
  name: string
  path: string
  size_mb: number
  modified: string
}

export function Training() {
  const { trainingStatus, setTrainingStatus, addLog } = useTradingStore()
  const [loading, setLoading] = useState(false)
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
  const [selectedFile, setSelectedFile] = useState('')
  const [timesteps, setTimesteps] = useState(1000000)

  // Fetch CSV files
  useEffect(() => {
    const fetchFiles = async () => {
      const { data } = await dataApi.getCsvFiles()
      if (data?.files) {
        setCsvFiles(data.files)
        if (data.files.length > 0) {
          setSelectedFile(data.files[0].path)
        }
      }
    }
    fetchFiles()
  }, [])

  // Poll training progress
  useEffect(() => {
    if (trainingStatus.status !== 'pretraining') return

    const interval = setInterval(async () => {
      const { data } = await trainingApi.getProgress()
      if (data) {
        setTrainingStatus(data)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [trainingStatus.status, setTrainingStatus])

  const handleStart = async () => {
    if (!selectedFile) {
      addLog('Error: No CSV file selected')
      return
    }

    setLoading(true)
    const { data, error } = await trainingApi.startPretrain(selectedFile, timesteps)
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setTrainingStatus({ status: 'pretraining' })
      addLog(data?.message || 'Training started')
    }
    setLoading(false)
  }

  const handleStop = async () => {
    setLoading(true)
    const { data, error } = await trainingApi.stop()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setTrainingStatus({ status: 'idle' })
      addLog(data?.message || 'Training stopped')
    }
    setLoading(false)
  }

  const handlePause = async () => {
    setLoading(true)
    const { data, error } = await trainingApi.pause()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setTrainingStatus({ status: 'paused' })
      addLog(data?.message || 'Training paused')
    }
    setLoading(false)
  }

  const handleResume = async () => {
    setLoading(true)
    const { data, error } = await trainingApi.resume()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setTrainingStatus({ status: 'pretraining' })
      addLog(data?.message || 'Training resumed')
    }
    setLoading(false)
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <h2
        className="text-2xl font-bold font-display"
        style={{ color: 'var(--text-primary)' }}
      >
        Training
      </h2>

      <div className="grid grid-cols-2 gap-6">
        {/* Training Config */}
        <div
          className="card"
          style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
        >
          <h3
            className="text-lg font-semibold mb-4"
            style={{ color: 'var(--text-primary)' }}
          >
            Pre-Training Configuration
          </h3>

          {/* CSV File Selection */}
          <div className="mb-4">
            <label className="label">CSV Data File</label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="input"
              disabled={trainingStatus.status !== 'idle'}
            >
              {csvFiles.length === 0 && (
                <option value="">No CSV files found</option>
              )}
              {csvFiles.map((file) => (
                <option key={file.path} value={file.path}>
                  {file.name} ({file.size_mb.toFixed(2)} MB)
                </option>
              ))}
            </select>
          </div>

          {/* Timesteps */}
          <div className="mb-4">
            <label className="label">Total Timesteps</label>
            <input
              type="number"
              value={timesteps}
              onChange={(e) => setTimesteps(Number(e.target.value))}
              className="input"
              min={10000}
              step={100000}
              disabled={trainingStatus.status !== 'idle'}
            />
          </div>

          {/* Controls */}
          <div className="flex gap-2">
            {trainingStatus.status === 'idle' && (
              <button
                onClick={handleStart}
                disabled={loading || !selectedFile}
                className="btn-success flex items-center gap-2"
              >
                <Play size={16} />
                Start Training
              </button>
            )}

            {trainingStatus.status === 'pretraining' && (
              <>
                <button
                  onClick={handlePause}
                  disabled={loading}
                  className="btn-secondary flex items-center gap-2"
                >
                  <Pause size={16} />
                  Pause
                </button>
                <button
                  onClick={handleStop}
                  disabled={loading}
                  className="btn-danger flex items-center gap-2"
                >
                  <Square size={16} />
                  Stop
                </button>
              </>
            )}

            {trainingStatus.status === 'paused' && (
              <>
                <button
                  onClick={handleResume}
                  disabled={loading}
                  className="btn-success flex items-center gap-2"
                >
                  <Play size={16} />
                  Resume
                </button>
                <button
                  onClick={handleStop}
                  disabled={loading}
                  className="btn-danger flex items-center gap-2"
                >
                  <Square size={16} />
                  Stop
                </button>
              </>
            )}
          </div>
        </div>

        {/* Training Progress */}
        <div
          className="card"
          style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
        >
          <h3
            className="text-lg font-semibold mb-4"
            style={{ color: 'var(--text-primary)' }}
          >
            Training Progress
          </h3>

          {/* Status */}
          <div className="flex items-center gap-2 mb-4">
            <span
              className={cn(
                "status-dot",
                trainingStatus.status === 'pretraining' && "running",
                trainingStatus.status === 'idle' && "stopped",
                trainingStatus.status === 'paused' && "paused",
                trainingStatus.status === 'error' && "error",
              )}
            />
            <span
              className="capitalize font-medium"
              style={{ color: 'var(--text-primary)' }}
            >
              {trainingStatus.status === 'pretraining' ? 'Training' : trainingStatus.status}
            </span>
          </div>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span style={{ color: 'var(--text-secondary)' }}>Progress</span>
              <span style={{ color: 'var(--text-primary)' }}>
                {(trainingStatus.progress * 100).toFixed(1)}%
              </span>
            </div>
            <div
              className="h-2 rounded-full overflow-hidden"
              style={{ backgroundColor: 'var(--bg-secondary)' }}
            >
              <div
                className="h-full bg-accent-blue transition-all duration-300"
                style={{ width: `${trainingStatus.progress * 100}%` }}
              />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Timesteps
              </p>
              <p
                className="text-lg font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {trainingStatus.timesteps.toLocaleString()}
              </p>
            </div>
            <div>
              <p
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Episodes
              </p>
              <p
                className="text-lg font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {trainingStatus.episodes}
              </p>
            </div>
            <div className="col-span-2">
              <p
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Best Reward
              </p>
              <p
                className={cn(
                  "text-lg font-semibold",
                  trainingStatus.best_reward > 0 ? "number-positive" : "number-neutral"
                )}
              >
                {trainingStatus.best_reward.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Error message */}
          {trainingStatus.error_message && (
            <div
              className="mt-4 p-2 rounded text-sm"
              style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--loss)' }}
            >
              {trainingStatus.error_message}
            </div>
          )}
        </div>
      </div>

      {/* CSV Files Info */}
      <div
        className="card"
        style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
      >
        <div className="flex items-center gap-2 mb-4">
          <FileText size={18} style={{ color: 'var(--text-secondary)' }} />
          <h3
            className="text-lg font-semibold"
            style={{ color: 'var(--text-primary)' }}
          >
            Available Data Files
          </h3>
        </div>

        {csvFiles.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>
            No CSV files found. Place your XAGUSD_1M.csv file in the data/historical/ directory.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <th className="text-left py-2" style={{ color: 'var(--text-secondary)' }}>File</th>
                  <th className="text-right py-2" style={{ color: 'var(--text-secondary)' }}>Size</th>
                  <th className="text-right py-2" style={{ color: 'var(--text-secondary)' }}>Modified</th>
                </tr>
              </thead>
              <tbody>
                {csvFiles.map((file) => (
                  <tr
                    key={file.path}
                    className="hover:bg-opacity-5 hover:bg-white"
                    style={{ borderBottom: '1px solid var(--border-color)' }}
                  >
                    <td className="py-2" style={{ color: 'var(--text-primary)' }}>{file.name}</td>
                    <td className="text-right py-2" style={{ color: 'var(--text-secondary)' }}>
                      {file.size_mb.toFixed(2)} MB
                    </td>
                    <td className="text-right py-2" style={{ color: 'var(--text-secondary)' }}>
                      {new Date(file.modified).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

