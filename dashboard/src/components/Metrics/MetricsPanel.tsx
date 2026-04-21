import { useEffect } from 'react'
import { TrendingUp, TrendingDown, Target, Activity } from 'lucide-react'
import { useTradingStore } from '../../store/tradingStore'
import { dataApi } from '../../api/client'
import { cn, formatCurrency, formatPercent } from '../../lib/utils'

export function MetricsPanel() {
  const { metrics, setMetrics } = useTradingStore()

  useEffect(() => {
    const fetchMetrics = async () => {
      const { data } = await dataApi.getMetrics()
      if (data) {
        setMetrics(data)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000)
    return () => clearInterval(interval)
  }, [setMetrics])

  return (
    <div
      className="card"
      style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
    >
      <h3
        className="text-lg font-semibold mb-4"
        style={{ color: 'var(--text-primary)' }}
      >
        Performance Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4">
        {/* Total PnL */}
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-2 mb-1">
            {metrics.total_pnl >= 0 ? (
              <TrendingUp size={16} className="text-profit" />
            ) : (
              <TrendingDown size={16} className="text-loss" />
            )}
            <span
              className="text-xs uppercase"
              style={{ color: 'var(--text-muted)' }}
            >
              Total PnL
            </span>
          </div>
          <p
            className={cn(
              "text-xl font-bold",
              metrics.total_pnl >= 0 ? "number-positive" : "number-negative"
            )}
          >
            {formatCurrency(metrics.total_pnl)}
          </p>
        </div>

        {/* Win Rate */}
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-2 mb-1">
            <Target size={16} style={{ color: 'var(--accent-blue)' }} />
            <span
              className="text-xs uppercase"
              style={{ color: 'var(--text-muted)' }}
            >
              Win Rate
            </span>
          </div>
          <p
            className="text-xl font-bold"
            style={{ color: 'var(--text-primary)' }}
          >
            {formatPercent(metrics.win_rate)}
          </p>
        </div>

        {/* Total Trades */}
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-2 mb-1">
            <Activity size={16} style={{ color: 'var(--accent-purple)' }} />
            <span
              className="text-xs uppercase"
              style={{ color: 'var(--text-muted)' }}
            >
              Total Trades
            </span>
          </div>
          <p
            className="text-xl font-bold"
            style={{ color: 'var(--text-primary)' }}
          >
            {metrics.total_trades}
          </p>
        </div>

        {/* Profit Factor */}
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-xs uppercase"
              style={{ color: 'var(--text-muted)' }}
            >
              Profit Factor
            </span>
          </div>
          <p
            className={cn(
              "text-xl font-bold",
              metrics.profit_factor >= 1 ? "number-positive" : "number-negative"
            )}
          >
            {metrics.profit_factor.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Win/Loss breakdown */}
      <div
        className="mt-4 pt-4 border-t"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <div className="flex justify-between text-sm">
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Wins: </span>
            <span className="number-positive font-medium">{metrics.winning_trades}</span>
            <span style={{ color: 'var(--text-muted)' }}> (Avg: </span>
            <span className="number-positive">{formatCurrency(metrics.avg_win)}</span>
            <span style={{ color: 'var(--text-muted)' }}>)</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Losses: </span>
            <span className="number-negative font-medium">{metrics.losing_trades}</span>
            <span style={{ color: 'var(--text-muted)' }}> (Avg: </span>
            <span className="number-negative">{formatCurrency(metrics.avg_loss)}</span>
            <span style={{ color: 'var(--text-muted)' }}>)</span>
          </div>
        </div>
      </div>
    </div>
  )
}

