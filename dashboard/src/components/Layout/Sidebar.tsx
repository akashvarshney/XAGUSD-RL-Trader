import { NavLink } from 'react-router-dom'
import { LayoutDashboard, GraduationCap, History } from 'lucide-react'
import { cn } from '../../lib/utils'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/training', icon: GraduationCap, label: 'Training' },
  { to: '/history', icon: History, label: 'History' },
]

export function Sidebar() {
  return (
    <aside
      className="w-56 border-r flex flex-col"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-color)'
      }}
    >
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map(({ to, icon: Icon, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md transition-colors",
                    isActive
                      ? "bg-accent-blue text-white"
                      : "hover:bg-opacity-10 hover:bg-white"
                  )
                }
                style={({ isActive }) => ({
                  color: isActive ? 'white' : 'var(--text-secondary)',
                })}
              >
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer */}
      <div
        className="p-4 border-t text-xs"
        style={{
          borderColor: 'var(--border-color)',
          color: 'var(--text-muted)',
        }}
      >
        <p>XAGUSD RL Trader v0.1.0</p>
        <p>Deep Reinforcement Learning</p>
      </div>
    </aside>
  )
}

