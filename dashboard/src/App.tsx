import { useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout/Layout'
import { Dashboard } from './pages/Dashboard'
import { Training } from './pages/Training'
import { History } from './pages/History'
import { useThemeStore } from './store/themeStore'
import { useWebSocket } from './hooks/useWebSocket'

function App() {
  const { theme } = useThemeStore()

  // Connect WebSocket
  useWebSocket()

  // Apply theme class to document
  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark')
    document.documentElement.classList.add(theme)
  }, [theme])

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App

