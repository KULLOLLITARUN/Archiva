import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Suppress "ECONNREFUSED" proxy errors while backend is booting up
function silentProxy(target) {
  return {
    target,
    changeOrigin: true,
    onError(err, req, res) {
      if (err.code === 'ECONNREFUSED') return  // backend still starting — ignore
      res.writeHead(502).end('Backend unavailable')
    },
  }
}

const BACKEND = silentProxy('http://127.0.0.1:8000')

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/register':    BACKEND,
      '/login':       BACKEND,
      '/me':          BACKEND,
      '/chat':        BACKEND,
      '/upload':      BACKEND,
      '/files':       BACKEND,
      '/documents':   BACKEND,
      '/docs-loaded': BACKEND,
      '/reload':      BACKEND,
      '/suggestions': BACKEND,
      '/admin':       BACKEND,
      '/stats':       BACKEND,
      '/health':      BACKEND,
    }
  }
})
