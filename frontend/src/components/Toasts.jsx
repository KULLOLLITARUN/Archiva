/** Toasts.jsx — short, self-dismissing status messages (upload results etc.). */

import { useCallback, useRef, useState } from 'react'
import { Check, Info, X } from 'lucide-react'

const ICON = { ok: Check, info: Info, err: X }
const LIFETIME_MS = 3500

export function useToasts() {
  const [toasts, setToasts] = useState([])
  const counter = useRef(0)
  const toast = useCallback((msg, type = 'ok') => {
    const id = ++counter.current
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), LIFETIME_MS)
  }, [])
  return [toasts, toast]
}

export function ToastStack({ toasts }) {
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map(t => {
        const Icon = ICON[t.type] || Info
        return (
          <div key={t.id} className={`toast toast--${t.type}`}>
            <span className="toast-icon"><Icon size={14} aria-hidden="true" /></span>
            <span>{t.msg}</span>
          </div>
        )
      })}
    </div>
  )
}
