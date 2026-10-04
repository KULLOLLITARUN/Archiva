/**
 * theme.js — Paper (light) / Ink (dark) theme state.
 *
 * index.html applies the theme before first paint; this hook takes over from
 * there. Until the user picks one explicitly, the theme follows the system
 * setting live (so flipping the OS to dark mode flips Archiva too). An
 * explicit pick is remembered in localStorage and wins from then on.
 */

import { useCallback, useEffect, useState } from 'react'

export const THEME_KEY = 'archiva-theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'

function savedTheme() {
  try {
    const t = localStorage.getItem(THEME_KEY)
    return t === 'paper' || t === 'ink' ? t : null
  } catch {
    return null   // storage blocked (private mode, sandboxed iframe)
  }
}

function systemTheme() {
  return typeof matchMedia === 'function' && matchMedia(DARK_QUERY).matches ? 'ink' : 'paper'
}

export function initialTheme() {
  return savedTheme() || systemTheme()
}

export function useTheme() {
  const [theme, setThemeState] = useState(initialTheme)

  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

  useEffect(() => {
    if (typeof matchMedia !== 'function') return undefined
    const mq = matchMedia(DARK_QUERY)
    const onChange = () => { if (!savedTheme()) setThemeState(systemTheme()) }
    mq.addEventListener?.('change', onChange)
    return () => mq.removeEventListener?.('change', onChange)
  }, [])

  const setTheme = useCallback((t) => {
    try { localStorage.setItem(THEME_KEY, t) } catch { /* not remembered, still applied */ }
    setThemeState(t)
  }, [])

  return [theme, setTheme]
}
