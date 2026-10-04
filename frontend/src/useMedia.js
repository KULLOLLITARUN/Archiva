import { useEffect, useState } from 'react'

// Breakpoints shared with styles.css — keep the two in sync.
export const WIDE = '(min-width: 1280px)'     // evidence is a column
export const NARROW = '(max-width: 899px)'    // library is a menu, evidence a bottom sheet

export function matches(query) {
  return typeof matchMedia === 'function' && matchMedia(query).matches
}

export function useMedia(query) {
  const [on, setOn] = useState(() => matches(query))
  useEffect(() => {
    if (typeof matchMedia !== 'function') return undefined
    const mq = matchMedia(query)
    const onChange = () => setOn(mq.matches)
    onChange()
    mq.addEventListener?.('change', onChange)
    return () => mq.removeEventListener?.('change', onChange)
  }, [query])
  return on
}
