/** InputBar.jsx — Auto-resizing textarea with glowing send button */

import { useState, useRef, useCallback, useEffect } from 'react'

export default function InputBar({ onSend, isLoading }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [value])

  const handleSend = useCallback(() => {
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [value, isLoading, onSend])

  const handleKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <div className="input-area">
      <div className="input-wrap">
        <textarea
          ref={textareaRef}
          className="input-field"
          placeholder={isLoading ? 'Archiva is searching…' : 'Ask anything about your documents…'}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKey}
          disabled={isLoading}
          rows={1}
          aria-label="Chat input"
        />
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={!value.trim() || isLoading}
          aria-label="Send message"
          title="Send (Enter)"
        >
          {isLoading ? '⏳' : '➤'}
        </button>
      </div>
      <p className="input-hint">
        Press <kbd style={{ background: 'var(--surface2)', padding: '1px 5px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace' }}>Enter</kbd> to send
        &nbsp;·&nbsp;
        <kbd style={{ background: 'var(--surface2)', padding: '1px 5px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace' }}>Shift+Enter</kbd> for new line
      </p>
    </div>
  )
}
