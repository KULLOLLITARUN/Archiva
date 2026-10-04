/** InputBar.jsx — the composer: auto-resizing textarea and send button. */

import { useState, useRef, useCallback, useEffect } from 'react'
import { ArrowUp, Loader2 } from 'lucide-react'

const MAX_HEIGHT = 200   // matches .composer textarea max-height

export default function InputBar({ onSend, isLoading, placeholder = 'Ask anything about your documents…', id = 'ask' }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, MAX_HEIGHT) + 'px'
  }, [value])

  const handleSend = useCallback(() => {
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setValue('')
  }, [value, isLoading, onSend])

  const handleKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <div className="composer">
      <label htmlFor={id} className="sr">Ask a question</label>
      <textarea
        id={id}
        ref={textareaRef}
        placeholder={isLoading ? 'Archiva is searching…' : placeholder}
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKey}
        disabled={isLoading}
        rows={1}
      />
      <div className="composer-bar">
        <button
          type="button"
          className="send"
          onClick={handleSend}
          disabled={!value.trim() || isLoading}
          aria-label="Send message"
          title="Send (Enter)"
        >
          {isLoading
            ? <Loader2 size={16} strokeWidth={2} className="ico spin" aria-hidden="true" />
            : <ArrowUp size={16} strokeWidth={2} className="ico" aria-hidden="true" />}
        </button>
      </div>
    </div>
  )
}
