/** ChatWindow.jsx — Scrollable message list with welcome screen */

import { useEffect, useRef } from 'react'
import MessageBubble      from './MessageBubble.jsx'
import TypingIndicator    from './TypingIndicator.jsx'
import SmartSuggestions   from './SmartSuggestions.jsx'
import { BRAND }          from '../brand.js'

const SUGGESTIONS = [
  { icon: '🔍', text: 'What are the main errors in the logs?' },
  { icon: '📊', text: 'Summarize the key findings across all documents' },
  { icon: '🔀', text: 'Compare the differences between the log files' },
  { icon: '💡', text: 'Explain the most critical issues found' },
]

export default function ChatWindow({ messages, isLoading, streamStatus, onSuggestionClick, dynamicTopics = [] }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const isEmpty = messages.length === 0 && !isLoading

  return (
    <div className="chat-window" role="log" aria-live="polite">

      {/* ── Welcome screen ──────────────────────────────── */}
      {isEmpty && (
        <div className="welcome">
          <div className="welcome-orb">🧠</div>
          <div>
            <h1>Ask your documents anything</h1>
            <p>
              Upload your files (PDF, DOCX, TXT) and start asking questions.
              <strong style={{ color: 'var(--accent2)' }}> {BRAND.name}</strong> reads them,
              finds the answers, and cites the exact source — so you always
              know where the information came from.
            </p>
          </div>
          <div className="suggestions">
            {/* Quick-start chips — only show when no doc-based topics yet */}
            {dynamicTopics.length === 0 &&
              SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                className="suggestion-btn"
                onClick={() => onSuggestionClick(s.text)}
                aria-label={s.text}
                style={{ animationDelay: `${i * 0.08}s` }}
              >
                <span>
                  <span className="suggestion-icon">{s.icon}</span>
                  <span>{s.text}</span>
                </span>
              </button>
            ))}
          </div>

          {/* Divider */}
          <div style={{ width: '100%', maxWidth: 560, height: 1, background: 'var(--border)', margin: '4px 0' }} />

          {/* Smart Suggestions — fully dynamic topic explorer */}
          <SmartSuggestions
            onPromptClick={onSuggestionClick}
            disabled={isLoading}
            dynamicTopics={dynamicTopics}
          />
        </div>
      )}

      {/* ── Messages ────────────────────────────────────── */}
      {messages.map(m => (
        <MessageBubble key={m.id} message={m} />
      ))}

      {/* ── Typing indicator ─────────────────────────────  */}
      {isLoading && !messages.some(m => m.streaming && m.content) && (
        <TypingIndicator status={streamStatus} />
      )}

      <div ref={bottomRef} />
    </div>
  )
}
