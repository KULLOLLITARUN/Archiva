/** ChatWindow.jsx — Scrollable message list with welcome screen */

import { useEffect, useRef } from 'react'
import { BarChart3, GitCompare, Lightbulb, PenLine, Search } from 'lucide-react'
import MessageBubble from './MessageBubble.jsx'
import TypingIndicator from './TypingIndicator.jsx'
import SmartSuggestions from './SmartSuggestions.jsx'
import { BRAND } from '../brand.js'

const SUGGESTIONS = [
  { icon: <Search size={13} />, text: 'What are the main errors in the logs?' },
  { icon: <BarChart3 size={13} />, text: 'Summarize the key findings across all documents' },
  { icon: <GitCompare size={13} />, text: 'Compare the differences between the log files' },
  { icon: <Lightbulb size={13} />, text: 'Explain the most critical issues found' },
]

export default function ChatWindow({ messages, isLoading, streamStatus, onSuggestionClick, dynamicTopics = [], hasDocs = false }) {
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
          <div className="welcome-orb"><PenLine size={20} strokeWidth={1.75} /></div>

          {hasDocs ? (
            <div>
              <h1>What do you want to know?</h1>
              <p>Ask a question — every answer traces back to a page in the documents above.</p>
            </div>
          ) : (
            <div>
              <h1>Ask your documents anything</h1>
              <p>
                Upload a file (PDF, DOCX, TXT) to get started.
                <strong style={{ color: 'var(--accent2)' }}> {BRAND.name}</strong> reads it,
                finds the answers, and cites the exact source — so you always
                know where the information came from.
              </p>
            </div>
          )}

          {/* Quick-start chips — only shown as a cold-start fallback, before
              any document exists to generate real topics from. Once docs
              are loaded, SmartSuggestions below surfaces topics grounded in
              their actual content instead of these generic examples. */}
          {dynamicTopics.length === 0 && !hasDocs && (
            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={s.text}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => onSuggestionClick(s.text)}
                  aria-label={s.text}
                  style={{ animationDelay: `${i * 0.08}s` }}
                >
                  <span className="suggestion-icon">{s.icon}</span>
                  <span>{s.text}</span>
                </button>
              ))}
            </div>
          )}

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
