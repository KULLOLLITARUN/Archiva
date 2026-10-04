/** ChatWindow.jsx — the scrolling centre: Home when empty, else the thread,
 *  with "Ask next" questions under the latest finished answer. */

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { stagger } from 'animejs'
import { ArrowRight } from 'lucide-react'
import MessageBubble from './MessageBubble.jsx'
import Home, { topicIcon } from './Home.jsx'
import { go } from '../motion.js'
import { nextQuestions } from '../suggest.js'
import { isFailedAnswer } from '../evidence.js'

const ICON = { size: 15, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

function AskNext({ items, onSend, disabled }) {
  return (
    <div className="next" role="group" aria-label="Ask next">
      <span className="next-h">Ask next</span>
      {items.map(({ label, prompt }) => {
        const Icon = topicIcon(label)
        return (
          <button key={prompt} type="button" className="next-q" onClick={() => onSend(prompt)} disabled={disabled}>
            <Icon {...ICON} /><span>{prompt}</span><ArrowRight {...ICON} size={14} />
          </button>
        )
      })}
    </div>
  )
}

export default function ChatWindow({
  messages, streamStatus, onSend, inputDisabled, topics = [], topicsLoaded = false, docCount = 0,
  selectedId = null, onSelect, onCite, onAddDocs,
}) {
  const scrollRef = useRef(null)
  const threadRef = useRef(null)
  const seen = useRef(new Set())
  const lastContent = messages.at(-1)?.content
  const last = messages.at(-1)
  // Not after a failed request: if the provider is down, the next question would fail too.
  const showNext = last?.role === 'bot' && !last.streaming && !isFailedAnswer(last) && !inputDisabled
  const next = useMemo(() => (showNext ? nextQuestions(topics, messages) : []), [showNext, topics, messages])

  // Follow the conversation as it grows.
  useEffect(() => {
    const el = scrollRef.current
    if (el && messages.length) el.scrollTop = el.scrollHeight
  }, [messages.length, lastContent])

  // New turns rise in; a finished answer's confidence bar fills once.
  useLayoutEffect(() => {
    const thread = threadRef.current
    if (!thread) { seen.current.clear(); return }
    const fresh = [...thread.children].filter(el => !seen.current.has(el.dataset.key))
    fresh.forEach(el => seen.current.add(el.dataset.key))
    if (fresh.length) go(fresh, { opacity: [0, 1], translateY: [14, 0], delay: stagger(110), duration: 650, ease: 'outExpo' })
    const bars = [...thread.querySelectorAll('.ans-foot .bar i:not([data-filled])')]
    bars.forEach(b => b.setAttribute('data-filled', ''))
    if (bars.length) go(bars, { width: ['0%', el => el.style.getPropertyValue('--v')], duration: 1100, ease: 'outExpo', delay: 150 })
  })

  return (
    <div className="scroll" ref={scrollRef}>
      {messages.length === 0 ? (
        <Home topics={topics} topicsLoaded={topicsLoaded} docCount={docCount}
          onSend={onSend} inputDisabled={inputDisabled} onAddDocs={onAddDocs} />
      ) : (
        <div className="thread" ref={threadRef} role="log" aria-live="polite" aria-relevant="additions">
          {messages.map(m => (
            <div key={m.id} data-key={m.id} className={m.role === 'user' ? 'turn turn--q' : 'turn'}>
              <MessageBubble message={m} selected={m.id === selectedId} onSelect={onSelect} onCite={onCite}
                docCount={docCount} status={streamStatus} />
            </div>
          ))}
          {next.length > 0 && (
            <div data-key={`next-${last.id}`} className="turn turn--next">
              <AskNext items={next} onSend={onSend} disabled={inputDisabled} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
