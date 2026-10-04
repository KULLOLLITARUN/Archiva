/** ChatWindow.jsx — the scrolling centre: Home when empty, else the thread. */

import { useEffect, useLayoutEffect, useRef } from 'react'
import { stagger } from 'animejs'
import MessageBubble from './MessageBubble.jsx'
import Home from './Home.jsx'
import { go } from '../motion.js'

export default function ChatWindow({
  messages, streamStatus, onSend, inputDisabled, topics = [], topicsLoaded = false, docCount = 0,
  selectedId = null, onSelect, onCite, onAddDocs,
}) {
  const scrollRef = useRef(null)
  const threadRef = useRef(null)
  const seen = useRef(new Set())
  const lastContent = messages.at(-1)?.content

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
        </div>
      )}
    </div>
  )
}
