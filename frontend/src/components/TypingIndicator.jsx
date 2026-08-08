/** TypingIndicator.jsx — Animated bouncing dots with stream status */

import { Sparkle } from 'lucide-react'

export default function TypingIndicator({ status }) {
  return (
    <div className="typing-row">
      <div className="avatar avatar--bot"><Sparkle size={16} /></div>
      <div className="typing-bubble">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
        {status && <span className="stream-status">&nbsp;&nbsp;{status}</span>}
      </div>
    </div>
  )
}
