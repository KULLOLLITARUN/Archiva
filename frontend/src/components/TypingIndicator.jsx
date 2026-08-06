/** TypingIndicator.jsx — Animated bouncing dots with stream status */

export default function TypingIndicator({ status }) {
  return (
    <div className="typing-row">
      <div className="avatar avatar--bot">◈</div>
      <div className="typing-bubble">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
        {status && <span className="stream-status">&nbsp;&nbsp;{status}</span>}
      </div>
    </div>
  )
}
