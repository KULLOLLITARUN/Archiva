/**
 * EvidencePanel.jsx — the right-hand column that shows how an answer was
 * verified. A column from 1280px up, a right drawer below that, and a
 * bottom sheet on phones (drag the header down to dismiss).
 *
 * Until an answer is selected it explains what the checks are.
 */

import { useRef } from 'react'
import { Check, Hash, Scale, ShieldCheck, X } from 'lucide-react'
import { matches, NARROW } from '../useMedia.js'

const ICON = { size: 16, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

function EmptyEvidence() {
  return (
    <div className="ev-empty">
      <div className="big"><ShieldCheck {...ICON} size={20} /></div>
      <h3>Evidence appears here</h3>
      <p>Select an answer to see how it was verified and the exact passages it came from.</p>
      <div className="how">
        <div><Hash {...ICON} /><span>Every number in an answer must appear in the source.</span></div>
        <div><Scale {...ICON} /><span>Answers that contradict the source are rejected and retried.</span></div>
        <div><Check {...ICON} /><span>No supporting passage means no answer &mdash; never a guess.</span></div>
      </div>
    </div>
  )
}

// Dragging the sheet down past this far (or a quarter of its height,
// whichever is less) dismisses it; anything shorter snaps back.
const DISMISS_PX = 140

function useSwipeDown(sheetRef, onDismiss) {
  const drag = useRef(null)

  const onPointerDown = e => {
    if (!matches(NARROW) || e.target.closest('button')) return
    drag.current = { y0: e.clientY, dy: 0 }
    sheetRef.current.style.transition = 'none'
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onPointerMove = e => {
    if (!drag.current) return
    drag.current.dy = Math.max(0, e.clientY - drag.current.y0)
    sheetRef.current.style.transform = `translateY(${drag.current.dy}px)`
  }
  const end = () => {
    if (!drag.current) return
    const { dy } = drag.current
    drag.current = null
    const sheet = sheetRef.current
    sheet.style.transition = ''
    sheet.style.transform = ''
    if (dy > Math.min(DISMISS_PX, sheet.offsetHeight * 0.25)) onDismiss()
  }
  return { onPointerDown, onPointerMove, onPointerUp: end, onPointerCancel: end }
}

export default function EvidencePanel({ open, inert, onClose, label, children }) {
  const ref = useRef(null)
  const swipe = useSwipeDown(ref, onClose)

  return (
    <aside
      ref={ref}
      id="evidence"
      className={`ev${open ? ' open' : ''}`}
      aria-label="Evidence"
      inert={inert ? '' : undefined}
    >
      <div className="ev-in">
        <div className="ev-top" {...swipe}>
          <h2>Evidence</h2>
          {label && <span className="ev-for">{label}</span>}
          <button type="button" className="ibtn" onClick={onClose} aria-label="Close evidence">
            <X {...ICON} />
          </button>
        </div>
        <div className="ev-body">{children || <EmptyEvidence />}</div>
      </div>
    </aside>
  )
}
