/**
 * Modal.jsx — the Reading Room dialog shared by the Playbook, Pipeline stats
 * and library maintenance: a centred panel on desktop, a bottom sheet on
 * phones (drag the header down to dismiss, like the Evidence sheet).
 *
 * Mount it only while open. It takes focus, keeps Tab inside, closes on
 * Escape or a click on the scrim, and hands focus back on close.
 */

import { useEffect, useId, useLayoutEffect, useRef } from 'react'
import { X } from 'lucide-react'
import { go } from '../motion.js'
import { useSwipeDown } from '../useSwipeDown.js'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]:not([tabindex="-1"])'

export default function Modal({ title, sub, icon: Icon, wide = false, onClose, children, className = '' }) {
  const ref = useRef(null)
  const scrimRef = useRef(null)
  const titleId = useId()
  const swipe = useSwipeDown(ref, onClose)

  // Focus the panel itself (not the close button, which would show a focus
  // ring nobody asked for), and return focus to the opener afterwards.
  useEffect(() => {
    const opener = document.activeElement
    ref.current.focus({ preventScroll: true })
    return () => opener?.focus?.({ preventScroll: true })
  }, [])

  useLayoutEffect(() => {
    go(scrimRef.current, { opacity: [0, 1], duration: 250, ease: 'outQuad' })
    go(ref.current, { opacity: [0, 1], translateY: [16, 0], duration: 500, ease: 'outExpo' })
  }, [])

  // Keys stop here: Escape closes this dialog only, and App's shortcuts
  // ("N" for a new conversation) must not act on the page behind it.
  const onKeyDown = e => {
    e.stopPropagation()
    if (e.key === 'Escape') { onClose(); return }
    if (e.key !== 'Tab') return
    const items = [...ref.current.querySelectorAll(FOCUSABLE)].filter(el => el.offsetParent !== null)
    if (!items.length) { e.preventDefault(); return }
    const first = items[0], last = items[items.length - 1]
    if (e.shiftKey && (document.activeElement === first || document.activeElement === ref.current)) {
      e.preventDefault(); last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus()
    }
  }

  return (
    <div ref={scrimRef} className="dlg-scrim" onPointerDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={ref} className={`dlg${wide ? ' dlg--wide' : ''} ${className}`} role="dialog" aria-modal="true"
        aria-labelledby={titleId} tabIndex={-1} onKeyDown={onKeyDown}>
        <div className="dlg-top" {...swipe}>
          {Icon && <span className="dlg-ico"><Icon size={17} strokeWidth={1.75} className="ico" aria-hidden="true" /></span>}
          <div className="dlg-t">
            <h2 id={titleId}>{title}</h2>
            {sub && <p>{sub}</p>}
          </div>
          <button type="button" className="ibtn" onClick={onClose} aria-label={`Close ${title}`}>
            <X size={16} strokeWidth={1.75} className="ico" aria-hidden="true" />
          </button>
        </div>
        <div className="dlg-body">{children}</div>
      </div>
    </div>
  )
}
