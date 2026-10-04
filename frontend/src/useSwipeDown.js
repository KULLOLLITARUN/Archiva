/**
 * useSwipeDown — drag a phone bottom sheet down by its header to dismiss
 * it (the Evidence sheet and the dialogs). Only active under NARROW, where
 * those panels are sheets; buttons in the header stay plain buttons.
 */

import { useRef } from 'react'
import { matches, NARROW } from './useMedia.js'

// Dragging the sheet down past this far (or a quarter of its height,
// whichever is less) dismisses it; anything shorter snaps back.
const DISMISS_PX = 140

export function useSwipeDown(sheetRef, onDismiss) {
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
