/**
 * useCiteLinking — hovering a citation lights the passage(s) it points at
 * in the Evidence panel, and hovering a passage lights the citations that
 * point at it. Only the answer whose evidence is open is linked: the panel
 * lists that answer's passages, so another answer's "1" means a different
 * passage.
 *
 * Citations are HTML inside rendered markdown (see citations.js), so this
 * listens on the document instead of attaching React handlers to each one.
 */

import { useEffect, useRef } from 'react'

const numbers = el => (el.dataset.ns || el.dataset.n || '').split(' ').filter(Boolean)

export function lightLinked(target, on, answerId) {
  if (!answerId) return
  const sel = CSS.escape(answerId)
  let cites, passages
  if (target.matches('.cite')) {
    if (target.dataset.a !== answerId) return
    cites = [target]
    passages = numbers(target).flatMap(n => [...document.querySelectorAll(`.psg[data-n="${n}"]`)])
  } else {
    const n = target.dataset.n
    passages = [target]
    cites = [...document.querySelectorAll(`.cite[data-a="${sel}"]`)].filter(c => numbers(c).includes(n))
  }
  for (const el of [...cites, ...passages]) el.classList.toggle('lit', on)
}

export function useCiteLinking(answerId) {
  const current = useRef(answerId)
  current.current = answerId

  useEffect(() => {
    const handle = on => e => {
      const t = e.target.closest?.('.cite, .psg')
      // Moving between children of the same element fires out/over pairs;
      // ignore the ones that stay inside it.
      if (!t || (e.relatedTarget && t.contains(e.relatedTarget))) return
      lightLinked(t, on, current.current)
    }
    const over = handle(true), out = handle(false)
    document.addEventListener('mouseover', over)
    document.addEventListener('mouseout', out)
    return () => {
      document.removeEventListener('mouseover', over)
      document.removeEventListener('mouseout', out)
    }
  }, [])
}
