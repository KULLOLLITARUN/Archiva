/**
 * Evidence.jsx — the Evidence panel's content for one answer: confidence
 * ring and verdict, the checks it passed, and the passages it cites.
 * What each line may claim is decided in evidence.js.
 */

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { AlertTriangle, Check, Cpu, FileSearch } from 'lucide-react'
import { spring, stagger } from 'animejs'
import { buildEvidence, highlight } from '../evidence.js'
import { go, reducedMotion } from '../motion.js'

const ICON = { size: 14, strokeWidth: 2, className: 'ico ico-sm', 'aria-hidden': true }
const RING = 2 * Math.PI * 24   // circumference of the r=24 ring

const CHECK_ICON = { ok: Check, warn: AlertTriangle, info: Cpu }
const TYPE_LABEL = { pdf: 'PDF', doc: 'DOC', ocr: 'OCR', txt: 'TXT' }

function Ring({ value }) {
  return (
    <div className={`ring${value < 0.6 ? ' ring--low' : ''}`} role="img" aria-label={`${Math.round(value * 100)}% confidence`}>
      <svg viewBox="0 0 58 58" aria-hidden="true">
        <circle className="trk" cx="29" cy="29" r="24" />
        <circle className="val" cx="29" cy="29" r="24"
          style={{ strokeDasharray: RING, strokeDashoffset: RING * (1 - value) }} data-to={RING * (1 - value)} />
      </svg>
      <b>{Math.round(value * 100)}%</b>
    </div>
  )
}

function Passage({ p, answer }) {
  const parts = useMemo(() => highlight(p.text, answer), [p.text, answer])
  return (
    <div className="psg" data-n={p.n}>
      <div className="psg-h">
        <span className="psg-num">{p.n}</span>
        <span className={`ftype ft-${p.type}`}>{TYPE_LABEL[p.type]}</span>
        <div className="psg-n">
          <b title={p.filename}>{p.filename}</b>
          {p.page != null && <span>page {p.page}</span>}
        </div>
      </div>
      <blockquote>
        {parts.map((s, i) => (s.mark ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>))}
        {p.truncated && '…'}
      </blockquote>
      <div className="psg-f">
        <span className="rel" aria-hidden="true">
          {[1, 2, 3, 4, 5].map(k => <i key={k} className={k <= p.rel ? 'on' : ''} />)}
        </span>
        {p.relLabel}
      </div>
    </div>
  )
}

const LIT_MS = 1200

export default function Evidence({ message, focus }) {
  const ev = useMemo(() => buildEvidence(message), [message])
  const ref = useRef(null)

  // Entrance per answer: the ring fills from empty, then the rows rise in.
  useLayoutEffect(() => {
    const root = ref.current
    const ring = root.querySelector('.ring .val')
    if (ring) go(ring, { strokeDashoffset: [RING, Number(ring.dataset.to)], duration: 1100, ease: 'outExpo', delay: 120 })
    go(root.querySelectorAll('.verdict, .checks li, .ev-sec, .psg'),
      { opacity: [0, 1], translateY: [8, 0], delay: stagger(45), duration: 500, ease: 'outExpo' })
  }, [message.id])

  // A clicked citation: bring its passage into view and flash it.
  useEffect(() => {
    if (!focus?.ns?.length) return undefined
    const passages = focus.ns.map(n => ref.current.querySelector(`.psg[data-n="${n}"]`)).filter(Boolean)
    if (!passages.length) return undefined
    passages[0].scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'nearest' })
    go(passages[0], { scale: [0.98, 1], ease: spring({ bounce: 0.4, duration: 500 }) })
    passages.forEach(p => p.classList.add('lit'))
    const t = setTimeout(() => passages.forEach(p => p.classList.remove('lit')), LIT_MS)
    return () => { clearTimeout(t); passages.forEach(p => p.classList.remove('lit')) }
  }, [focus])

  return (
    <div ref={ref} className={`evidence ev--${ev.kind}`}>
      <div className="verdict">
        {ev.confidence != null
          ? <Ring value={Math.max(0, Math.min(1, ev.confidence))} />
          : <div className="ring ring--none" aria-hidden="true"><FileSearch size={20} strokeWidth={1.75} className="ico" /></div>}
        <div>
          <h3>{ev.title}</h3>
          <p>{ev.note}</p>
        </div>
      </div>

      {ev.checks.length > 0 && (
        <ul className="checks">
          {ev.checks.map(c => {
            const Icon = CHECK_ICON[c.state]
            return (
              <li key={c.label} className={c.state === 'ok' ? '' : c.state}>
                <Icon {...ICON} />{c.label}{c.meta && <span>{c.meta}</span>}
              </li>
            )
          })}
        </ul>
      )}

      {ev.passages.length > 0 && (
        <>
          <div className="ev-sec">Cited passages <span>{ev.passages.length}</span></div>
          {ev.passages.map(p => <Passage key={p.n} p={p} answer={message.content || ''} />)}
        </>
      )}
    </div>
  )
}
