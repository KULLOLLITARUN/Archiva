/**
 * MessageBubble.jsx — one turn of the conversation.
 *
 * Questions are dark bubbles. Answers read as text on the page: a frame
 * appears on hover and stays on the answer whose evidence is open, so a long
 * thread isn't a stack of boxes. Each answer carries its intent, a
 * Grounded / Self-healed pill, numbered citations, and a confidence bar.
 */

import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ArrowUpRight, Check, RefreshCw } from 'lucide-react'
import BrandMark from './BrandMark.jsx'
import { answerHtml } from '../citations.js'
import { buildEvidence } from '../evidence.js'

const INTENT_LABEL = { qa: 'Q&A', explain: 'Explain', summarize: 'Summary', compare: 'Compare', meta: 'About Archiva' }
const ICON = { size: 13, strokeWidth: 2, className: 'ico ico-sm', 'aria-hidden': true }

const STATUS_PILL = {
  grounded: { cls: 'pill-ok', Icon: Check, label: 'Grounded' },
  healed: { cls: 'pill-heal', Icon: RefreshCw, label: 'Self-healed' },
  unverified: { cls: 'pill-heal', Icon: AlertTriangle, label: 'Not fully verified' },
}

const notCovered = n => (n === 1 ? 'your document doesn’t' : n > 1 ? `none of your ${n} documents` : 'none of your documents')

const seconds = ms => `${(ms / 1000).toFixed(1)} s`

/** Shown until the first token arrives. The backend doesn't report live
 *  progress, so this says what happens and how long it's been, no more. */
function Thinking({ status }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t0 = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <div className="thinking" role="status">
      <span className="thinking-t">{status || 'Searching your documents'}</span>
      <span className="thinking-s">Each answer is checked against its sources before you see it.</span>
      <span className="thinking-lines" aria-hidden="true"><i /><i /><i /></span>
      {elapsed >= 2 && <span className="thinking-e tabular">{elapsed} s</span>}
    </div>
  )
}

export default function MessageBubble({ message, selected = false, onSelect, onCite, docCount = 0, status }) {
  const { id, role, content, sources = [], intent, streaming, isError, latency_ms } = message
  const ev = useMemo(() => (role === 'bot' && !streaming ? buildEvidence(message) : null), [message, role, streaming])
  const html = useMemo(
    () => (role === 'bot' ? answerHtml(content, sources, { answerId: id, streaming }) : ''),
    [role, content, sources, id, streaming],
  )

  if (role === 'user') {
    return <div className="q">{content}</div>
  }

  if (streaming && !content) {
    return (
      <article className="ans ans--wait">
        <BrandMark className="ans-av" />
        <div className="ans-card"><Thinking status={status} /></div>
      </article>
    )
  }

  if (ev?.kind === 'notfound' || ev?.kind === 'error') {
    const nf = ev.kind === 'notfound'
    return (
      <article className={`ans ans--nf${nf ? '' : ' ans--err'}${selected ? ' sel' : ''}`} data-a={id}>
        <BrandMark className="ans-av" />
        <div className="ans-card">
          <div className="ans-body">
            {nf ? (
              <p><strong>Not in your documents.</strong> Archiva answers only from what you&rsquo;ve loaded,
                and {notCovered(docCount)} cover this.</p>
            ) : (
              <p><strong>No answer.</strong> {content}</p>
            )}
          </div>
        </div>
      </article>
    )
  }

  const selectable = Boolean(onSelect) && !streaming && !isError
  const pill = ev && STATUS_PILL[ev.kind]
  const files = new Set(sources.map(s => s.filename)).size

  const onClick = e => {
    if (!selectable) return
    const cite = e.target.closest('.cite')
    if (cite) {
      e.preventDefault()
      onCite?.(id, cite.dataset.ns.split(' ').map(Number))
      return
    }
    if (e.target.closest('a')) return            // links in the answer work as links
    onSelect(id)
  }
  const onKeyDown = e => {
    // Enter on a citation button arrives here too; that one is its own click.
    if (e.target !== e.currentTarget) return
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(id) }
  }

  return (
    <article className={`ans${selected ? ' sel' : ''}`} data-a={id}>
      <BrandMark className="ans-av" />
      <div
        className="ans-card"
        {...(selectable ? {
          role: 'button', tabIndex: 0, 'aria-pressed': selected,
          'aria-label': 'Show evidence for this answer', onClick, onKeyDown,
        } : {})}
      >
        {!streaming && (intent || pill) && (
          <div className="ans-head">
            {INTENT_LABEL[intent] && <span className="pill pill-intent">{INTENT_LABEL[intent]}</span>}
            {pill && <span className={`pill ${pill.cls}`}><pill.Icon {...ICON} />{pill.label}</span>}
          </div>
        )}
        <div className={`ans-body${streaming ? ' streaming' : ''}`} dangerouslySetInnerHTML={{ __html: html }} />
        {!streaming && ev && (
          <div className="ans-foot">
            {ev.confidence != null && (
              <span className="conf" title="Confidence">
                <span className="bar"><i style={{ '--v': `${Math.round(ev.confidence * 100)}%` }} /></span>
                {Math.round(ev.confidence * 100)}%
              </span>
            )}
            {files > 0 && <span>{files} source{files === 1 ? '' : 's'}</span>}
            {latency_ms > 0 && <span className="tabular">{seconds(latency_ms)}</span>}
            {selectable && <span className="see">Evidence <ArrowUpRight {...ICON} /></span>}
          </div>
        )}
      </div>
    </article>
  )
}
