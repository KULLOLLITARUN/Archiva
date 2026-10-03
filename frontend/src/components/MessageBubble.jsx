/** MessageBubble.jsx — Beautiful animated message with badges */

import { ClipboardList, GitCompare, Lightbulb } from 'lucide-react'
import ArchivaLogo from './ArchivaLogo.jsx'
import SourceBadge from './SourceBadge.jsx'
import { maskModel, modelTierStyle } from '../brand.js'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const INTENT_TAG = {
  compare: { icon: <GitCompare size={12} />, label: 'Compare' },
  summarize: { icon: <ClipboardList size={12} />, label: 'Summary' },
  explain: { icon: <Lightbulb size={12} />, label: 'Explain' },
  qa: null,
}

const REASON_LABELS = {
  passed_all_checks: 'Accepted first try',
  low_overlap: 'Expanded search',
  low_overlap_retry_model: 'Used stronger AI',
  answer_too_short: 'Answer too short, retried',
  ungrounded_numbers: 'Hallucination check',
  ungrounded_numbers_strong_model_failed: 'Numbers unverifiable',
  possible_contradiction: 'Contradiction detected',
  no_results_after_retry: 'Not found after retry',
  explicit_not_found: 'Not in documents',
  answer_too_long: 'Trimmed with stronger AI',
  max_attempts_no_result: 'Max attempts reached',
  no_chunks_retrieved: 'No matching chunks',
}

// Colors here are hardcoded (not var(--…)) because they're computed in JS,
// not CSS — kept in sync with the palette in styles.css :root by hand.
// These are the "as text" variants (darker/more saturated than the base
// --red/--accent tokens) since a badge on the light --surface background
// needs the text itself to carry contrast, not just a translucent tint.
function badgeVariant(attempts, reason) {
  const isRefused = reason?.includes('not_found') || reason?.includes('no_result') || reason?.includes('no_chunks')
  if (isRefused) return { bg: 'rgba(107,95,77,0.10)', color: '#6b5f4d', border: 'rgba(107,95,77,0.3)' }
  if (attempts >= 3) return { bg: 'rgba(161,72,58,0.10)', color: '#a1483a', border: 'rgba(161,72,58,0.3)' }
  return { bg: 'rgba(139,74,18,0.10)', color: '#8b4a12', border: 'rgba(139,74,18,0.3)' }
}

// ── Inline citations ─────────────────────────────────────────────────────────
// The model writes "[Source: file.docx, page 1]" after every claim, and the
// same provenance is already shown as source pills under the answer - so the
// text version just repeats it (five bullets, five identical tags). When
// pills are present the inline tags are hidden for display only; the stored
// answer and exports keep them, and an answer with no pills keeps them too,
// since then they are the only provenance shown.
const INLINE_CITATION_RE = /[ \t]*[\[【]\s*Source:[^\]】]*[\]】]/gi

// A citation still streaming in ("[Sou", "[Source: Azu") has no closing
// bracket yet and would flash on screen before being removed.
const OPEN_CITATION_TAIL_RE = /[ \t]*[\[【]\s*([A-Za-z]{0,6}:?[^\]】]{0,200})$/

export function stripInlineCitations(text, streaming = false) {
  let out = (text || '').replace(INLINE_CITATION_RE, '')
  if (streaming) {
    const tail = out.match(OPEN_CITATION_TAIL_RE)
    // Only drop it if what follows the bracket is (the start of) "source:".
    if (tail) {
      const inside = tail[1].toLowerCase()
      if ('source:'.startsWith(inside) || inside.startsWith('source:')) {
        out = out.slice(0, tail.index)
      }
    }
  }
  return out.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
}

function isNotFound(text) {
  return text?.toLowerCase().includes('not found in the document')
}

// ── Source grouping ──────────────────────────────────────────────────────────
// The API returns one entry per retrieved CHUNK, so a single answer built
// from 3 chunks of the same file used to render 3 near-identical pills
// ("file.pdf p.1 3%", "file.pdf p.1 3%", "file.pdf p.2 3%"). Grouped by
// filename here so each document gets exactly one pill listing every page
// it contributed. The per-chunk score is dropped entirely rather than
// shown as a fake "3%" — it's an unbounded RRF fusion score (see README),
// not a confidence percentage, and displaying it as one was actively
// misleading, not just cluttered.
function groupSourcesByFile(sources) {
  const byFile = new Map()
  for (const src of sources) {
    const key = src.filename || 'Unknown'
    if (!byFile.has(key)) byFile.set(key, { filename: key, pages: new Set() })
    if (src.page != null) byFile.get(key).pages.add(src.page)
  }
  return [...byFile.values()].map(f => ({
    filename: f.filename,
    pages: [...f.pages].sort((a, b) => a - b),
  }))
}

export default function MessageBubble({ message }) {
  const {
    role, content, sources, intent, model_used,
    latency_ms, isError, streaming,
    reflected, attempts, reflection_reason, confidence,
    failure_type, retrieval_latency_ms, tokens_used,
  } = message

  if (role === 'user') {
    return (
      <div className="msg-row msg-row--user">
        <div className="bubble bubble--user">
          <p className="bubble-text" style={{ whiteSpace: 'pre-wrap' }}>{content}</p>
        </div>
        <div className="avatar avatar--user">You</div>
      </div>
    )
  }

  const notFound = isNotFound(content)
  const intentTag = intent ? INTENT_TAG[intent] : null
  const showBadge = !streaming && (reflected || (attempts != null && attempts > 1))
  const variant = badgeVariant(attempts, reflection_reason)
  const reasonText = REASON_LABELS[reflection_reason] ?? (reflection_reason || '').replace(/_/g, ' ')
  const groupedSources = sources?.length ? groupSourcesByFile(sources) : []
  // Pills carry the provenance, so the repeated inline tags are hidden (see stripInlineCitations).
  const displayContent = groupedSources.length ? stripInlineCitations(content, streaming) : content

  // ── Model display: always use branded label, never the raw id ──────────────
  const { label: modelLabel, tier: modelTier } = maskModel(model_used)

  return (
    <div className="msg-row msg-row--bot">
      <div className="avatar avatar--bot"><ArchivaLogo size={20} /></div>
      <div className="bot-content">

        {/* Intent tag */}
        {intentTag && !streaming && (
          <div className="intent-tag">
            <span>{intentTag.icon}</span>
            <span>{intentTag.label}</span>
          </div>
        )}

        {/* Bubble */}
        <div className={`bubble bubble--bot${notFound ? ' bubble--notfound' : ''}${isError ? ' bubble--error' : ''}`}>
          <div
            className="bubble-text markdown-body"
            dangerouslySetInnerHTML={{
              __html: DOMPurify.sanitize(marked.parse(displayContent + (streaming ? ' ▋' : '')))
            }}
          />

          {/* Reflection badge */}
          {showBadge && (
            <div
              className="reflection-badge"
              style={{ background: variant.bg, color: variant.color, border: `1px solid ${variant.border}` }}
            >
              {reasonText}
              {typeof confidence === 'number' && (
                <span className="reflection-badge__confidence">
                  &nbsp;· {Math.round(confidence * 100)}% confident
                </span>
              )}
            </div>
          )}

          {/* Sources */}
          {!streaming && groupedSources.length > 0 && !notFound && (
            <div className="source-list">
              {groupedSources.map((src) => <SourceBadge key={src.filename} source={src} />)}
            </div>
          )}
        </div>


      </div>
    </div>
  )
}
