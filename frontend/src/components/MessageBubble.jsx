/** MessageBubble.jsx — Beautiful animated message with badges */

import SourceBadge from './SourceBadge.jsx'
import { maskModel, modelTierStyle } from '../brand.js'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const INTENT_TAG = {
  compare:   { icon: '🔀', label: 'Compare' },
  summarize: { icon: '📋', label: 'Summary' },
  explain:   { icon: '💡', label: 'Explain' },
  qa:        null,
}

const REASON_LABELS = {
  passed_all_checks:                       '✓ Accepted first try',
  low_overlap:                             '🔄 Expanded search',
  low_overlap_retry_model:                 '🔄 Used stronger AI',
  answer_too_short:                        '🔄 Answer too short, retried',
  ungrounded_numbers:                      '🛡 Hallucination check',
  ungrounded_numbers_strong_model_failed:  '⚠ Numbers unverifiable',
  possible_contradiction:                  '⚠ Contradiction detected',
  no_results_after_retry:                  '✕ Not found after retry',
  explicit_not_found:                      '✕ Not in documents',
  answer_too_long:                         '✂ Trimmed with stronger AI',
  max_attempts_no_result:                  '✕ Max attempts reached',
  no_chunks_retrieved:                     '✕ No matching chunks',
}

function badgeVariant(attempts, reason) {
  const isRefused = reason?.includes('not_found') || reason?.includes('no_result') || reason?.includes('no_chunks')
  if (isRefused)    return { bg: 'rgba(156,163,175,0.12)', color: '#9ca3af',   border: 'rgba(156,163,175,0.25)' }
  if (attempts >= 3) return { bg: 'rgba(248,113,113,0.12)', color: '#f87171', border: 'rgba(248,113,113,0.3)' }
  return               { bg: 'rgba(251,191,36,0.12)',  color: '#fbbf24',  border: 'rgba(251,191,36,0.3)' }
}

function isNotFound(text) {
  return text?.toLowerCase().includes('not found in the document')
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

  const notFound   = isNotFound(content)
  const intentTag  = intent ? INTENT_TAG[intent] : null
  const showBadge  = !streaming && (reflected || (attempts != null && attempts > 1))
  const variant    = badgeVariant(attempts, reflection_reason)
  const reasonText = REASON_LABELS[reflection_reason] ?? (reflection_reason || '').replace(/_/g, ' ')

  // ── Model display: always use branded label, never the raw id ──────────────
  const { label: modelLabel, tier: modelTier } = maskModel(model_used)

  return (
    <div className="msg-row msg-row--bot">
      <div className="avatar avatar--bot">◈</div>
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
              __html: DOMPurify.sanitize(marked.parse(content + (streaming ? ' ▋' : '')))
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
          {!streaming && sources?.length > 0 && !notFound && (
            <div className="source-list">
              {sources.map((src, i) => <SourceBadge key={i} source={src} />)}
            </div>
          )}
        </div>


      </div>
    </div>
  )
}
