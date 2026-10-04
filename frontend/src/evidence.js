/**
 * evidence.js — turns a finished /chat answer into what the Evidence panel
 * shows: a verdict, the checks it passed, and the passages it cites.
 *
 * Only claims what the backend actually verified. Reflection
 * (agents/reflection.py) runs its checks in order and stops at the first
 * failure, and the response carries only the FINAL outcome:
 *   - "passed_all_checks": every check passed on the accepted attempt.
 *   - "<reason>_max_attempts_reached": the best answer still failed <reason>;
 *     checks after that one never ran, so they are not listed as passed.
 *   - "decomposed_N_subquestions": multi-hop; each sub-answer was reflected
 *     on separately, and failure_type is set if any of them failed.
 * Why an earlier, rejected attempt failed is not reported, so the panel
 * says a draft was rejected without naming a reason it doesn't know.
 */

import { maskModel } from './brand.js'
import { fileType } from './docs.js'

export { fileType }

const MAX_SUFFIX = '_max_attempts_reached'

const FAILED_CHECK = {
  ungrounded_numbers: 'A number in the answer is not in the source',
  ungrounded_numbers_strong_model_failed: 'A number in the answer is not in the source',
  low_overlap: 'Wording barely overlaps the passages',
  low_overlap_retry_model: 'Wording barely overlaps the passages',
  possible_contradiction: 'May contradict the source',
  answer_too_long: 'Answer longer than the check allows',
  answer_too_short: 'Answer too short to verify',
}

const NOT_FOUND_REASONS = ['explicit_not_found', 'no_chunks_retrieved', 'no_results', 'answer_too_short_max_attempts']

export function isNotFoundAnswer(m) {
  const reason = m.reflection_reason || ''
  return (m.content || '').toLowerCase().includes('not found in the document')
    || NOT_FOUND_REASONS.some(r => reason.startsWith(r))
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`
const seconds = ms => `${(ms / 1000).toFixed(1)} s`

/**
 * The backend's score is an unbounded fusion score, not a probability, so
 * relevance is shown relative to the best passage of the same answer: the
 * top passage gets 5 bars, the rest scale down from it (minimum 1).
 */
function relevanceBars(score, best) {
  if (!(best > 0) || !(score > 0)) return 1
  return Math.max(1, Math.min(5, Math.round((score / best) * 5)))
}
const REL_LABEL = ['', 'Weak match', 'Partial match', 'Good match', 'Strong match', 'Top match']

export function buildEvidence(m) {
  const attempts = m.attempts ?? 1
  const reason = m.reflection_reason || ''
  const { label: model } = maskModel(m.model_used)
  const sources = m.sources || []
  const files = new Set(sources.map(s => s.filename)).size
  const best = Math.max(0, ...sources.map(s => s.score || 0))
  const passages = sources.map((s, i) => {
    const rel = relevanceBars(s.score, best)
    return {
      n: i + 1,
      filename: s.filename || 'Unknown',
      page: s.page,
      type: fileType(s.filename),
      text: s.text || '',
      truncated: (s.text || '').length >= 200,   // the API sends the first 200 characters
      rel,
      relLabel: REL_LABEL[rel],
    }
  })
  const modelCheck = { state: 'info', label: `Answered by ${model}`, meta: m.latency_ms ? seconds(m.latency_ms) : '' }

  if (m.isError || reason === 'error') {
    return { kind: 'error', title: 'No answer', note: m.content || 'The request failed.', confidence: null, checks: [], passages: [] }
  }

  if (isNotFoundAnswer(m)) {
    return {
      kind: 'notfound',
      title: 'Not in your documents',
      note: 'No passage supported an answer, so Archiva declined instead of guessing.',
      confidence: null,
      checks: [{ state: 'ok', label: 'Refused rather than guessed' }, modelCheck],
      passages: [],
    }
  }

  const cites = files
    ? { state: 'ok', label: 'Cites its sources', meta: plural(files, 'doc') }
    : { state: 'warn', label: 'No source cited' }
  const retried = attempts > 1
    ? [{ state: 'warn', label: `Earlier ${attempts > 2 ? 'drafts' : 'draft'} rejected, retried`, meta: plural(attempts, 'attempt') }]
    : []
  const flagged = m.flagged ? [{ state: 'warn', label: 'Flagged by the output validator' }] : []
  const confidence = typeof m.confidence === 'number' ? m.confidence : null

  if (reason.endsWith(MAX_SUFFIX)) {
    const base = reason.slice(0, -MAX_SUFFIX.length)
    return {
      kind: 'unverified',
      title: 'Best effort, not fully verified',
      note: `After ${plural(attempts, 'attempt')} the best answer still failed a check. Read it against the passages.`,
      confidence,
      checks: [{ state: 'warn', label: FAILED_CHECK[base] || 'Failed a quality check' }, cites, ...flagged, modelCheck],
      passages,
    }
  }

  const multi = reason.match(/^decomposed_(\d+)_subquestions$/)
  if (multi) {
    const failed = m.failure_type && !['UNKNOWN', 'NONE'].includes(m.failure_type)
    return {
      kind: failed ? 'unverified' : 'grounded',
      title: failed ? 'Partly verified' : 'Grounded answer',
      note: `Split into ${multi[1]} sub-questions, each checked against its own sources.`,
      confidence,
      checks: [
        failed ? { state: 'warn', label: 'A sub-answer failed a check' } : { state: 'ok', label: 'Every sub-answer passed its checks' },
        cites, ...flagged, modelCheck,
      ],
      passages,
    }
  }

  const passed = [
    cites,
    { state: 'ok', label: 'Numbers match the source' },
    { state: 'ok', label: 'No contradictions found' },
    { state: 'ok', label: 'Wording matches the passages' },
  ]
  return {
    kind: retried.length ? 'healed' : 'grounded',
    title: retried.length ? 'Grounded after self-healing' : 'Grounded answer',
    note: retried.length
      ? 'A first draft failed a check, so Archiva retried before answering.'
      : 'Passed every check on the first attempt.',
    confidence,
    checks: [...retried, ...passed, ...flagged, modelCheck],
    passages,
  }
}

// ── Passage highlighting ──────────────────────────────────────────────────
// Approximate: the backend doesn't say which span supported which claim, so
// mark the passage words that also appear in the answer. Single words only
// count when they're distinctive (numbers, acronyms, long non-stopwords); common
// words only count as part of a run of two or more shared words.

const STOP = new Set(('the a an and or of to in on for is are was were be been by with as at from that this these those it its ' +
  'into than then there their they them which who whom what when where how why not no but if so such can could should would ' +
  'will may might must do does did has have had also each any all more most other some only same very just about over under ' +
  'source page document').split(' '))

const WORD_RE = /[\p{L}\p{N}][\p{L}\p{N}@._'-]*[\p{L}\p{N}]|[\p{L}\p{N}]/gu
const norm = w => w.toLowerCase().replace(/[.,]/g, '')

export function highlight(text, answer) {
  const answerWords = new Set((answer.match(WORD_RE) || []).map(norm))
  const tokens = [...text.matchAll(WORD_RE)].map(t => ({ start: t.index, end: t.index + t[0].length, raw: t[0], w: norm(t[0]) }))
  const shared = tokens.map(t => answerWords.has(t.w))
  // Numbers, long words, and acronyms (ICICI, IFSC) are specific enough alone;
  // a lone digit ("1", a page or step number) is not.
  const distinctive = t => (/\d/.test(t.w) && t.w.length >= 2) || (t.w.length >= 6 && !STOP.has(t.w)) || /^[A-Z]{3,}$/.test(t.raw)

  // Group shared tokens into runs (adjacent tokens with only spaces/punctuation between).
  const marks = []
  let i = 0
  while (i < tokens.length) {
    if (!shared[i]) { i++; continue }
    let j = i
    while (j + 1 < tokens.length && shared[j + 1] && /^[\s,:;·-]*$/.test(text.slice(tokens[j].end, tokens[j + 1].start))) j++
    const run = tokens.slice(i, j + 1)
    const content = run.filter(t => !STOP.has(t.w))
    if ((run.length >= 2 && content.length >= 1) || run.some(distinctive)) {
      marks.push([run[0].start, run[run.length - 1].end])
    }
    i = j + 1
  }

  const out = []
  let pos = 0
  for (const [s, e] of marks) {
    if (s > pos) out.push({ text: text.slice(pos, s), mark: false })
    out.push({ text: text.slice(s, e), mark: true })
    pos = e
  }
  if (pos < text.length) out.push({ text: text.slice(pos), mark: false })
  return out
}
