/**
 * history.js — turns a saved conversation (GET /conversations/{id}) back
 * into the messages App renders, and dates the conversation list.
 */

/**
 * Each saved turn becomes a question and a finished answer. A turn's
 * `checks` hold what the done frame reported when it was answered; turns
 * saved before checks were recorded are marked `restored` so evidence.js
 * doesn't present defaults ("1 attempt, 100%") as a verified result.
 */
export function turnsToMessages(turns = []) {
  return turns.flatMap((t, i) => {
    const checks = t.checks || null
    return [
      { id: `h${i}-q`, role: 'user', content: t.query },
      {
        id: `h${i}-a`,
        role: 'bot',
        content: t.answer,
        sources: t.sources || [],
        intent: t.intent || 'qa',
        streaming: false,
        ...(checks
          ? {
            model_used: checks.model_used ?? 'none',
            latency_ms: checks.latency_ms ?? 0,
            flagged: checks.flagged ?? false,
            attempts: checks.attempts ?? 1,
            reflected: checks.reflected ?? false,
            reflection_reason: checks.reflection_reason ?? 'not_reflected',
            confidence: checks.confidence ?? null,
            failure_type: checks.failure_type ?? null,
          }
          : { restored: true, model_used: undefined, latency_ms: 0, confidence: null }),
      },
    ]
  })
}

const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR

/** "just now", "5 min ago", "3 h ago", "yesterday", "4 days ago", then the date. */
export function timeAgo(iso, now = Date.now()) {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const d = Math.max(0, now - t)
  if (d < MIN) return 'just now'
  if (d < HOUR) return `${Math.floor(d / MIN)} min ago`
  if (d < DAY) return `${Math.floor(d / HOUR)} h ago`
  if (d < 2 * DAY) return 'yesterday'
  if (d < 7 * DAY) return `${Math.floor(d / DAY)} days ago`
  return new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}
