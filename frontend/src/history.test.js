import { describe, expect, it } from 'vitest'
import { timeAgo, turnsToMessages } from './history.js'
import { buildEvidence } from './evidence.js'

const SOURCES = [{ filename: 'plan.pdf', page: 2, text: 'Deployment takes 3-8 minutes.', score: 4 }]

describe('turnsToMessages', () => {
  it('restores a turn with its saved checks, so its evidence matches the original', () => {
    const [q, a] = turnsToMessages([{
      query: 'How long?', answer: 'About 3-8 minutes.', intent: 'qa', sources: SOURCES,
      checks: { attempts: 2, reflection_reason: 'passed_all_checks', confidence: 0.74, flagged: false, model_used: 'openai/gpt-oss-20b', latency_ms: 4100 },
    }])
    expect(q).toMatchObject({ role: 'user', content: 'How long?' })
    expect(a.restored).toBeUndefined()
    const ev = buildEvidence(a)
    expect(ev.kind).toBe('healed')
    expect(ev.confidence).toBe(0.74)
  })

  it('does not present a turn saved without checks as verified', () => {
    const [, a] = turnsToMessages([{ query: 'How long?', answer: 'About 3-8 minutes.', intent: 'qa', sources: SOURCES }])
    const ev = buildEvidence(a)
    expect(ev.kind).toBe('unrecorded')
    expect(ev.confidence).toBeNull()
    expect(ev.checks).toEqual([])
    expect(ev.passages).toHaveLength(1)
  })

  it('still shows an old "not found" turn as a refusal, without inventing the model', () => {
    const [, a] = turnsToMessages([{ query: 'Q', answer: 'Not found in the document.', intent: 'qa', sources: [] }])
    const ev = buildEvidence(a)
    expect(ev.kind).toBe('notfound')
    expect(ev.checks.map(c => c.label)).toEqual(['Refused rather than guessed'])
  })
})

describe('timeAgo', () => {
  const now = Date.parse('2026-10-04T12:00:00Z')
  it.each([
    ['2026-10-04T11:59:40Z', 'just now'],
    ['2026-10-04T11:55:00Z', '5 min ago'],
    ['2026-10-04T09:00:00Z', '3 h ago'],
    ['2026-10-03T09:00:00Z', 'yesterday'],
    ['2026-09-30T12:00:00Z', '4 days ago'],
  ])('%s -> %s', (iso, label) => expect(timeAgo(iso, now)).toBe(label))

  it('is empty for a missing time', () => expect(timeAgo(undefined, now)).toBe(''))
})

describe('a provider outage', () => {
  it('is "No answer", not a grounded answer with 0% confidence', () => {
    const ev = buildEvidence({
      role: 'bot', content: 'Service temporarily unavailable: the AI provider could not be reached or its rate limit / daily quota was reached.',
      reflection_reason: 'provider_unavailable', attempts: 1, confidence: 0, flagged: true, sources: [],
    })
    expect(ev.kind).toBe('error')
    expect(ev.title).toBe('No answer')
    expect(ev.confidence).toBeNull()
    expect(ev.checks).toEqual([])
  })
})
