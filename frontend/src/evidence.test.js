import { describe, expect, it } from 'vitest'
import { buildEvidence, fileType, highlight } from './evidence.js'

const base = {
  id: 'a1', role: 'bot', content: 'The invoice lists ICICI as the bank and ifox@icici as the UPI ID.',
  model_used: 'openai/gpt-oss-20b', latency_ms: 2400, attempts: 1, reflected: false,
  reflection_reason: 'passed_all_checks', confidence: 0.92, flagged: false,
  sources: [
    { filename: 'invoice.pdf', page: 1, text: 'Bank Details · Name ICICI · UPI ID ifox@icici', score: 0.03 },
    { filename: 'invoice.pdf', page: 1, text: 'Total Amount After Tax 4,490.00', score: 0.012 },
  ],
}
const labels = ev => ev.checks.map(c => `${c.state}:${c.label}`)

describe('buildEvidence', () => {
  it('reports a first-try answer as grounded with every check passed', () => {
    const ev = buildEvidence(base)
    expect(ev.kind).toBe('grounded')
    expect(ev.confidence).toBe(0.92)
    expect(labels(ev)).toEqual([
      'ok:Cites its sources', 'ok:Numbers match the source', 'ok:No contradictions found',
      'ok:Wording matches the passages', 'info:Answered by Archiva Swift',
    ])
    expect(ev.checks[0].meta).toBe('1 doc')
    expect(ev.checks.at(-1).meta).toBe('2.4 s')
  })

  it('marks an answer accepted after a retry as self-healed, without inventing why', () => {
    const ev = buildEvidence({ ...base, attempts: 2, reflected: true })
    expect(ev.kind).toBe('healed')
    expect(ev.checks[0]).toMatchObject({ state: 'warn', label: 'Earlier draft rejected, retried', meta: '2 attempts' })
  })

  it('does not claim later checks passed when the best answer still failed one', () => {
    const ev = buildEvidence({ ...base, attempts: 3, reflection_reason: 'ungrounded_numbers_strong_model_failed_max_attempts_reached', confidence: 0 })
    expect(ev.kind).toBe('unverified')
    expect(labels(ev)).toContain('warn:A number in the answer is not in the source')
    expect(labels(ev)).not.toContain('ok:No contradictions found')
    expect(labels(ev)).not.toContain('ok:Numbers match the source')
  })

  it('shows a refusal with no ring and no passages', () => {
    const ev = buildEvidence({ ...base, content: 'Not found in the document.', reflection_reason: 'explicit_not_found', confidence: 0, sources: [] })
    expect(ev.kind).toBe('notfound')
    expect(ev.confidence).toBeNull()
    expect(ev.passages).toEqual([])
  })

  it('flags an answer the output validator flagged', () => {
    expect(labels(buildEvidence({ ...base, flagged: true }))).toContain('warn:Flagged by the output validator')
  })

  it('reports multi-hop answers by sub-question outcome', () => {
    const ok = buildEvidence({ ...base, reflection_reason: 'decomposed_2_subquestions', failure_type: 'UNKNOWN' })
    expect(ok.kind).toBe('grounded')
    expect(ok.note).toMatch(/2 sub-questions/)
    const bad = buildEvidence({ ...base, reflection_reason: 'decomposed_2_subquestions', failure_type: 'HALLUCINATION' })
    expect(bad.kind).toBe('unverified')
  })

  it('numbers passages and ranks relevance against the best one', () => {
    const [a, b] = buildEvidence(base).passages
    expect([a.n, b.n]).toEqual([1, 2])
    expect(a.rel).toBe(5)
    expect(a.relLabel).toBe('Top match')
    expect(b.rel).toBe(2)
  })

  it('warns when an answer cites nothing', () => {
    expect(labels(buildEvidence({ ...base, sources: [] }))[0]).toBe('warn:No source cited')
  })
})

describe('fileType', () => {
  it('maps extensions to badge types', () => {
    expect(fileType('a.PDF')).toBe('pdf')
    expect(fileType('plan.docx')).toBe('doc')
    expect(fileType('notes.md')).toBe('txt')
  })
})

describe('highlight', () => {
  const marked = (text, answer) => highlight(text, answer).filter(s => s.mark).map(s => s.text)

  it('marks distinctive words and numbers the answer shares with the passage', () => {
    expect(marked('Name ICICI · Branch Surat · UPI ID ifox@icici', base.content)).toEqual(['ICICI', 'UPI ID ifox@icici'])
    expect(marked('Total Amount After Tax 4,490.00', 'The total is 4,490.00.')).toEqual(['4,490.00'])
  })

  it('marks runs of shared words but not lone common words', () => {
    expect(marked('Take a snapshot of the OS disk first', 'Take a snapshot of the OS disk.')).toEqual(['Take a snapshot of the OS disk'])
    expect(marked('the bank is open', 'the answer')).toEqual([])
    expect(marked('Zone 1 only', 'Use step 1.')).toEqual([])
  })

  it('keeps the full text when nothing matches', () => {
    expect(highlight('nothing here', 'unrelated')).toEqual([{ text: 'nothing here', mark: false }])
  })
})
