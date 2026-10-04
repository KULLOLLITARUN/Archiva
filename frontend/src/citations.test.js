import { describe, expect, it } from 'vitest'
import { answerHtml, citedNumbers, dropOpenTag, parseTag } from './citations.js'

const sources = [
  { filename: 'Azure_VM_Cloning_Plan.docx', page: 1 },
  { filename: 'Azure_VM_Cloning_Plan.docx', page: 2 },
  { filename: 'A4-S1-GST-Invoice.pdf', page: 1 },
]
const cites = html => [...new DOMParser().parseFromString(html, 'text/html').querySelectorAll('.cite')]
  .map(b => ({ n: b.textContent, ns: b.dataset.ns }))

describe('parseTag', () => {
  it('reads file and page, and several files separated by semicolons', () => {
    expect(parseTag(' a.pdf, page 3')).toEqual([{ file: 'a.pdf', page: 3 }])
    expect(parseTag(' a.pdf, page 1; b.docx, pages 2-3')).toEqual([{ file: 'a.pdf', page: 1 }, { file: 'b.docx', page: 2 }])
    expect(parseTag(' notes.md')).toEqual([{ file: 'notes.md', page: null }])
  })
})

describe('citedNumbers', () => {
  it('matches file and page', () => {
    expect(citedNumbers(' Azure_VM_Cloning_Plan.docx, page 2', sources)).toEqual([2])
  })

  it('falls back to every passage from the file when the page is unknown', () => {
    expect(citedNumbers(' Azure_VM_Cloning_Plan.docx', sources)).toEqual([1, 2])
    expect(citedNumbers(' Azure_VM_Cloning_Plan.docx, page 9', sources)).toEqual([1, 2])
  })

  it('tolerates the look-alike hyphens models copy into file names', () => {
    expect(citedNumbers(' A4‑S1‑GST‑Invoice.pdf, page 1', sources)).toEqual([3])
  })

  it('returns nothing for a file that is not among the sources', () => {
    expect(citedNumbers(' other.pdf, page 1', sources)).toEqual([])
  })
})

describe('answerHtml', () => {
  it('replaces tags with numbered citations, including the full-width variant', () => {
    const html = answerHtml('Snapshot first. [Source: Azure_VM_Cloning_Plan.docx, page 1] Pay by UPI.【Source: A4-S1-GST-Invoice.pdf, page 1】',
      sources, { answerId: 'a1' })
    expect(html).not.toContain('Source:')
    expect(cites(html)).toEqual([{ n: '1', ns: '1' }, { n: '3', ns: '3' }])
  })

  it('collapses the same citation repeated back to back', () => {
    const html = answerHtml('Step one. [Source: A4-S1-GST-Invoice.pdf, page 1] [Source: A4-S1-GST-Invoice.pdf, page 1]', sources)
    expect(cites(html)).toHaveLength(1)
  })

  it('drops a tag that points at no listed source', () => {
    expect(cites(answerHtml('Claim. [Source: other.pdf, page 1]', sources))).toEqual([])
  })

  it('keeps tags as text when there are no sources, since then they are the only provenance', () => {
    expect(answerHtml('Claim. [Source: a.pdf, page 1]', [])).toContain('[Source: a.pdf, page 1]')
  })

  it('leaves ordinary brackets and links alone', () => {
    const html = answerHtml('See [the docs](https://example.com) and [note].', sources)
    expect(html).toContain('href="https://example.com"')
    expect(html).toContain('[note]')
  })

  it('hides a half-written tag while streaming, but not an unrelated bracket', () => {
    expect(answerHtml('Snapshot first. [Sou', sources, { streaming: true })).not.toContain('[Sou')
    expect(dropOpenTag('Use option [A')).toBe('Use option [A')
  })
})

describe('model-written source lines', () => {
  // Shape copied from a real answer: a bullet that only restates the
  // source, then the machine tag with a "|" separator and look-alike characters.
  const real = [
    '1. **Phase 0 – Safety snapshot**',
    '   * Create a safety snapshot of that OS disk.',
    '   * *Source:* Azure_VM_Cloning_Plan.docx, Page 1【Source: Azure_VM_Cloning_Plan.docx | Page 2】',
    '2. **Phase 1 – Clone 1**',
  ].join('\n')

  it('reads the "|" page separator', () => {
    expect(parseTag(' Azure_VM_Cloning_Plan.docx | Page 2')).toEqual([{ file: 'azure_vm_cloning_plan.docx', page: 2 }])
  })

  it('drops the restated source and moves its citation onto the claim above', () => {
    const doc = new DOMParser().parseFromString(answerHtml(real, sources, { answerId: 'a1' }), 'text/html')
    expect(doc.body.textContent).not.toMatch(/Source:/)
    const item = [...doc.querySelectorAll('li')].find(li => li.textContent.includes('safety snapshot of that OS disk'))
    expect(item.querySelector('.cite').dataset.ns).toBe('2')
    expect(doc.querySelectorAll('li').length).toBe(3)   // the source-only bullet is gone
  })

  it('removes an inline lead-in before a tag but keeps the sentence', () => {
    const html = answerHtml('Pay by UPI. Source: A4-S1-GST-Invoice.pdf, page 1 [Source: A4-S1-GST-Invoice.pdf, page 1]', sources)
    expect(html).toContain('Pay by UPI.')
    expect(html).not.toMatch(/Source:/)
    expect(cites(html)).toEqual([{ n: '3', ns: '3' }])
  })
})

describe('answerHtml look-alike spaces', () => {
  it('draws a narrow no-break space between words as a normal space', () => {
    // Seen live: "the hidden state for token t" rendered as "tokent".
    const html = answerHtml('The hidden state for token t is built from token t-1.', [])
    expect(html).toContain('token t is built from token t-1')
    expect(html).not.toMatch(/[  ]/)
  })

  it('still matches a citation whose file name has look-alike characters', () => {
    const sources = [{ filename: 'A4-S1-Invoice.pdf', page: 1, text: 't', score: 1 }]
    const html = answerHtml('Total is 5. [Source: A4‑S1‑Invoice.pdf, page 1]', sources, { answerId: 'a1' })
    expect(html).toContain('class="cite"')
  })
})
