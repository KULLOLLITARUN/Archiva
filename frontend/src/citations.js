/**
 * citations.js — turns the model's inline "[Source: file.pdf, page 1]" tags
 * into numbered citation buttons that point at the answer's passages.
 *
 * Passage n is sources[n - 1] (the order the Evidence panel lists them in).
 * A tag names a file and page, and several passages can share both, so a
 * citation shows the first matching number and lights all of them.
 */

import { marked } from 'marked'
import DOMPurify from 'dompurify'

const TAG_RE = /[ \t]*[[【]\s*Source:([^\]】]*)[\]】]/gi
// A citation still streaming in ("[Sou", "[Source: Azu") has no closing
// bracket yet and would flash on screen before being replaced.
const OPEN_TAG_TAIL_RE = /[ \t]*[[【]\s*([A-Za-z]{0,6}:?[^\]】]{0,200})$/

// Models swap in look-alike characters when they copy a file name
// (non-breaking hyphens, narrow spaces); same folding as main.py.
export function normalizeName(text = '') {
  return text.replace(/[‐-―−]/g, '-').replace(/[\s  ]+/g, ' ').trim().toLowerCase()
}

/** "a.pdf, page 1; b.docx | Page 2" → [{ file: 'a.pdf', page: 1 }, { file: 'b.docx', page: 2 }] */
export function parseTag(inner) {
  return inner.split(';').map(part => {
    const m = part.match(/^(.*?)(?:\s*[,|]\s*(?:page|pages|p\.|pp\.)\s*(\d+).*)?$/i)
    const file = normalizeName((m?.[1] || part).replace(/^\s*["']|["']\s*$/g, ''))
    return { file, page: m?.[2] ? Number(m[2]) : null }
  }).filter(c => c.file)
}

/** Passage numbers (1-based) a tag refers to: same file and page, else same file. */
export function citedNumbers(inner, sources) {
  const names = sources.map(s => normalizeName(s.filename))
  const out = new Set()
  for (const { file, page } of parseTag(inner)) {
    const sameFile = names.flatMap((n, i) => (n === file || n.includes(file) || file.includes(n) ? [i + 1] : []))
    const samePage = sameFile.filter(n => page == null || sources[n - 1].page === page)
    for (const n of (samePage.length ? samePage : sameFile)) out.add(n)
  }
  return [...out].sort((a, b) => a - b)
}

// The model sometimes also writes the source out for people to read,
// "*Source:* plan.docx, Page 1 [Source: plan.docx | Page 1]", often as a
// bullet of its own. With the tag shown as a citation number, that text is
// a duplicate: drop the lead-in, then fold a line left holding only tags
// onto the line above, so the number sits at the end of the claim.
const LEAD_IN_RE = /(?<![[【]\s{0,3})(?:\*\*|\*|_)?Sources?:(?:\*\*|\*|_)?[^\n[【]*?(?=[[【]\s*Source:)/gi
const TAGS_ONLY_LINE_RE = /^\s*(?:[-*+]|\d+[.)])?\s*((?:[ \t]*[[【]\s*Source:[^\]】]*[\]】])+)\s*$/i

export function foldSourceLines(text) {
  const lines = text.replace(LEAD_IN_RE, '').split('\n')
  const out = []
  for (const line of lines) {
    const tagsOnly = line.match(TAGS_ONLY_LINE_RE)
    const prev = out.findLastIndex(l => l.trim())
    if (tagsOnly && prev >= 0) out[prev] = out[prev].replace(/\s+$/, '') + ' ' + tagsOnly[1].trim()
    else out.push(line)
  }
  return out.join('\n')
}

/** Removes a half-written tag at the end of a streaming answer. */
export function dropOpenTag(text) {
  const tail = text.match(OPEN_TAG_TAIL_RE)
  if (!tail) return text
  const inside = tail[1].toLowerCase()
  return 'source:'.startsWith(inside) || inside.startsWith('source:') ? text.slice(0, tail.index) : text
}

function citeButton(nums, answerId) {
  const label = `Source${nums.length > 1 ? 's' : ''} ${nums.join(', ')}`
  return `<button type="button" class="cite" data-a="${answerId}" data-ns="${nums.join(' ')}" aria-label="${label}">${nums[0]}</button>`
}

/**
 * Sanitised HTML for an answer: markdown rendered, each source tag replaced
 * by a numbered citation. With no sources to point at, tags stay as text,
 * since then they're the only provenance shown. Repeating the same
 * citation back to back ("…[1] [1]") collapses to one.
 */
export function answerHtml(content = '', sources = [], { answerId = '', streaming = false } = {}) {
  let text = streaming ? dropOpenTag(content) : content
  if (sources.length) {
    text = foldSourceLines(text)
    text = text.replace(TAG_RE, (_, inner) => {
      const nums = citedNumbers(inner, sources)
      return nums.length ? ` ${citeButton(nums, answerId)}` : ''
    })
    text = text.replace(/(<button type="button" class="cite"[^>]*>\d+<\/button>)(\s*\1)+/g, '$1')
  }
  text = text.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
  return DOMPurify.sanitize(marked.parse(text), { ADD_ATTR: ['data-a', 'data-ns'] })
}
