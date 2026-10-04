/**
 * suggest.js — "Ask next" questions under the latest answer.
 *
 * Picked from the same /suggestions topics as the home page's starter
 * cards (questions the backend generated from the documents), ranked by
 * how many words they share with the last question and answer. Nothing
 * here claims a question is answerable; it's the same pool Home offers.
 */

const STOP = new Set(('the a an and or of to in on for is are was were be been by with as at from that this these those it its ' +
  'into than then there their they them which who whom what when where how why not no but if so such can could should would ' +
  'will may might must do does did has have had also each any all more most other some only same very just about over under ' +
  'your you used use using source page document documents').split(' '))

const words = text => new Set(
  (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 2 && !STOP.has(w)),
)
const norm = q => q.trim().toLowerCase().replace(/\s+/g, ' ')

/**
 * Up to `n` questions not yet asked in this conversation, most related to
 * the latest exchange first, at most `perTopic` from one topic so the
 * row isn't three rewordings of the same thing.
 */
export function nextQuestions(topics = [], messages = [], n = 3, perTopic = 2) {
  const asked = new Set(messages.filter(m => m.role === 'user').map(m => norm(m.content || '')))
  const lastQ = [...messages].reverse().find(m => m.role === 'user')?.content || ''
  const lastA = [...messages].reverse().find(m => m.role === 'bot')?.content || ''
  const context = words(`${lastQ} ${lastA}`)

  const pool = topics.flatMap((t, ti) => (t.prompts || []).map((prompt, pi) => ({ label: t.label, prompt, order: ti * 100 + pi })))
    .filter(c => !asked.has(norm(c.prompt)))
    .map(c => {
      let score = 0
      for (const w of words(`${c.label} ${c.prompt}`)) if (context.has(w)) score++
      return { ...c, score }
    })
    .sort((a, b) => b.score - a.score || a.order - b.order)

  const picked = []
  const used = {}
  for (const c of pool) {
    if ((used[c.label] || 0) >= perTopic) continue
    picked.push({ label: c.label, prompt: c.prompt })
    used[c.label] = (used[c.label] || 0) + 1
    if (picked.length === n) break
  }
  return picked
}
