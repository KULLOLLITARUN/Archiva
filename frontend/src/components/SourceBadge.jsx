/** SourceBadge.jsx — Clickable source reference chip.
 *
 * Takes one GROUPED source ({ filename, pages: number[] }) — MessageBubble
 * merges every chunk pulled from the same file into a single entry before
 * this renders, so one document contributing 3 chunks shows one pill, not
 * three near-identical ones.
 *
 * No relevance score is shown. The API's `score` is an unbounded RRF
 * fusion score (see README), not a 0-100 confidence value — displaying it
 * as "3%" read as "barely relevant" when it means nothing of the sort, so
 * it's intentionally omitted rather than shown misleadingly.
 */
import { FileText } from 'lucide-react'

// Collapses [1, 2, 3, 5] into "pp. 1-3, 5" instead of listing every page.
function formatPages(pages) {
  if (!pages || pages.length === 0) return null
  const ranges = []
  let start = pages[0]
  let prev = pages[0]

  for (let i = 1; i <= pages.length; i++) {
    const current = pages[i]
    if (current === prev + 1) {
      prev = current
      continue
    }
    ranges.push(start === prev ? `${start}` : `${start}-${prev}`)
    start = prev = current
  }

  return `${ranges.length > 1 || ranges[0]?.includes('-') ? 'pp.' : 'p.'} ${ranges.join(', ')}`
}

export default function SourceBadge({ source }) {
  const name = source.filename || 'Unknown'
  const pageLabel = formatPages(source.pages)

  return (
    <div className="source-badge" title={pageLabel ? `${name} — ${pageLabel}` : name}>
      <span className="source-badge-icon"><FileText size={12} /></span>
      <span className="source-badge-name">{name}</span>
      {pageLabel && <span className="source-badge-page">{pageLabel}</span>}
    </div>
  )
}
