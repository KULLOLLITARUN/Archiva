/** SourceBadge.jsx — Clickable source reference chip */

import { FileText } from 'lucide-react'

export default function SourceBadge({ source }) {
  const name  = source.filename || 'Unknown'
  const page  = source.page || '?'
  const score = source.score != null ? Math.round(source.score * 100) : null

  return (
    <div className="source-badge" title={`${name} — Page ${page}`}>
      <span className="source-badge-icon"><FileText size={12} /></span>
      <span className="source-badge-name">{name}</span>
      <span className="source-badge-page">p.{page}</span>
      {score != null && <span className="source-badge-page">{score}%</span>}
    </div>
  )
}
