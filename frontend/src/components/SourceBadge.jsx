/** SourceBadge.jsx — Clickable source reference chip */

function fileIcon(filename) {
  const ext = filename?.split('.').pop()?.toLowerCase()
  if (ext === 'pdf')  return '📕'
  if (ext === 'docx') return '📘'
  return '📄'
}

export default function SourceBadge({ source }) {
  const name  = source.filename || 'Unknown'
  const page  = source.page || '?'
  const score = source.score != null ? Math.round(source.score * 100) : null

  return (
    <div className="source-badge" title={`${name} — Page ${page}`}>
      <span className="source-badge-icon">{fileIcon(name)}</span>
      <span className="source-badge-name">{name}</span>
      <span className="source-badge-page">p.{page}</span>
      {score != null && <span className="source-badge-page">{score}%</span>}
    </div>
  )
}
