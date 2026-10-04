/**
 * docs.js — what the UI knows about a document from /docs-loaded:
 * which files can be uploaded, the type badge to show, and its status line.
 */

// Mirrors SUPPORTED_EXTENSIONS in ingestion/parser.py.
export const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.csv', '.html', '.htm']
export const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(',')

export function extensionOf(filename = '') {
  const i = filename.lastIndexOf('.')
  return i < 0 ? '' : filename.slice(i).toLowerCase()
}

export function isAccepted(filename) {
  return ACCEPTED_EXTENSIONS.includes(extensionOf(filename))
}

/** Colour family for the type badge: pdf | doc | ocr | txt (see .ft-* in styles.css). */
export function fileType(filename = '') {
  const ext = extensionOf(filename)
  if (ext === '.pdf') return 'pdf'
  if (ext === '.docx' || ext === '.doc') return 'doc'
  return 'txt'
}

/**
 * Badge for a library row. Only scans go through background OCR, so a
 * processing or failed document is a scan, and /docs-loaded marks an
 * indexed one with `ocr`. (Scans indexed before that flag existed show
 * as PDF; nothing recorded how they were read.)
 */
export function docBadge(f) {
  if (f.ocr || f.status === 'processing' || f.status === 'failed') return { type: 'ocr', label: 'OCR' }
  const ext = extensionOf(f.filename)
  const label = { '.pdf': 'PDF', '.docx': 'DOC', '.doc': 'DOC', '.md': 'MD', '.csv': 'CSV', '.html': 'HTML', '.htm': 'HTML' }[ext] || 'TXT'
  return { type: fileType(f.filename), label }
}

/**
 * Status line and progress (0..1, or null when unknown) for a library row.
 * progress is [pages_done, pages_total] while OCR runs, null while queued.
 */
export function docStatus(f) {
  if (f.status === 'processing') {
    const [done, total] = f.progress || []
    return total
      ? { tone: 'busy', text: `Reading page ${Math.min(done + 1, total)} of ${total}`, progress: done / total }
      : { tone: 'busy', text: 'Queued for OCR', progress: null }
  }
  if (f.status === 'failed') return { tone: 'bad', text: f.message || 'Processing failed' }
  const n = f.chunk_count ?? 0
  return { tone: 'ok', text: `${n.toLocaleString()} passage${n === 1 ? '' : 's'}` }
}
