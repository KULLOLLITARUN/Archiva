/**
 * DocsStrip.jsx — Compact always-visible row of the currently loaded
 * documents, docked under the header.
 *
 * Previously the only signal that documents were loaded was the "N docs
 * ready" pill in the header — the actual files were hidden behind the
 * Documents panel until a user opened it. This surfaces them directly so
 * the empty chat reads as "your document workspace," not a blank AI
 * chatbot waiting for a first message.
 */

import { FileText, Plus } from 'lucide-react'

export default function DocsStrip({ docsInfo, onManage }) {
  const { files = [] } = docsInfo

  if (files.length === 0) return null

  return (
    <div className="docs-strip" role="list" aria-label="Loaded documents">
      {files.map(f => (
        <button
          key={f.filename}
          type="button"
          role="listitem"
          className="docs-strip-chip"
          onClick={onManage}
          title={`${f.filename} — ${f.chunk_count} chunk${f.chunk_count !== 1 ? 's' : ''}`}
        >
          <FileText size={13} />
          <span className="docs-strip-chip-name">{f.filename}</span>
          <span className="docs-strip-chip-count">{f.chunk_count}</span>
        </button>
      ))}
      <button type="button" className="docs-strip-add" onClick={onManage} title="Add or manage documents">
        <Plus size={13} />
        <span>Add</span>
      </button>
    </div>
  )
}
