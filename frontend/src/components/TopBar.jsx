/**
 * TopBar.jsx — the conversation's header: library menu (phones), title,
 * backend status, export (only once there is something to export) and the
 * evidence toggle.
 */

import { useEffect, useRef, useState } from 'react'
import { Download, FileDown, FileText, Menu, PanelRight } from 'lucide-react'
import { apiHealth } from '../api.js'

const ICON = { size: 16, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

function useOnline() {
  const [online, setOnline] = useState(true)
  useEffect(() => {
    const check = () => apiHealth().then(() => setOnline(true)).catch(() => setOnline(false))
    check()
    const id = setInterval(check, 30_000)
    return () => clearInterval(id)
  }, [])
  return online
}

function ExportMenu({ onExport }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const away = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    const esc = e => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', away)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', away); document.removeEventListener('keydown', esc) }
  }, [open])

  const pick = format => { setOpen(false); onExport(format) }

  return (
    <div className="menu-wrap" ref={ref}>
      <button type="button" className="tbtn" aria-haspopup="menu" aria-expanded={open}
        aria-label="Export conversation" onClick={() => setOpen(o => !o)}>
        <Download {...ICON} /><span>Export</span>
      </button>
      {open && (
        <div className="menu" role="menu">
          <button type="button" role="menuitem" onClick={() => pick('markdown')}><FileText {...ICON} />Markdown (.md)</button>
          <button type="button" role="menuitem" onClick={() => pick('pdf')}><FileDown {...ICON} />PDF</button>
        </div>
      )}
    </div>
  )
}

export default function TopBar({
  title, subtitle, docsInfo, backendStatus, hasMessages, evOpen,
  onMenu, onToggleEvidence, onExport,
}) {
  const online = useOnline()
  const total = docsInfo.total_files || 0

  let status = { cls: 'status--ok', label: `${total} ready` }
  if (backendStatus) status = { cls: 'status--wait', label: backendStatus }
  else if (!online) status = { cls: 'status--off', label: 'Offline' }
  else if (total === 0) status = { cls: 'status--idle', label: 'No documents' }

  return (
    <header className="top">
      <button type="button" className="ibtn only-narrow" onClick={onMenu}
        aria-label="Open library" aria-controls="library">
        <Menu {...ICON} />
      </button>
      <div className="top-title">
        <b>{title}</b>
        <span>{subtitle}</span>
      </div>
      <div className="top-sp">
        <span className={`status ${status.cls}`} role="status"><i />{status.label}</span>
        {hasMessages && <ExportMenu onExport={onExport} />}
        {hasMessages && (
          <button type="button" className="ibtn" onClick={onToggleEvidence}
            aria-label={evOpen ? 'Hide evidence' : 'Show evidence'} aria-pressed={evOpen} aria-controls="evidence">
            <PanelRight {...ICON} />
          </button>
        )}
      </div>
    </header>
  )
}
