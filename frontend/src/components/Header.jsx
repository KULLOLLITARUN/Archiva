/**
 * Header.jsx — Archiva branded header.
 *
 * Logo: custom SVG mark (geometric stacked-pages with a spark).
 * No model names are shown here — only status info.
 */

import { useEffect, useState } from 'react'
import { BookOpen, FolderOpen, Settings } from 'lucide-react'
import { apiHealth as checkHealth } from '../api.js'
import { BRAND } from '../brand.js'

/* ── Archiva SVG logo mark ──────────────────────────────────────────────────
   Three stacked parallelogram "pages" — a simple, solid ink-gold mark.
   Fully original, no trademark conflicts.
---------------------------------------------------------------------------- */
function ArchivaLogo() {
  return (
    <svg
      className="archiva-logo"
      viewBox="0 0 38 38"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Archiva logo"
    >
      {/* Bottom page (farthest back) */}
      <rect x="6" y="22" width="22" height="10" rx="3" fill="var(--accent)" opacity="0.35" />

      {/* Middle page */}
      <rect x="4" y="15" width="22" height="10" rx="3" fill="var(--accent)" opacity="0.65" />

      {/* Top page (foreground) */}
      <rect x="2" y="8" width="22" height="10" rx="3" fill="var(--accent)" />
    </svg>
  )
}

export default function Header({ docsInfo, onUploadClick, onPlaybookClick, onAdminClick, backendStatus = '' }) {
  const { total_files = 0 } = docsInfo
  const [online, setOnline] = useState(false)

  useEffect(() => {
    checkHealth().then(() => setOnline(true)).catch(() => setOnline(false))
    const id = setInterval(() =>
      checkHealth().then(() => setOnline(true)).catch(() => setOnline(false)),
    30_000)
    return () => clearInterval(id)
  }, [])

  // If a backendStatus message is set, show that instead of the normal pill
  const isConnecting = backendStatus !== ''

  const statusLabel = isConnecting
    ? backendStatus
    : !online
      ? 'Offline'
      : total_files > 0
        ? `${total_files} doc${total_files !== 1 ? 's' : ''} ready`
        : 'No documents'

  const dotClass = isConnecting
    ? 'status-dot status-dot--connecting'
    : !online
      ? 'status-dot'
      : total_files > 0
        ? 'status-dot status-dot--ready'
        : 'status-dot status-dot--empty'

  return (
    <header className="header">
      {/* Brand */}
      <div className="header-brand">
        <ArchivaLogo />
        <div>
          <div className="header-name">{BRAND.name}</div>
          <div className="header-tagline">{BRAND.sub}</div>
        </div>
      </div>

      {/* Status pill */}
      <div className="header-center">
        <div className="header-status">
          <span className={dotClass} />
          {statusLabel}
        </div>
      </div>

      {/* Actions */}
      <div className="header-actions">
        {/* Admin dashboard button */}
        {onAdminClick && (
          <button className="hdr-btn hdr-btn--admin" onClick={onAdminClick} title="Admin Dashboard">
            <span className="icon"><Settings size={15} /></span>
            <span>Admin</span>
          </button>
        )}

        <button className="hdr-btn" onClick={onPlaybookClick} title="Open Playbook">
          <span className="icon"><BookOpen size={15} /></span>
          <span>Playbook</span>
        </button>
        <button className="hdr-btn hdr-btn--accent" onClick={onUploadClick} title="Manage Documents">
          <span className="icon"><FolderOpen size={15} /></span>
          <span>Documents</span>
        </button>
      </div>
    </header>
  )
}
