/**
 * Header.jsx — Archiva branded header.
 *
 * Logo: custom SVG mark (geometric stacked-pages with a spark).
 * No model names are shown here — only status info.
 */

import { useEffect, useState } from 'react'
import { apiHealth as checkHealth } from '../api.js'
import { BRAND } from '../brand.js'

/* ── Archiva SVG logo mark ──────────────────────────────────────────────────
   Three stacked parallelogram "pages" with a vertical spark/lightning bolt.
   Colours: indigo top → teal bottom. Fully original, no trademark conflicts.
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
      <defs>
        <linearGradient id="lg1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#7c6fff" />
          <stop offset="100%" stopColor="#2dd4bf" />
        </linearGradient>
        <linearGradient id="lg2" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#9b8dff" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.5" />
        </linearGradient>
        <linearGradient id="lg3" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#b8a6ff" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.3" />
        </linearGradient>
      </defs>

      {/* Bottom page (farthest back) */}
      <rect x="6" y="22" width="22" height="10" rx="3" fill="url(#lg3)" />

      {/* Middle page */}
      <rect x="4" y="15" width="22" height="10" rx="3" fill="url(#lg2)" />

      {/* Top page (foreground) */}
      <rect x="2" y="8" width="22" height="10" rx="3" fill="url(#lg1)" />

      {/* Spark / lightning bolt — floats over all pages */}
      <path
        d="M27 4 L22 18 H27 L20 34 L31 16 H25.5 L30 4Z"
        fill="url(#lg1)"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth="0.5"
      />
    </svg>
  )
}

export default function Header({ docsInfo, onUploadClick, onPlaybookClick, onAdminClick }) {
  const { total_files = 0 } = docsInfo
  const [online, setOnline] = useState(false)

  useEffect(() => {
    checkHealth().then(() => setOnline(true)).catch(() => setOnline(false))
    const id = setInterval(() =>
      checkHealth().then(() => setOnline(true)).catch(() => setOnline(false)),
    30_000)
    return () => clearInterval(id)
  }, [])

  const statusLabel = !online
    ? 'Offline'
    : total_files > 0
      ? `${total_files} doc${total_files !== 1 ? 's' : ''} ready`
      : 'No documents'

  const dotClass = !online
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
            <span className="icon">⚙</span>
            <span>Admin</span>
          </button>
        )}

        <button className="hdr-btn" onClick={onPlaybookClick} title="Open Playbook">

          <span className="icon">📖</span>
          <span>Playbook</span>
        </button>
        <button className="hdr-btn hdr-btn--accent" onClick={onUploadClick} title="Manage Documents">
          <span className="icon">📂</span>
          <span>Documents</span>
        </button>
      </div>
    </header>
  )
}
