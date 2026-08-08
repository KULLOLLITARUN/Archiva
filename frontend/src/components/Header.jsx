/**
 * Header.jsx — Archiva branded header.
 *
 * Logo: shared ArchivaLogo mark (geometric stacked-pages) — see
 * ArchivaLogo.jsx. No model names are shown here — only status info.
 */

import { useEffect, useState } from 'react'
import { BookOpen, FolderOpen, Settings } from 'lucide-react'
import { apiHealth as checkHealth } from '../api.js'
import { BRAND } from '../brand.js'
import ArchivaLogo from './ArchivaLogo.jsx'

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
