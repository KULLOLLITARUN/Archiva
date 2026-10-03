/**
 * UploadPanel.jsx — Slide-in document management panel.
 *
 * Features:
 *  - Drag-and-drop OR click-to-browse file upload (.txt, .pdf, .docx)
 *  - Per-file upload progress with status badges
 *  - "Re-index Documents" button (POST /reload) to sync test_docs/ folder
 *  - Live file list with chunk counts and delete buttons
 *  - Animated toast notifications for upload results
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import {
  AlertTriangle, Check, FileText, FolderOpen, Info,
  RefreshCw, Trash2, Upload, X,
} from 'lucide-react'
import { apiUpload as uploadFile, apiReload as reloadDocs, apiDeleteFile as deleteFile, apiClearAllDocs } from '../api.js'
import { BRAND } from '../brand.js'

const ACCEPTED = '.txt,.pdf,.docx'
const ACCEPTED_TYPES = ['text/plain', 'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document']

function fileIcon() {
  return <FileText size={15} />
}

function StatusPill({ status }) {
  const map = {
    ok: { label: 'indexed', cls: 'pill--ok' },
    duplicate: { label: 'duplicate', cls: 'pill--skip' },
    uploading: { label: 'uploading', cls: 'pill--busy' },
    processing: { label: 'OCR running', cls: 'pill--busy' },
    failed: { label: 'failed', cls: 'pill--err' },
    error: { label: 'error', cls: 'pill--err' },
    limit: { label: 'limit', cls: 'pill--err' },
  }
  const { label, cls } = map[status] || { label: status, cls: '' }
  return <span className={`up-pill ${cls}`}>{label}</span>
}

/** One-line description of a listed document's state. */
export function fileMeta(f) {
  if (f.status === 'processing') {
    const [done, total] = f.progress || []
    return total ? `OCR in progress · page ${done}/${total}` : 'Queued for OCR'
  }
  if (f.status === 'failed') return f.message || 'Processing failed'
  return `${f.chunk_count} chunks`
}

const TOAST_ICON = { ok: <Check size={14} />, info: <Info size={14} />, err: <X size={14} /> }

function Toast({ toasts }) {
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast--${t.type}`}>
          <span className="toast-icon">{TOAST_ICON[t.type] ?? TOAST_ICON.info}</span>
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  )
}

export default function UploadPanel({ isOpen, docsInfo, onClose, onClearChat, onDocsChanged }) {
  const { files = [], total_files = 0, total_chunks = 0 } = docsInfo

  const [uploads, setUploads] = useState([])   // { name, status, chunks, msg }
  const [dragging, setDragging] = useState(false)
  const [reloading, setReloading] = useState(false)
  const [reloadLog, setReloadLog] = useState(null)
  const [clearing, setClearing] = useState(false)  // clear-all in progress
  const [confirmClear, setConfirmClear] = useState(false) // show confirm step
  const [toasts, setToasts] = useState([])
  const fileInputRef = useRef(null)
  const toastCounter = useRef(0)

  /** Show an auto-dismissing toast */
  const toast = useCallback((msg, type = 'ok') => {
    const id = ++toastCounter.current
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500)
  }, [])

  /** Upload a single File object */
  const handleFile = useCallback(async (file) => {
    if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED.includes('.' + file.name.split('.').pop())) {
      toast(`${file.name}: unsupported type`, 'err')
      return
    }
    const entry = { name: file.name, status: 'uploading', chunks: 0, msg: '' }
    setUploads(prev => [entry, ...prev])

    try {
      const res = await uploadFile(file)
      setUploads(prev => prev.map(u =>
        u.name === file.name && u.status === 'uploading'
          ? { ...u, status: res.status, chunks: res.chunk_count, msg: res.message }
          : u
      ))
      if (res.status === 'ok') {
        toast(`${file.name} — ${res.chunk_count} chunks indexed`, 'ok')
        onDocsChanged?.()
      } else if (res.status === 'processing') {
        toast(`${file.name} — scanned PDF, running OCR in the background`, 'info')
        onDocsChanged?.()
      } else if (res.status === 'duplicate') {
        toast(`${file.name} already indexed`, 'info')
      } else {
        toast(`${file.name}: ${res.message}`, 'err')
      }
    } catch (err) {
      setUploads(prev => prev.map(u =>
        u.name === file.name && u.status === 'uploading'
          ? { ...u, status: 'error', msg: err.message }
          : u
      ))
      toast(`${file.name}: ${err.message}`, 'err')
    }
  }, [toast, onDocsChanged])

  /** Drop handler */
  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = Array.from(e.dataTransfer.files)
    dropped.forEach(handleFile)
  }, [handleFile])

  /** Input change */
  const onInputChange = useCallback((e) => {
    Array.from(e.target.files).forEach(handleFile)
    e.target.value = ''   // reset so same file can be re-selected
  }, [handleFile])

  /** Re-index test_docs/ */
  const handleReload = useCallback(async () => {
    setReloading(true)
    setReloadLog(null)
    try {
      const res = await reloadDocs()
      setReloadLog(res)
      toast(`Re-indexed: +${res.loaded} files, ${res.total_chunks} chunks total`, 'ok')
      onDocsChanged?.()
    } catch (err) {
      toast(`Reload failed: ${err.message}`, 'err')
    } finally {
      setReloading(false)
    }
  }, [toast, onDocsChanged])

  /** Delete file from store */
  const handleDelete = useCallback(async (fileId, filename) => {
    try {
      await deleteFile(fileId)
      toast(`${filename} removed`, 'info')
      onDocsChanged?.()
    } catch (err) {
      toast(`Delete failed: ${err.message}`, 'err')
    }
  }, [toast, onDocsChanged])

  /** Clear ALL documents */
  const handleClearAll = useCallback(async () => {
    setClearing(true)
    try {
      const res = await apiClearAllDocs()
      toast(res.message || `Cleared ${res.count} document(s)`, 'info')
      onClearChat?.()
      onDocsChanged?.()
    } catch (err) {
      toast(`Clear failed: ${err.message}`, 'err')
    } finally {
      setClearing(false)
      setConfirmClear(false)
    }
  }, [toast, onDocsChanged, onClearChat])

  return (
    <>
      <Toast toasts={toasts} />

      {/* Backdrop */}
      <div
        className={`up-overlay ${isOpen ? 'up-overlay--open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`up-panel ${isOpen ? 'up-panel--open' : ''}`} aria-label="Document management">

        {/* ── Header ─────────────────────────────────────────────── */}
        <div className="up-header">
          <div className="up-header-left">
            <span className="up-header-icon"><FolderOpen size={17} /></span>
            <span className="up-header-title">{BRAND.name} — Documents</span>
          </div>
          <button className="up-close-btn" onClick={onClose} aria-label="Close panel"><X size={14} /></button>
        </div>

        {/* ── Stats bar ──────────────────────────────────────────── */}
        <div className="up-stats-bar">
          <div className="up-stat">
            <span className="up-stat-value">{total_files}</span>
            <span className="up-stat-label">files</span>
          </div>
          <div className="up-stat-divider" />
          <div className="up-stat">
            <span className="up-stat-value">{total_chunks}</span>
            <span className="up-stat-label">chunks</span>
          </div>
          <div className="up-stat-divider" />
          <div className="up-stat">
            <span className={`up-stat-value ${total_files > 0 ? 'up-stat-value--green' : 'up-stat-value--dim'}`}>
              {total_files > 0 ? 'ready' : 'empty'}
            </span>
            <span className="up-stat-label">status</span>
          </div>
        </div>

        <div className="up-body">

          {/* ── Drop zone ────────────────────────────────────────── */}
          <div
            className={`up-dropzone ${dragging ? 'up-dropzone--active' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Upload document"
            onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED}
              multiple
              onChange={onInputChange}
              className="up-hidden-input"
              aria-hidden="true"
            />
            <div className="up-dropzone-icon">
              {dragging ? <FolderOpen size={26} strokeWidth={1.5} /> : <Upload size={26} strokeWidth={1.5} />}
            </div>
            <p className="up-dropzone-primary">
              {dragging ? 'Drop to upload' : 'Drop files or click to browse'}
            </p>
            <p className="up-dropzone-secondary">.txt · .pdf · .docx · scanned PDFs are OCR'd</p>
          </div>

          {/* ── Recent uploads ───────────────────────────────────── */}
          {uploads.length > 0 && (
            <div className="up-section">
              <h3 className="up-section-title">Upload Queue</h3>
              <ul className="up-upload-list">
                {uploads.map((u, i) => (
                  <li key={i} className="up-upload-item">
                    <span className="up-upload-icon">{fileIcon(u.name)}</span>
                    <span className="up-upload-name">{u.name}</span>
                    <StatusPill status={u.status} />
                    {u.chunks > 0 && (
                      <span className="up-upload-chunks">{u.chunks} chunks</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* ── Re-index button ──────────────────────────────────── */}
          <div className="up-section">
            <h3 className="up-section-title">Server Docs Folder</h3>
            <p className="up-hint">Re-scans the <code>test_docs/</code> folder on the server and indexes any new files into {BRAND.name}.</p>
            <button
              className={`up-reload-btn ${reloading ? 'up-reload-btn--busy' : ''}`}
              onClick={handleReload}
              disabled={reloading}
            >
              {reloading
                ? <><span className="up-spinner" />Indexing…</>
                : <><RefreshCw size={14} /> Re-index Documents</>
              }
            </button>

            {reloadLog && (
              <div className="up-reload-result">
                <div className="up-reload-row">
                  <span className="up-reload-label">Indexed</span>
                  <span className="up-reload-val up-reload-val--ok">+{reloadLog.loaded}</span>
                </div>
                <div className="up-reload-row">
                  <span className="up-reload-label">Skipped</span>
                  <span className="up-reload-val">{reloadLog.skipped}</span>
                </div>
                {reloadLog.failed > 0 && (
                  <div className="up-reload-row">
                    <span className="up-reload-label">Failed</span>
                    <span className="up-reload-val up-reload-val--err">{reloadLog.failed}</span>
                  </div>
                )}
                <div className="up-reload-row">
                  <span className="up-reload-label">Total chunks</span>
                  <span className="up-reload-val">{reloadLog.total_chunks}</span>
                </div>
              </div>
            )}
          </div>

          {/* ── Loaded documents ─────────────────────────────────── */}
          <div className="up-section">
            <h3 className="up-section-title">Loaded Documents</h3>
            {files.length === 0 ? (
              <p className="up-empty">No documents indexed yet.</p>
            ) : (
              <ul className="up-file-list">
                {files.map((f, i) => (
                  <li key={i} className="up-file-item">
                    <span className="up-file-icon">{fileIcon(f.filename)}</span>
                    <div className="up-file-info">
                      <span className="up-file-name">{f.filename}</span>
                      <span className="up-file-meta">{fileMeta(f)}</span>
                    </div>
                    {(f.status === 'processing' || f.status === 'failed') && (
                      <StatusPill status={f.status} />
                    )}
                    {f.file_id && (
                      <button
                        className="up-delete-btn"
                        title={f.status === 'processing' ? `Cancel and remove ${f.filename}` : `Remove ${f.filename}`}
                        onClick={() => handleDelete(f.file_id, f.filename)}
                        aria-label={`Delete ${f.filename}`}
                      ><X size={12} /></button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* ── Danger zone ──────────────────────────────────────── */}
          <div className="up-section up-section--last up-danger-zone">
            <h3 className="up-section-title up-section-title--danger">
              <AlertTriangle size={11} style={{ display: 'inline', verticalAlign: -1, marginRight: 4 }} />
              Danger Zone
            </h3>

            <div className="up-danger-row">
              <div>
                <div className="up-danger-label">Clear All Documents</div>
                <div className="up-danger-desc">Permanently removes all {total_chunks} chunks from {total_files} file(s). Cannot be undone.</div>
              </div>

              {!confirmClear ? (
                <button
                  className="up-clear-btn"
                  onClick={() => setConfirmClear(true)}
                  disabled={total_files === 0 || clearing}
                  title={total_files === 0 ? 'No documents to clear' : 'Clear all RAG data'}
                >
                  <Trash2 size={13} /> Clear All
                </button>
              ) : (
                <div className="up-confirm-row">
                  <span className="up-confirm-label">Sure?</span>
                  <button className="up-confirm-yes" onClick={handleClearAll} disabled={clearing}>
                    {clearing ? <><span className="up-spinner" />Clearing…</> : 'Yes, clear'}
                  </button>
                  <button className="up-confirm-no" onClick={() => setConfirmClear(false)} disabled={clearing}>Cancel</button>
                </div>
              )}
            </div>

            <div className="up-danger-divider" />

            <button className="up-danger-btn" onClick={onClearChat}>
              <Trash2 size={13} /> Clear Chat History
            </button>
          </div>

        </div>
      </aside>
    </>
  )
}
