/**
 * AdminDashboard.jsx — Admin control panel for Archiva.
 * Tabs: Overview, Documents
 * Users and Audit Log tabs removed (no auth / user management).
 */
import { useState, useEffect, useCallback } from 'react'
import {
  adminGetStats, adminGetDocuments,
  adminDeleteDocument, adminDeleteAllDocuments,
} from '../api.js'

const TABS = ['Overview', 'Documents']

export default function AdminDashboard({ onClose }) {
  const [tab,   setTab]   = useState('Overview')
  const [data,  setData]  = useState({})
  const [busy,  setBusy]  = useState(false)
  const [toast, setToast] = useState('')

  function showToast(msg) {
    setToast(msg)
    setTimeout(() => setToast(''), 3000)
  }

  const fetchTab = useCallback(async (t) => {
    setBusy(true)
    try {
      if (t === 'Overview') {
        const stats = await adminGetStats()
        setData(prev => ({ ...prev, stats }))
      }
      if (t === 'Documents') {
        const res = await adminGetDocuments()
        setData(prev => ({ ...prev, docs: res.documents }))
      }
    } catch (e) {
      showToast(`Error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => { fetchTab(tab) }, [tab, fetchTab])

  async function deleteDoc(id) {
    if (!confirm('Delete this document?')) return
    try {
      await adminDeleteDocument(id)
      showToast('Document deleted.')
      fetchTab('Documents')
    } catch (e) { showToast(`Error: ${e.message}`) }
  }

  async function deleteAllDocs() {
    if (!confirm('WARNING: This will delete ALL documents. Are you sure?')) return
    try {
      const res = await adminDeleteAllDocuments()
      showToast(`Deleted ${res.count} documents.`)
      fetchTab('Documents')
    } catch (e) { showToast(`Error: ${e.message}`) }
  }

  return (
    <div className="admin-overlay">
      <div className="admin-panel">

        {/* Header */}
        <div className="admin-header">
          <div className="admin-header-left">
            <span className="admin-badge">⚙ Admin</span>
            <span className="admin-title">System Dashboard</span>
          </div>
          <button className="admin-close-btn" onClick={onClose} aria-label="Close admin">✕</button>
        </div>

        {/* Tabs */}
        <div className="admin-tabs">
          {TABS.map(t => (
            <button
              key={t}
              className={`admin-tab-btn ${tab === t ? 'admin-tab-btn--active' : ''}`}
              onClick={() => setTab(t)}
            >{t}</button>
          ))}
          <button className="admin-refresh-btn" onClick={() => fetchTab(tab)} disabled={busy} title="Refresh">
            {busy ? '⟳' : '↺'}
          </button>
        </div>

        {/* Content */}
        <div className="admin-body">

          {/* ── Overview ───────────────────────────────────────────────── */}
          {tab === 'Overview' && (
            <div className="admin-overview">
              {data.stats ? (
                <>
                  <div className="admin-stat-grid">
                    {[
                      { label: 'Total Documents', value: data.stats.total_docs,    icon: '📄', color: 'var(--green)'   },
                      { label: 'Total Chunks',    value: data.stats.total_chunks,  icon: '🧩', color: 'var(--yellow)'  },
                      { label: 'Total Queries',   value: data.stats.total_queries, icon: '💬', color: 'var(--orange)'  },
                      { label: 'Success Rate',    value: `${data.stats.success_rate}%`, icon: '✅', color: 'var(--green)' },
                      { label: 'Live Chunks',     value: data.stats.store_chunks,  icon: '🗄', color: 'var(--accent)'  },
                    ].map(s => (
                      <div className="admin-stat-card" key={s.label}>
                        <span className="admin-stat-icon">{s.icon}</span>
                        <span className="admin-stat-value" style={{ color: s.color }}>{s.value ?? '—'}</span>
                        <span className="admin-stat-label">{s.label}</span>
                      </div>
                    ))}
                  </div>
                  {data.stats.failure_rate != null && (
                    <div className="admin-info-row">
                      Failure rate: <strong style={{ color: 'var(--red)' }}>{data.stats.failure_rate}%</strong>
                      &nbsp;·&nbsp; Avg latency: <strong style={{ color: 'var(--green)' }}>{data.stats.avg_latency_ms}ms</strong>
                    </div>
                  )}
                </>
              ) : <div className="admin-loading">Loading stats…</div>}
            </div>
          )}

          {/* ── Documents ─────────────────────────────────────────────── */}
          {tab === 'Documents' && (
            <div className="admin-table-wrap">
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '12px' }}>
                <button
                  className="admin-del-btn"
                  style={{ background: 'var(--red)', color: 'white', padding: '6px 12px' }}
                  onClick={deleteAllDocs}
                  disabled={!data.docs || data.docs.length === 0}
                >
                  Delete All Documents
                </button>
              </div>
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Filename</th><th>Type</th><th>Chunks</th><th>Uploaded</th><th>Status</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {(data.docs || []).map(d => (
                    <tr key={d.id} className={d.is_deleted ? 'admin-row--deleted' : ''}>
                      <td className="admin-cell-main">
                        <span style={{ fontSize: 14 }}>
                          {d.file_type === 'pdf' ? '📕' : d.file_type === 'docx' ? '📘' : '📄'}
                          &nbsp;{d.filename}
                        </span>
                      </td>
                      <td className="admin-cell-dim">{d.file_type}</td>
                      <td>{d.chunk_count}</td>
                      <td className="admin-cell-dim">{d.upload_time?.slice(0, 10)}</td>
                      <td>
                        <span className={`admin-status ${d.is_deleted ? 'admin-status--del' : 'admin-status--ok'}`}>
                          {d.is_deleted ? 'Deleted' : 'Active'}
                        </span>
                      </td>
                      <td>
                        {!d.is_deleted && (
                          <button className="admin-del-btn" onClick={() => deleteDoc(d.id)}>Delete</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(!data.docs || data.docs.length === 0) &&
                <div className="admin-empty">No documents found.</div>}
            </div>
          )}
        </div>

        {/* Toast */}
        {toast && <div className="admin-toast">{toast}</div>}
      </div>
    </div>
  )
}
