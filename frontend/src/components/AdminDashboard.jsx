/**
 * AdminDashboard.jsx — "Pipeline stats", opened from the library footer:
 * what is indexed, and how questions went.
 *
 * /admin/stats mixes two scopes, and the labels say which is which:
 *   - in-memory counters (monitor/logger.py) cover only the time since the
 *     server started and reset on restart;
 *   - Postgres feedback_logs is all time. Its success_rate is the share of
 *     logged answers the OUTPUT VALIDATOR did not flag — not the reflection
 *     checks — so it is labelled as exactly that.
 */

import { useCallback, useEffect, useState } from 'react'
import { FileText, Gauge, RefreshCw } from 'lucide-react'
import Modal from './Modal.jsx'
import { adminDeleteAllDocuments, adminDeleteDocument, adminGetDocuments, adminGetStats } from '../api.js'
import { docBadge } from '../docs.js'

const TABS = ['Overview', 'Documents']
const n = v => (typeof v === 'number' ? v.toLocaleString() : '—')
const plural = (k, word) => `${n(k)} ${word}${k === 1 ? '' : 's'}`

function Tile({ label, value, note }) {
  return (
    <div className="st-tile">
      <span className="st-l">{label}</span>
      <b className="tabular">{value}</b>
      {note && <span className="st-note">{note}</span>}
    </div>
  )
}

/** Share of questions settled on attempt 1, 2 and 3: one bar, labelled. */
function Attempts({ counts }) {
  const parts = [
    { key: '1', label: '1st attempt', cls: 'a1' },
    { key: '2', label: '2nd attempt', cls: 'a2' },
    { key: '3', label: '3rd attempt', cls: 'a3' },
  ].map(p => ({ ...p, v: counts?.[p.key] || 0 }))
  const total = parts.reduce((s, p) => s + p.v, 0)
  if (!total) return null
  return (
    <div className="st-attempts">
      <div className="st-bar" role="img"
        aria-label={parts.map(p => `${p.label}: ${p.v}`).join(', ')}>
        {parts.filter(p => p.v).map(p => (
          <i key={p.key} className={p.cls} style={{ flexGrow: p.v }} title={`${p.label}: ${p.v}`} />
        ))}
      </div>
      <ul className="st-legend">
        {parts.map(p => (
          <li key={p.key}><i className={p.cls} aria-hidden="true" />{p.label}
            <b className="tabular">{p.v}</b><span className="tabular">{Math.round((p.v / total) * 100)}%</span></li>
        ))}
      </ul>
    </div>
  )
}

function Overview({ s }) {
  const ref = s.reflection_stats || {}
  const asked = s.total_queries || 0
  const usage = s.model_usage || {}
  return (
    <div className="st">
      <section>
        <h3 className="st-h">Library</h3>
        <div className="st-grid">
          <Tile label="Documents" value={n(s.store_files)} />
          <Tile label="Passages" value={n(s.store_chunks)} />
        </div>
      </section>

      <section>
        <h3 className="st-h">Since the server started <span>resets on restart</span></h3>
        <div className="st-grid">
          <Tile label="Questions" value={n(asked)} />
          <Tile label="Average response" value={s.latency_samples ? `${(s.avg_latency_ms / 1000).toFixed(1)} s` : '—'}
            note={s.latency_samples ? `last ${plural(s.latency_samples, 'answer')}` : null} />
          <Tile label="Average confidence" value={asked ? `${Math.round((ref.avg_confidence || 0) * 100)}%` : '—'} />
          <Tile label="Blocked by safety" value={n(s.blocked_queries)} />
          <Tile label="Flagged by validator" value={n(s.flagged_responses)} />
        </div>
        {asked > 0 && (
          <>
            <h4 className="st-sub">Attempts per question</h4>
            <Attempts counts={ref.accepted_attempt} />
            <h4 className="st-sub">Model used</h4>
            <ul className="st-list">
              <li>Fast model<b className="tabular">{n(usage.fast)}</b></li>
              <li>Strong model<b className="tabular">{n(usage.strong)}</b></li>
              {/* logger.py counts anything that isn't the fast or strong model here,
                  including no call at all (a refusal before generation). */}
              <li>Other or no model<b className="tabular">{n(usage.none)}</b></li>
            </ul>
          </>
        )}
      </section>

      <section>
        <h3 className="st-h">All time</h3>
        <div className="st-grid">
          <Tile label="Answers logged" value={n(s.answers_logged)} />
          <Tile label="Not flagged by validator" value={s.answers_logged ? `${s.success_rate}%` : '—'}
            note="share of logged answers" />
        </div>
      </section>
    </div>
  )
}

function DocRow({ d, confirming, onAsk, onCancel, onDelete }) {
  const badge = docBadge(d)
  const state = d.is_deleted ? 'Removed' : d.status === 'ready' ? 'Active' : d.status === 'processing' ? 'Reading' : 'Failed'
  return (
    <tr className={d.is_deleted ? 'is-del' : ''}>
      <td className="dt-name">
        <div>
          <span className={`ftype ft-${badge.type}`} aria-hidden="true">{badge.label}</span>
          <span title={d.filename}>{d.filename}</span>
        </div>
      </td>
      <td className="tabular" data-l="Passages">{n(d.chunk_count)}</td>
      <td className="tabular" data-l="Uploaded">{d.upload_time?.slice(0, 10) || '—'}</td>
      <td data-l="Status"><span className={`dt-state dt-${state.toLowerCase()}`}>{state}</span></td>
      <td className="dt-act">
        {!d.is_deleted && (confirming ? (
          <span className="doc-confirm" role="group" aria-label={`Remove ${d.filename}?`}>
            <button type="button" className="doc-confirm-yes" onClick={onDelete}>Remove</button>
            <button type="button" onClick={onCancel}>Keep</button>
          </span>
        ) : (
          <button type="button" className="dt-del" onClick={onAsk} aria-label={`Remove ${d.filename}`}>Remove</button>
        ))}
      </td>
    </tr>
  )
}

function Documents({ docs, onDeleted, toast }) {
  const [confirming, setConfirming] = useState(null)   // a doc id, or 'all'
  const [showRemoved, setShowRemoved] = useState(false)
  const [clearing, setClearing] = useState(false)
  const removed = docs.filter(d => d.is_deleted).length
  const shown = showRemoved ? docs : docs.filter(d => !d.is_deleted)

  const remove = async d => {
    setConfirming(null)
    try {
      await adminDeleteDocument(d.id)
      toast?.(`${d.filename} removed`, 'info')
      onDeleted()
    } catch (e) {
      toast?.(`Couldn't remove ${d.filename}: ${e.message}`, 'err')
    }
  }

  const removeAll = async () => {
    setConfirming(null)
    setClearing(true)
    try {
      // Not res.count: the endpoint counts rows already removed earlier too.
      await adminDeleteAllDocuments()
      toast?.('Library cleared', 'info')
      onDeleted()
    } catch (e) {
      toast?.(`Couldn't clear the library: ${e.message}`, 'err')
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="dt-wrap">
      <div className="dt-bar">
        <span>{plural(docs.length - removed, 'active document')}</span>
        {removed > 0 && (
          <label className="dt-toggle">
            <input type="checkbox" checked={showRemoved} onChange={e => setShowRemoved(e.target.checked)} />
            Show {n(removed)} removed
          </label>
        )}
      </div>
      {shown.length ? (
        <table className="dt">
          <thead><tr><th>Document</th><th>Passages</th><th>Uploaded</th><th>Status</th><th><span className="sr">Actions</span></th></tr></thead>
          <tbody>
            {shown.map(d => (
              <DocRow key={d.id} d={d} confirming={confirming === d.id}
                onAsk={() => setConfirming(d.id)} onCancel={() => setConfirming(null)} onDelete={() => remove(d)} />
            ))}
          </tbody>
        </table>
      ) : <p className="st-empty">No documents yet.</p>}

      {docs.length - removed > 0 && (
        <div className="danger">
          <div>
            <b>Remove every document</b>
            <span>Empties the library and stops any OCR in progress. This can&rsquo;t be undone.</span>
          </div>
          {confirming === 'all' ? (
            <span className="doc-confirm" role="group" aria-label="Remove every document?">
              <button type="button" className="doc-confirm-yes" onClick={removeAll}>Remove all</button>
              <button type="button" onClick={() => setConfirming(null)}>Keep</button>
            </span>
          ) : (
            <button type="button" className="dt-del" onClick={() => setConfirming('all')} disabled={clearing}>
              {clearing ? 'Removing…' : 'Remove all'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default function AdminDashboard({ onClose, onDocsChanged, toast }) {
  const [tab, setTab] = useState('Overview')
  const [stats, setStats] = useState(null)
  const [docs, setDocs] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const fetchTab = useCallback(async t => {
    setBusy(true)
    setError('')
    try {
      if (t === 'Overview') setStats(await adminGetStats())
      else setDocs((await adminGetDocuments()).documents || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => { fetchTab(tab) }, [tab, fetchTab])

  // A removal here bypasses the library, so refresh it as well as this list.
  const onDeleted = () => { fetchTab('Documents'); onDocsChanged?.() }

  const data = tab === 'Overview' ? stats : docs
  return (
    <Modal title="Pipeline stats" sub="What is indexed, and how questions went" icon={Gauge} wide onClose={onClose} className="adm">
      <div className="adm-bar">
        <div className="seg" role="tablist" aria-label="Pipeline stats">
          {TABS.map(t => (
            <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
              {t === 'Documents' && <FileText size={14} strokeWidth={1.75} className="ico" aria-hidden="true" />}{t}
            </button>
          ))}
        </div>
        <button type="button" className="ibtn" onClick={() => fetchTab(tab)} disabled={busy} aria-label="Refresh">
          <RefreshCw size={16} strokeWidth={1.75} className={`ico${busy ? ' spin' : ''}`} aria-hidden="true" />
        </button>
      </div>

      <div role="tabpanel" aria-busy={busy || undefined}>
        {error && <p className="st-err" role="alert">Couldn&rsquo;t load {tab.toLowerCase()}: {error}</p>}
        {!data && !error && <p className="st-empty">Loading…</p>}
        {data && tab === 'Overview' && <Overview s={data} />}
        {data && tab === 'Documents' && <Documents docs={data} onDeleted={onDeleted} toast={toast} />}
      </div>
    </Modal>
  )
}
