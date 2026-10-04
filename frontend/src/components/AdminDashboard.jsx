/**
 * AdminDashboard.jsx — "Stats & maintenance", opened from the library
 * footer: what is indexed, how questions went, and library upkeep
 * (re-index the server's test_docs/ folder, remove every document).
 *
 * /admin/stats mixes two scopes, and the labels say which is which:
 *   - in-memory counters (monitor/logger.py) cover only the time since the
 *     server started and reset on restart;
 *   - Postgres feedback_logs is all time. Its success_rate is the share of
 *     logged answers the OUTPUT VALIDATOR did not flag — not the reflection
 *     checks — so it is labelled as exactly that.
 */

import { useCallback, useEffect, useState } from 'react'
import { FileText, Gauge, RefreshCw, Wrench } from 'lucide-react'
import Modal from './Modal.jsx'
import { adminDeleteAllDocuments, adminDeleteDocument, adminGetDocuments, adminGetStats, apiReload } from '../api.js'
import { docBadge } from '../docs.js'

const TABS = ['Overview', 'Documents', 'Maintenance']
const TAB_ICON = { Documents: FileText, Maintenance: Wrench }
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
  const [confirming, setConfirming] = useState(null)
  const [showRemoved, setShowRemoved] = useState(false)
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
    </div>
  )
}

/** What the last re-index did, from POST /reload's own counts and log. */
function ReloadResult({ r }) {
  if (r.status === 'no_docs') {
    return <p className="mt-res" role="status">The server had no test_docs/ folder. It has been created, and is empty.</p>
  }
  const problems = (r.log || []).filter(e => e.status === 'error' || e.status === 'limit')
  return (
    <div className="mt-res" role="status">
      <p>
        {r.loaded ? <><b>{plural(r.loaded, 'new file')}</b> indexed</> : 'No new files'}
        {r.skipped > 0 && <>, {n(r.skipped)} already in the library</>}
        {r.failed > 0 && <>, <b className="mt-bad">{n(r.failed)} failed</b></>}.
      </p>
      {problems.length > 0 && (
        <ul>{problems.map(e => <li key={e.file}><b>{e.file}</b> {e.msg || 'the library is full'}</li>)}</ul>
      )}
    </div>
  )
}

function Maintenance({ docCount, onChanged, toast }) {
  const [reloading, setReloading] = useState(false)
  const [result, setResult] = useState(null)
  const [confirming, setConfirming] = useState(false)
  const [clearing, setClearing] = useState(false)

  const reindex = async () => {
    setReloading(true)
    setResult(null)
    try {
      const r = await apiReload()
      setResult(r)
      if (r.loaded) onChanged()
    } catch (e) {
      toast?.(`Re-index failed: ${e.message}`, 'err')
    } finally {
      setReloading(false)
    }
  }

  const removeAll = async () => {
    setConfirming(false)
    setClearing(true)
    try {
      const res = await adminDeleteAllDocuments()
      toast?.(`Library cleared: ${plural(res.count, 'document')} removed`, 'info')
      onChanged()
    } catch (e) {
      toast?.(`Couldn't clear the library: ${e.message}`, 'err')
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="mt">
      <div className="mt-row">
        <div>
          <b>Re-index the server folder</b>
          <span>Adds files placed in <code>test_docs/</code> on the server. Files already in the library are skipped.</span>
        </div>
        <button type="button" className="dt-del mt-go" onClick={reindex} disabled={reloading}>
          <RefreshCw size={14} strokeWidth={1.75} className={`ico${reloading ? ' spin' : ''}`} aria-hidden="true" />
          {reloading ? 'Indexing…' : 'Re-index'}
        </button>
      </div>
      {result && <ReloadResult r={result} />}

      <div className="mt-row danger">
        <div>
          <b>Remove every document</b>
          <span>Empties the library and stops any OCR in progress. This can&rsquo;t be undone.</span>
        </div>
        {confirming ? (
          <span className="doc-confirm" role="group" aria-label="Remove every document?">
            <button type="button" className="doc-confirm-yes" onClick={removeAll}>Remove all</button>
            <button type="button" onClick={() => setConfirming(false)}>Keep</button>
          </span>
        ) : (
          <button type="button" className="dt-del" onClick={() => setConfirming(true)} disabled={clearing || !docCount}>
            {clearing ? 'Removing…' : 'Remove all'}
          </button>
        )}
      </div>
    </div>
  )
}

export default function AdminDashboard({ onClose, onDocsChanged, toast, docCount = 0 }) {
  const [tab, setTab] = useState('Overview')
  const [stats, setStats] = useState(null)
  const [docs, setDocs] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const fetchTab = useCallback(async t => {
    if (t === 'Maintenance') return
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

  // Changes made here bypass the library, so refresh it as well as this
  // dialog's own data (the other tabs refetch when opened).
  const onChanged = () => { setDocs(null); setStats(null); fetchTab(tab); onDocsChanged?.() }

  const data = tab === 'Overview' ? stats : docs
  return (
    <Modal title="Stats & maintenance" sub="What is indexed, how questions went, and library upkeep" icon={Gauge} wide onClose={onClose} className="adm">
      <div className="adm-bar">
        <div className="seg" role="tablist" aria-label="Stats and maintenance">
          {TABS.map(t => {
            const Icon = TAB_ICON[t]
            return (
              <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
                {Icon && <Icon size={14} strokeWidth={1.75} className="ico" aria-hidden="true" />}{t}
              </button>
            )
          })}
        </div>
        {tab !== 'Maintenance' && (
          <button type="button" className="ibtn" onClick={() => fetchTab(tab)} disabled={busy} aria-label="Refresh">
            <RefreshCw size={16} strokeWidth={1.75} className={`ico${busy ? ' spin' : ''}`} aria-hidden="true" />
          </button>
        )}
      </div>

      <div role="tabpanel" aria-busy={busy || undefined}>
        {tab === 'Maintenance' ? (
          <Maintenance docCount={docCount} onChanged={onChanged} toast={toast} />
        ) : (
          <>
            {error && <p className="st-err" role="alert">Couldn&rsquo;t load {tab.toLowerCase()}: {error}</p>}
            {!data && !error && <p className="st-empty">Loading…</p>}
            {data && tab === 'Overview' && <Overview s={data} />}
            {data && tab === 'Documents' && <Documents docs={data} onDeleted={onChanged} toast={toast} />}
          </>
        )}
      </div>
    </Modal>
  )
}
