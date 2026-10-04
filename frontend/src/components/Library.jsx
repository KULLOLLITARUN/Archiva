/**
 * Library.jsx — the documents in the sidebar: upload (drop or browse),
 * one row per document with its type, passage count or OCR progress, and
 * an inline two-step remove.
 *
 * Scans are OCR'd in the background; App polls /docs-loaded while any are
 * processing, so the progress here moves without this component polling.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'
import { stagger } from 'animejs'
import { apiDeleteFile, apiUpload } from '../api.js'
import { ACCEPT_ATTR, docBadge, docStatus, isAccepted } from '../docs.js'
import { go } from '../motion.js'

const ICON = { size: 16, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

function DocRow({ f, confirming, removing, onAskRemove, onCancel, onRemove }) {
  const badge = docBadge(f)
  const status = docStatus(f)
  const busy = f.status === 'processing'
  const removeLabel = `${busy ? 'Cancel and remove' : 'Remove'} ${f.filename}`

  return (
    <div className={`doc doc--${status.tone}${removing ? ' doc--removing' : ''}`} aria-busy={busy || removing || undefined}>
      <span className={`ftype ft-${badge.type}`} aria-hidden="true">{badge.label}</span>
      <div className="doc-m">
        <div className="doc-n" title={f.filename}>{f.filename}</div>
        {confirming ? (
          <div className="doc-confirm" role="group" aria-label={`Remove ${f.filename}?`}>
            <span>{busy ? 'Stop OCR and remove?' : 'Remove from library?'}</span>
            <button type="button" className="doc-confirm-yes" onClick={onRemove}>Remove</button>
            <button type="button" onClick={onCancel}>Keep</button>
          </div>
        ) : (
          <>
            <div className="doc-s" title={status.tone === 'bad' ? status.text : undefined}>{status.text}</div>
            {busy && (
              <div className={`progress${status.progress == null ? ' progress--wait' : ''}`}
                role="progressbar" aria-label={`OCR progress for ${f.filename}`}
                aria-valuemin={0} aria-valuemax={100}
                aria-valuenow={status.progress == null ? undefined : Math.round(status.progress * 100)}>
                <i style={status.progress == null ? undefined : { width: `${Math.max(4, status.progress * 100)}%` }} />
              </div>
            )}
          </>
        )}
      </div>
      {!confirming && (
        <button type="button" className="doc-x" onClick={onAskRemove} disabled={removing}
          aria-label={removeLabel} title={removeLabel}>
          <X {...ICON} size={14} />
        </button>
      )}
    </div>
  )
}

export default function Library({ docsInfo, onDocsChanged, toast, uploadRef }) {
  const { files = [] } = docsInfo
  const [uploads, setUploads] = useState([])          // files being sent: { key, name }
  const [confirming, setConfirming] = useState(null)  // file_id awaiting "Remove?"
  const [removing, setRemoving] = useState(() => new Set())
  const [over, setOver] = useState(false)
  const inputRef = useRef(null)
  const listRef = useRef(null)
  const seen = useRef(new Set())
  const dragDepth = useRef(0)

  // Lets Home's "Add your first document" open this picker.
  useEffect(() => {
    if (!uploadRef) return undefined
    uploadRef.current = () => inputRef.current?.click()
    return () => { uploadRef.current = null }
  }, [uploadRef])

  // Rows rise in as they first appear (initial load, a finished upload);
  // rows already on screen don't replay when /docs-loaded is polled.
  useLayoutEffect(() => {
    const fresh = [...listRef.current.querySelectorAll('li[data-id]')].filter(li => !seen.current.has(li.dataset.id))
    fresh.forEach(li => seen.current.add(li.dataset.id))
    if (fresh.length) go(fresh, { opacity: [0, 1], translateX: [-8, 0], delay: stagger(40), duration: 500, ease: 'outExpo' })
  })

  const upload = useCallback(async (file) => {
    if (!isAccepted(file.name)) {
      toast(`${file.name}: unsupported type`, 'err')
      return
    }
    const key = `${file.name}-${Date.now()}-${Math.random()}`
    setUploads(u => [...u, { key, name: file.name }])
    try {
      const res = await apiUpload(file)
      if (res.status === 'ok') toast(`${file.name}: ${res.chunk_count} passage${res.chunk_count === 1 ? '' : 's'} indexed`, 'ok')
      else if (res.status === 'processing') toast(`${file.name}: scanned, reading it with OCR`, 'info')
      else if (res.status === 'duplicate') toast(`${file.name} is already in the library`, 'info')
      else toast(`${file.name}: ${res.message}`, 'err')
      if (res.status === 'ok' || res.status === 'processing') await onDocsChanged()
    } catch (err) {
      toast(`${file.name}: ${err.message}`, 'err')
    } finally {
      setUploads(u => u.filter(x => x.key !== key))
    }
  }, [toast, onDocsChanged])

  const uploadAll = useCallback(list => { Array.from(list).forEach(upload) }, [upload])

  const remove = useCallback(async (f) => {
    setConfirming(null)
    setRemoving(s => new Set(s).add(f.file_id))
    try {
      await apiDeleteFile(f.file_id)
      toast(`${f.filename} removed`, 'info')
      await onDocsChanged()
    } catch (err) {
      toast(`Couldn't remove ${f.filename}: ${err.message}`, 'err')
    } finally {
      setRemoving(s => { const n = new Set(s); n.delete(f.file_id); return n })
    }
  }, [toast, onDocsChanged])

  // dragenter/leave fire for every child crossed; count depth so the
  // highlight doesn't flicker while the pointer moves over the rows.
  const drag = {
    onDragEnter: e => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); dragDepth.current++; setOver(true) } },
    onDragOver: e => { if (e.dataTransfer?.types?.includes('Files')) e.preventDefault() },
    onDragLeave: () => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setOver(false) },
    onDrop: e => { e.preventDefault(); dragDepth.current = 0; setOver(false); uploadAll(e.dataTransfer.files) },
  }

  return (
    <div className={`lib-wrap${over ? ' over' : ''}`} {...drag}>
      <button type="button" className={`drop${over ? ' over' : ''}`} onClick={() => inputRef.current?.click()}>
        <span className="drop-ico"><Upload {...ICON} /></span>
        <span>
          <span className="drop-fine">Drop files or <b>browse</b></span>
          <span className="drop-touch"><b>Add documents</b></span>
          <small>PDF · DOCX · TXT · MD · CSV · HTML · scans</small>
        </span>
      </button>
      <input ref={inputRef} type="file" multiple accept={ACCEPT_ATTR} hidden
        onChange={e => { uploadAll(e.target.files); e.target.value = '' }} />

      <ul className="lib" ref={listRef} aria-label="Documents">
        {uploads.map(u => (
          <li key={u.key}>
            <div className="doc doc--busy" aria-busy="true">
              <span className="ftype ft-txt" aria-hidden="true">···</span>
              <div className="doc-m">
                <div className="doc-n" title={u.name}>{u.name}</div>
                <div className="doc-s">Uploading…</div>
                <div className="progress progress--wait"><i /></div>
              </div>
            </div>
          </li>
        ))}
        {files.map(f => (
          <li key={f.file_id || f.filename} data-id={f.file_id || f.filename}>
            <DocRow f={f}
              confirming={confirming === f.file_id}
              removing={removing.has(f.file_id)}
              onAskRemove={() => setConfirming(f.file_id)}
              onCancel={() => setConfirming(null)}
              onRemove={() => remove(f)} />
          </li>
        ))}
        {!files.length && !uploads.length && (
          <li className="lib-empty">No documents yet. Add one to start asking questions.</li>
        )}
      </ul>
    </div>
  )
}
