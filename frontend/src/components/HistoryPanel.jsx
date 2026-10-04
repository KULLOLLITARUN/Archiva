/**
 * HistoryPanel.jsx — "Conversations", opened from the library footer:
 * recent sessions (the backend keeps each one's last 10 turns), reopened
 * with a click, removed with an inline confirm.
 */

import { useCallback, useEffect, useState } from 'react'
import { History, MessageSquare, X } from 'lucide-react'
import Modal from './Modal.jsx'
import { apiDeleteConversation, apiListConversations } from '../api.js'
import { timeAgo } from '../history.js'

const ICON = { size: 15, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

export default function HistoryPanel({ currentId, onOpen, onDeleted, onClose, toast }) {
  const [rows, setRows] = useState(null)
  const [error, setError] = useState('')
  const [confirming, setConfirming] = useState(null)
  const [opening, setOpening] = useState(null)

  const load = useCallback(() => {
    setError('')
    apiListConversations().then(setRows).catch(e => setError(e.message))
  }, [])
  useEffect(load, [load])

  const open = async id => {
    setOpening(id)
    try {
      await onOpen(id)
      onClose()
    } catch (e) {
      toast?.(e.message, 'err')
      setOpening(null)
    }
  }

  const remove = async r => {
    setConfirming(null)
    try {
      await apiDeleteConversation(r.session_id)
      setRows(list => list.filter(x => x.session_id !== r.session_id))
      onDeleted?.(r.session_id)
      toast?.('Conversation deleted', 'info')
    } catch (e) {
      toast?.(`Couldn't delete it: ${e.message}`, 'err')
    }
  }

  return (
    <Modal title="Conversations" sub="Pick up where you left off" icon={History} onClose={onClose} className="hist">
      {error && <p className="st-err" role="alert">Couldn&rsquo;t load conversations: {error}</p>}
      {!rows && !error && <p className="st-empty">Loading…</p>}
      {rows && !rows.length && (
        <p className="st-empty">No saved conversations yet. Each question you ask is kept here.</p>
      )}
      {rows?.length > 0 && (
        <>
          <ul className="hist-list">
            {rows.map(r => {
              const current = r.session_id === currentId
              return (
                <li key={r.session_id} className={`hist-row${current ? ' is-current' : ''}`} aria-busy={opening === r.session_id || undefined}>
                  <button type="button" className="hist-open" onClick={() => open(r.session_id)} disabled={opening !== null}
                    aria-current={current || undefined}>
                    <span className="hist-ico"><MessageSquare {...ICON} /></span>
                    <span className="hist-m">
                      <b>{r.title}</b>
                      <span>
                        {r.turns} question{r.turns === 1 ? '' : 's'} · {timeAgo(r.updated_at)}
                        {current && <em> · open now</em>}
                      </span>
                    </span>
                  </button>
                  {confirming === r.session_id ? (
                    <span className="doc-confirm" role="group" aria-label={`Delete "${r.title}"?`}>
                      <button type="button" className="doc-confirm-yes" onClick={() => remove(r)}>Delete</button>
                      <button type="button" onClick={() => setConfirming(null)}>Keep</button>
                    </span>
                  ) : (
                    <button type="button" className="doc-x hist-x" onClick={() => setConfirming(r.session_id)}
                      aria-label={`Delete "${r.title}"`} title="Delete conversation">
                      <X {...ICON} size={14} />
                    </button>
                  )}
                </li>
              )
            })}
          </ul>
          <p className="hist-note">Each conversation keeps its last 10 questions.</p>
        </>
      )}
    </Modal>
  )
}
