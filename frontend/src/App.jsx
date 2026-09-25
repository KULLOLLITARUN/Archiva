/**
 * App.jsx — Root application component.
 * Fully open — no auth, no login, no user state.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import Header          from './components/Header.jsx'
import DocsStrip       from './components/DocsStrip.jsx'
import ChatWindow      from './components/ChatWindow.jsx'
import InputBar        from './components/InputBar.jsx'
import UploadPanel     from './components/UploadPanel.jsx'
import PlaybookPanel   from './components/PlaybookPanel.jsx'
import AdminDashboard  from './components/AdminDashboard.jsx'
import {
  streamChat, apiGetFiles, apiGetSuggestions, waitForBackend,
} from './api.js'
import './styles.css'
import './auth-admin.css'


export default function App() {
  const [sessionId]                    = useState(() => crypto.randomUUID())
  const [messages,     setMessages]    = useState([])
  const [isLoading,    setIsLoading]   = useState(false)
  const [streamStatus, setStreamStatus]= useState('')
  const [showPanel,    setShowPanel]   = useState(false)
  const [showPlaybook, setShowPlaybook]= useState(false)
  const [showAdmin,    setShowAdmin]   = useState(false)
  const [docsInfo,     setDocsInfo]    = useState({ files: [], total_files: 0, total_chunks: 0 })
  const [dynTopics,    setDynTopics]   = useState([])
  const [backendReady, setBackendReady]= useState(false)
  const [backendStatus,setBackendStatus]=useState('Connecting to backend…')



  const cancelStreamRef = useRef(null)

  // ── Docs + dynamic suggestions ────────────────────────────────────────────

  const refreshDocs = useCallback(() => {
    apiGetFiles()
      .then(setDocsInfo)
      .catch(() => setDocsInfo({ files: [], total_files: 0, total_chunks: 0 }))
  }, [])

  const refreshSuggestions = useCallback(() => {
    apiGetSuggestions()
      .then(data => { if (data.generated) setDynTopics(data.topics) })
      .catch(() => {})
  }, [])

  // Wait for backend, then load initial data
  useEffect(() => {
    waitForBackend(60_000, 1_500).then(ready => {
      if (ready) {
        setBackendReady(true)
        setBackendStatus('')
        refreshDocs()
        refreshSuggestions()
      } else {
        setBackendStatus('Backend unavailable — reload to retry.')
      }
    })
  }, [refreshDocs, refreshSuggestions])
  useEffect(() => () => cancelStreamRef.current?.(), [])

  // ── Chat handler ──────────────────────────────────────────────────────────

  const handleSend = useCallback((text) => {
    if (!text.trim() || isLoading) return

    cancelStreamRef.current?.()
    cancelStreamRef.current = null

    const userMsg  = { id: crypto.randomUUID(), role: 'user', content: text }
    const botMsgId = crypto.randomUUID()

    setMessages(prev => [...prev, userMsg, {
      id: botMsgId, role: 'bot', content: '',
      sources: [], intent: 'qa', model_used: 'none',
      latency_ms: 0, flagged: false, streaming: true,
      attempts: 1, reflected: false,
      reflection_reason: 'not_reflected', confidence: 1.0,
      search_queries: [], failure_type: null,
      retrieval_latency_ms: 0, reranker_scores: [], tokens_used: 0,
    }])
    setIsLoading(true)
    setStreamStatus('Searching your documents…')

    streamChat({
      message: text,
      session_id: sessionId,
      onToken(token) {
        setStreamStatus('')
        setMessages(prev =>
          prev.map(m => m.id === botMsgId ? { ...m, content: m.content + token } : m)
        )
      },
      onDone(d) {
        setMessages(prev =>
          prev.map(m => m.id === botMsgId ? {
            ...m,
            sources:              d.sources              || [],
            intent:               d.intent               || 'qa',
            model_used:           d.model_used           || 'none',
            latency_ms:           d.latency_ms           || 0,
            flagged:              d.flagged              || false,
            streaming:            false,
            attempts:             d.attempts             ?? 1,
            reflected:            d.reflected            ?? false,
            reflection_reason:    d.reflection_reason    ?? 'not_reflected',
            confidence:           d.confidence           ?? 1.0,
            search_queries:       d.search_queries       ?? [],
            failure_type:         d.failure_type         ?? null,
            retrieval_latency_ms: d.retrieval_latency_ms ?? 0,
            reranker_scores:      d.reranker_scores      ?? [],
            tokens_used:          d.tokens_used          ?? 0,
          } : m)
        )
        setIsLoading(false)
        setStreamStatus('')
        cancelStreamRef.current = null
      },
      onError(err) {
        setMessages(prev =>
          prev.map(m => m.id === botMsgId
            ? { ...m, content: 'Something went wrong. Please try again.', streaming: false, isError: true }
            : m
          )
        )
        setIsLoading(false)
        setStreamStatus('')
        cancelStreamRef.current = null
      },
    })
  }, [sessionId, isLoading])

  const handleClearChat = useCallback(() => {
    cancelStreamRef.current?.()
    cancelStreamRef.current = null
    setMessages([])
    setIsLoading(false)
    setStreamStatus('')
  }, [])


  // ── Panel controls ────────────────────────────────────────────────────────

  const openPanel = () => { setShowPanel(true);   setShowPlaybook(false); setShowAdmin(false) }
  const openPlay  = () => { setShowPlaybook(true); setShowPanel(false);   setShowAdmin(false) }

  return (
    <div className="app">
      {/* Upload panel */}
      <UploadPanel
        isOpen={showPanel}
        docsInfo={docsInfo}
        onClose={() => setShowPanel(false)}
        onClearChat={handleClearChat}
        onDocsChanged={() => { refreshDocs(); refreshSuggestions() }}
      />

      {/* Playbook panel */}
      <PlaybookPanel
        isOpen={showPlaybook}
        onClose={() => setShowPlaybook(false)}
      />

      {/* Admin dashboard (modal) */}
      {showAdmin && (
        <AdminDashboard
          onClose={() => setShowAdmin(false)}
        />
      )}

      {/* Main chat area */}
      <div className={`main-pane${showPanel ? ' main-pane--panel-open' : ''}${showPlaybook ? ' main-pane--play-open' : ''}`}>
        <Header
          docsInfo={docsInfo}
          onUploadClick={openPanel}
          onPlaybookClick={openPlay}
          onAdminClick={() => setShowAdmin(true)}
          backendStatus={backendStatus}
        />

        <DocsStrip docsInfo={docsInfo} onManage={openPanel} />

        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          streamStatus={streamStatus}
          onSuggestionClick={handleSend}
          dynamicTopics={dynTopics}
          hasDocs={docsInfo.total_files > 0}
        />
        <InputBar
          onSend={handleSend}
          isLoading={isLoading || messages.some(m => m.streaming) || !backendReady}
        />
      </div>
    </div>
  )
}
