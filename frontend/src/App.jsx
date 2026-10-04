/**
 * App.jsx — Root application component.
 * Fully open — no auth, no login, no user state.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import Sidebar         from './components/Sidebar.jsx'
import TopBar          from './components/TopBar.jsx'
import EvidencePanel   from './components/EvidencePanel.jsx'
import Evidence        from './components/Evidence.jsx'
import { ToastStack, useToasts } from './components/Toasts.jsx'
import ChatWindow      from './components/ChatWindow.jsx'
import InputBar        from './components/InputBar.jsx'
import { useCiteLinking } from './useCiteLinking.js'
import PlaybookPanel   from './components/PlaybookPanel.jsx'
import AdminDashboard  from './components/AdminDashboard.jsx'
import HistoryPanel    from './components/HistoryPanel.jsx'
import {
  streamChat, apiGetFiles, apiGetSuggestions, waitForBackend, apiExportConversation, apiGetConversation,
} from './api.js'
import { turnsToMessages } from './history.js'
import { useTheme } from './theme.js'
import { useMedia, WIDE, NARROW } from './useMedia.js'
import './styles.css'
import './auth-admin.css'


const SUGGEST_RETRIES = 4
const SUGGEST_RETRY_MS = 20_000   // a few tries span the 1-minute rate-limit window

export default function App() {
  // The server keeps each session's turns (follow-ups read them, and they
  // can be reopened from Conversations), so a new conversation needs a new id.
  const [sessionId,    setSessionId]   = useState(() => crypto.randomUUID())
  const [messages,     setMessages]    = useState([])
  const [isLoading,    setIsLoading]   = useState(false)
  const [streamStatus, setStreamStatus]= useState('')
  const [showPlaybook, setShowPlaybook]= useState(false)
  const [showAdmin,    setShowAdmin]   = useState(false)
  const [showHistory,  setShowHistory] = useState(false)
  const [docsInfo,     setDocsInfo]    = useState({ files: [], total_files: 0, total_chunks: 0 })
  const [dynTopics,    setDynTopics]   = useState([])
  // False until /suggestions first answers, so Home can say "finding
  // topics" instead of implying the documents have none.
  const [topicsLoaded, setTopicsLoaded]= useState(false)
  const [backendReady, setBackendReady]= useState(false)
  const [backendStatus,setBackendStatus]=useState('Connecting to backend…')
  // The answer whose evidence is shown. Each new answer takes over when it
  // finishes, so the panel always matches the latest reply until the user
  // picks an earlier one.
  const [selectedId,   setSelectedId]  = useState(null)



  const cancelStreamRef = useRef(null)
  const [toasts, toast] = useToasts()

  // ── Docs + dynamic suggestions ────────────────────────────────────────────

  const refreshDocs = useCallback(() => {
    return apiGetFiles()
      .then(setDocsInfo)
      .catch(() => setDocsInfo({ files: [], total_files: 0, total_chunks: 0 }))
  }, [])

  // A failed request (usually 429 - /suggestions allows 10 a minute) keeps
  // the cards already shown and tries again; it used to clear them, so one
  // rate-limited load hid the starter cards until the page was reloaded.
  const suggestRetry = useRef({ timer: null, left: 0 })
  const refreshSuggestions = useCallback((retries = SUGGEST_RETRIES) => {
    clearTimeout(suggestRetry.current.timer)
    suggestRetry.current.left = retries
    apiGetSuggestions()
      // Always sync from the response, even when generated is false (e.g.
      // the store is now empty after a delete) — gating this on `generated`
      // meant deleting all documents left the old topic pills stuck on
      // screen forever, since there was never a fresh `true` response to
      // replace them with.
      .then(data => { setDynTopics(data.topics || []); setTopicsLoaded(true) })
      .catch(() => {
        const left = suggestRetry.current.left
        if (left > 0) suggestRetry.current.timer = setTimeout(() => refreshSuggestions(left - 1), SUGGEST_RETRY_MS)
        else setTopicsLoaded(true)
      })
  }, [])
  useEffect(() => () => clearTimeout(suggestRetry.current.timer), [])

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

  // Scanned PDFs are OCR'd in the background: poll while any document is
  // still processing, and refresh suggestions once the last one finishes
  // (it only becomes searchable at that point).
  const hasProcessing = (docsInfo.files || []).some(f => f.status === 'processing')
  const wasProcessingRef = useRef(false)
  useEffect(() => {
    if (!hasProcessing) return undefined
    const id = setInterval(refreshDocs, 3000)
    return () => clearInterval(id)
  }, [hasProcessing, refreshDocs])
  useEffect(() => {
    if (wasProcessingRef.current && !hasProcessing) refreshSuggestions()
    wasProcessingRef.current = hasProcessing
  }, [hasProcessing, refreshSuggestions])

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
        setSelectedId(botMsgId)
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
    setSelectedId(null)
    setIsLoading(false)
    setStreamStatus('')
    setSessionId(crypto.randomUUID())
  }, [])

  // Reopen a saved conversation and continue it: follow-ups go to the same
  // session, so the server's memory of earlier turns carries on.
  const openConversation = useCallback(async (id) => {
    const { turns } = await apiGetConversation(id)
    cancelStreamRef.current?.()
    cancelStreamRef.current = null
    const restored = turnsToMessages(turns)
    setMessages(restored)
    setSessionId(id)
    setSelectedId(restored.filter(m => m.role === 'bot').at(-1)?.id ?? null)
    setIsLoading(false)
    setStreamStatus('')
  }, [])


  // ── Layout: library menu (phones) and evidence column/drawer/sheet ───────

  const wide   = useMedia(WIDE)
  const narrow = useMedia(NARROW)
  const [theme, setTheme]      = useTheme()
  const [sideOpen, setSideOpen] = useState(false)
  const [evOpen,   setEvOpen]   = useState(false)
  const hasMessages = messages.length > 0

  // Desktop has room, so the evidence column opens with the first message;
  // on smaller screens it waits to be asked for, because there it covers
  // the conversation.
  useEffect(() => {
    if (!hasMessages) setEvOpen(false)
    else if (wide) setEvOpen(true)
  }, [hasMessages]) // eslint-disable-line react-hooks/exhaustive-deps

  // Crossing a breakpoint changes what "open" means (column vs drawer), so
  // reset instead of carrying a column straight into a covering drawer.
  useEffect(() => { setEvOpen(wide && hasMessages) }, [wide]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!narrow) setSideOpen(false) }, [narrow])

  const closeOverlays = useCallback(() => {
    setSideOpen(false)
    if (!wide) setEvOpen(false)
  }, [wide])

  const startNewChat = useCallback(() => {
    handleClearChat()
    setSideOpen(false)
  }, [handleClearChat])

  // Escape closes whichever overlay is open; "N" starts a new conversation
  // when the user isn't typing somewhere.
  useEffect(() => {
    const onKey = e => {
      if (e.key === 'Escape') { closeOverlays(); return }
      const typing = e.target.closest?.('input, textarea, select, [contenteditable="true"]')
      if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey && (e.key === 'n' || e.key === 'N')) {
        e.preventDefault()
        startNewChat()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [closeOverlays, startNewChat])

  // Resolves once the library list is fresh, so a row being removed stays
  // dimmed until it's actually gone rather than flickering back.
  const docsChanged = useCallback(() => {
    refreshSuggestions()
    return refreshDocs()
  }, [refreshDocs, refreshSuggestions])

  const openPlay  = () => { setShowPlaybook(true); setShowAdmin(false); setShowHistory(false); setSideOpen(false) }
  const openStats = () => { setShowAdmin(true); setShowPlaybook(false); setShowHistory(false); setSideOpen(false) }
  const openHist  = () => { setShowHistory(true); setShowPlaybook(false); setShowAdmin(false); setSideOpen(false) }

  // Clicking an answer always reveals its evidence, on every screen size.
  const selectAnswer = useCallback((id) => {
    setSelectedId(id)
    setEvOpen(true)
    setSideOpen(false)
  }, [])
  // A citation click selects its answer and points the panel at the passage.
  const [focus, setFocus] = useState(null)
  const citeAnswer = useCallback((id, ns) => {
    selectAnswer(id)
    setFocus({ id, ns, t: Date.now() })
  }, [selectAnswer])

  // "Add your first document" on Home: the library's file picker, after
  // opening the library on phones where it's a menu.
  const uploadRef = useRef(null)
  const addDocs = useCallback(() => {
    if (narrow) setSideOpen(true)
    uploadRef.current?.()
  }, [narrow])

  const selectedMsg = messages.find(m => m.id === selectedId && !m.streaming) || null
  const answerNo = selectedMsg ? messages.filter(m => m.role === 'bot').indexOf(selectedMsg) + 1 : 0

  const inputDisabled = isLoading || messages.some(m => m.streaming) || !backendReady
  useCiteLinking(selectedMsg?.id)

  const scrimOn = (narrow && sideOpen) || (!wide && evOpen)
  const total   = docsInfo.total_files || 0
  const docsLabel = `${total === 1 ? 'the' : `all ${total}`} document${total === 1 ? '' : 's'}`
  const questions = messages.filter(m => m.role === 'user')
  const title    = questions[0]?.content || 'New conversation'
  const subtitle = hasMessages
    ? `${questions.length} question${questions.length === 1 ? '' : 's'} · ${docsLabel}`
    : total ? `Searching ${docsLabel}` : 'No documents yet'

  return (
    <>
      <div className={`shell${evOpen ? '' : ' no-ev'}`}>
        <Sidebar
          docsInfo={docsInfo}
          open={sideOpen}
          inert={narrow && !sideOpen}
          onClose={() => setSideOpen(false)}
          onNewChat={startNewChat}
          onDocsChanged={docsChanged}
          toast={toast}
          uploadRef={uploadRef}
          onPlaybook={openPlay}
          onStats={openStats}
          onHistory={openHist}
          theme={theme}
          onTheme={setTheme}
        />

        <main className="center">
          <TopBar
            title={title}
            subtitle={subtitle}
            docsInfo={docsInfo}
            backendStatus={backendStatus}
            hasMessages={hasMessages}
            evOpen={evOpen}
            onMenu={() => setSideOpen(true)}
            onToggleEvidence={() => setEvOpen(o => !o)}
            onExport={format => apiExportConversation(sessionId, format).catch(err => alert(err.message))}
          />

          <ChatWindow
            messages={messages}
            streamStatus={streamStatus}
            onSend={handleSend}
            inputDisabled={inputDisabled}
            topics={dynTopics}
            topicsLoaded={topicsLoaded}
            docCount={total}
            selectedId={selectedMsg?.id}
            onSelect={selectAnswer}
            onCite={citeAnswer}
            onAddDocs={addDocs}
          />

          {hasMessages && (
            <div className="dock">
              <InputBar id="ask-dock" onSend={handleSend} isLoading={inputDisabled} placeholder="Ask a follow-up…" />
            </div>
          )}
        </main>

        <EvidencePanel open={evOpen} inert={!evOpen} onClose={() => setEvOpen(false)}
          label={answerNo ? `· answer ${answerNo}` : ''}>
          {selectedMsg && <Evidence key={selectedMsg.id} message={selectedMsg}
            focus={focus?.id === selectedMsg.id ? focus : null} />}
        </EvidencePanel>
      </div>
      <div className={`scrim${scrimOn ? ' on' : ''}`} onClick={closeOverlays} aria-hidden="true" />

      <ToastStack toasts={toasts} />
      {showPlaybook && <PlaybookPanel onClose={() => setShowPlaybook(false)} />}
      {showHistory && (
        <HistoryPanel
          currentId={hasMessages ? sessionId : null}
          onOpen={openConversation}
          onDeleted={id => { if (id === sessionId) handleClearChat() }}
          onClose={() => setShowHistory(false)}
          toast={toast}
        />
      )}
      {showAdmin && (
        <AdminDashboard
          onClose={() => setShowAdmin(false)}
          onDocsChanged={docsChanged}
          toast={toast}
          docCount={total}
        />
      )}
    </>
  )
}
