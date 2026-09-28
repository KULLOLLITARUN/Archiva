/**
 * api.js — Archiva v4 API client.
 * Auth fully removed — all requests are open, no tokens needed.
 */

// Use relative URLs so all requests go through Vite's dev proxy (avoids CORS).
const BASE = import.meta.env.VITE_API_URL || ''

// Safe JSON parse — returns null if body is empty or not JSON
async function safeJson(res) {
  try { return await res.json() } catch { return null }
}

// ── Backend readiness probe ────────────────────────────────────────────────────

/**
 * Poll /health until the backend responds or maxWaitMs elapses.
 * Returns true if the backend came up, false if we timed out.
 */
export async function waitForBackend(maxWaitMs = 60_000, intervalMs = 1_500) {
  const deadline = Date.now() + maxWaitMs
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(1_000) })
      if (res.ok) return true
    } catch {
      // ECONNREFUSED or timeout — backend not ready yet, keep polling
    }
    await new Promise(r => setTimeout(r, intervalMs))
  }
  return false
}

// ── Chat SSE stream ────────────────────────────────────────────────────────────

export async function streamChat({ message, session_id, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const reader  = res.body.getReader()
    const decoder = new TextDecoder()
    let   buffer  = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() ?? ''
      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data:')) continue
        try {
          const payload = JSON.parse(line.slice(5).trim())
          if (payload.done) { onDone(payload); return }
          if (payload.token) onToken(payload.token)
        } catch {}
      }
    }
  } catch (err) {
    onError(err.message)
  }
}

// ── File management ────────────────────────────────────────────────────────────

export async function apiUpload(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const e = await safeJson(res)
    throw new Error(e?.detail || e?.message || `Upload failed (${res.status})`)
  }
  return res.json()
}


export async function apiDeleteFile(fileId) {
  const res = await fetch(`${BASE}/files/${fileId}`, { method: 'DELETE' })
  if (!res.ok) { const e = await safeJson(res); throw new Error(e?.detail || 'Delete failed') }
  return res.json()
}

export async function apiClearAllDocs() {
  const res = await fetch(`${BASE}/documents/clear-all`, { method: 'DELETE' })
  if (!res.ok) { const e = await safeJson(res); throw new Error(e?.detail || 'Clear failed') }
  return res.json()
}

export async function apiGetFiles() {
  const res = await fetch(`${BASE}/docs-loaded`)
  if (!res.ok) throw new Error('Failed to load files')
  return res.json()
}

export async function apiReload() {
  const res = await fetch(`${BASE}/reload`, { method: 'POST' })
  if (!res.ok) throw new Error('Reload failed')
  return res.json()
}

// ── Suggestions ────────────────────────────────────────────────────────────────

export async function apiGetSuggestions() {
  const res = await fetch(`${BASE}/suggestions`)
  if (!res.ok) return { topics: [], generated: false }
  const data = await res.json()
  return { topics: data.topics || [], generated: data.generated || false }
}

// ── Admin ──────────────────────────────────────────────────────────────────────

export async function adminGetDocuments() {
  const res = await fetch(`${BASE}/admin/documents`)
  if (!res.ok) throw new Error('Failed to fetch documents')
  return res.json()
}
export async function adminGetStats() {
  const res = await fetch(`${BASE}/admin/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}
export async function adminDeleteDocument(id) {
  const res = await fetch(`${BASE}/admin/documents/${id}`, { method: 'DELETE' })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Failed') }
  return res.json()
}
export async function adminDeleteAllDocuments() {
  const res = await fetch(`${BASE}/admin/bulk-delete/documents`, { method: 'DELETE' })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Failed') }
  return res.json()
}

// ── Export ────────────────────────────────────────────────────────────────────

export async function apiExportConversation(sessionId, format = 'markdown') {
  const res = await fetch(`${BASE}/conversations/${sessionId}/export?format=${format}`)
  if (!res.ok) {
    const e = await safeJson(res)
    throw new Error(e?.detail || `Export failed (${res.status})`)
  }
  const blob = await res.blob()
  const ext = format === 'pdf' ? 'pdf' : 'md'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `conversation_${sessionId}.${ext}`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export async function apiHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Server offline')
  return res.json()
}
