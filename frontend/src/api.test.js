import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiGetSuggestions, apiUpload, streamChat } from './api.js'

afterEach(() => {
  vi.unstubAllGlobals()
})

// ── Helpers to build a fake fetch Response backed by a controllable SSE stream ──

function sseChunk(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`
}

/** A fake ReadableStream reader that yields the given string chunks, then done. */
function fakeReader(chunks) {
  const encoder = new TextEncoder()
  let i = 0
  return {
    read: vi.fn(async () => {
      if (i < chunks.length) {
        return { value: encoder.encode(chunks[i++]), done: false }
      }
      return { value: undefined, done: true }
    }),
  }
}

function fakeStreamResponse(chunks, { ok = true } = {}) {
  return {
    ok,
    json: async () => ({}),
    body: { getReader: () => fakeReader(chunks) },
  }
}

// ── streamChat ──────────────────────────────────────────────────────────────────

describe('streamChat', () => {
  it('emits tokens via onToken and calls onDone with the final payload', async () => {
    const chunks = [
      sseChunk({ token: 'Hel', done: false }),
      sseChunk({ token: 'lo', done: false }),
      sseChunk({ token: '', done: true, sources: [], model_used: 'x' }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => fakeStreamResponse(chunks)))

    const tokens = []
    let doneCalled = false
    let donePayload = null

    await streamChat({
      message: 'hi', session_id: 's1',
      onToken: t => tokens.push(t),
      onDone: p => { doneCalled = true; donePayload = p },
      onError: () => { throw new Error('onError should not be called') },
    })

    expect(tokens.join('')).toBe('Hello')
    expect(doneCalled).toBe(true)
    expect(donePayload.model_used).toBe('x')
  })

  it('reassembles an SSE frame split across two reader chunks', async () => {
    const full = sseChunk({ token: 'whole-token', done: false })
    const splitPoint = Math.floor(full.length / 2)
    const chunks = [
      full.slice(0, splitPoint),
      full.slice(splitPoint) + sseChunk({ token: '', done: true }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => fakeStreamResponse(chunks)))

    const tokens = []
    await streamChat({
      message: 'hi', session_id: 's1',
      onToken: t => tokens.push(t),
      onDone: () => {},
      onError: () => { throw new Error('onError should not be called') },
    })

    expect(tokens.join('')).toBe('whole-token')
  })

  it('silently skips a malformed JSON line instead of crashing the stream', async () => {
    const chunks = [
      'data: {not valid json\n\n',
      sseChunk({ token: 'ok', done: false }),
      sseChunk({ token: '', done: true }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => fakeStreamResponse(chunks)))

    const tokens = []
    await streamChat({
      message: 'hi', session_id: 's1',
      onToken: t => tokens.push(t),
      onDone: () => {},
      onError: () => { throw new Error('onError should not be called') },
    })

    expect(tokens).toEqual(['ok'])
  })

  it('calls onError when the HTTP response is not ok', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      json: async () => ({ detail: 'boom' }),
    })))

    let errorMsg = null
    await streamChat({
      message: 'hi', session_id: 's1',
      onToken: () => {},
      onDone: () => { throw new Error('onDone should not be called') },
      onError: msg => { errorMsg = msg },
    })

    expect(errorMsg).toBe('boom')
  })
})

// ── apiGetSuggestions ──────────────────────────────────────────────────────────

describe('apiGetSuggestions', () => {
  it('returns topics/generated from a successful response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ topics: [{ label: 'Pricing' }], generated: true }),
    })))

    const result = await apiGetSuggestions()
    expect(result).toEqual({ topics: [{ label: 'Pricing' }], generated: true })
  })

  it('throws on a non-ok response instead of reporting no topics', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 429 })))

    await expect(apiGetSuggestions()).rejects.toThrow('Suggestions failed (429)')
  })
})

// ── apiUpload ──────────────────────────────────────────────────────────────────

describe('apiUpload', () => {
  it('throws the server-provided detail message on failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      json: async () => ({ detail: 'File too large' }),
    })))

    await expect(apiUpload(new File(['x'], 'a.txt'))).rejects.toThrow('File too large')
  })

  it('resolves with the parsed JSON body on success', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: 'ok', chunk_count: 3 }),
    })))

    const result = await apiUpload(new File(['x'], 'a.txt'))
    expect(result).toEqual({ status: 'ok', chunk_count: 3 })
  })
})
