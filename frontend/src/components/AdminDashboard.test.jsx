import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import AdminDashboard from './AdminDashboard.jsx'
import * as api from '../api.js'

vi.mock('../api.js', () => ({
  adminGetStats: vi.fn(),
  adminGetDocuments: vi.fn(),
  adminDeleteDocument: vi.fn(() => Promise.resolve({ deleted: true })),
  adminDeleteAllDocuments: vi.fn(() => Promise.resolve({ deleted: true, count: 4 })),
  apiReload: vi.fn(),
}))

const STATS = {
  store_files: 4, store_chunks: 1048, total_queries: 5, blocked_queries: 0, flagged_responses: 1,
  avg_latency_ms: 4200, latency_samples: 5,
  model_usage: { fast: 3, strong: 1, none: 1 },
  reflection_stats: { avg_confidence: 0.82, accepted_attempt: { 1: 3, 2: 2, 3: 0 }, retry_search: 2, retry_model: 0, best_effort: 1, refused: 1 },
  answers_logged: 0, success_rate: 0,
}

const DOCS = [
  { id: 'a', filename: 'report.pdf', file_type: 'pdf', chunk_count: 12, upload_time: '2026-10-01T10:00:00', is_deleted: false, status: 'ready' },
  { id: 'b', filename: 'old.txt', file_type: 'txt', chunk_count: 3, upload_time: '2026-09-01T10:00:00', is_deleted: true, status: 'ready' },
]

beforeEach(() => {
  vi.clearAllMocks()
  api.adminGetStats.mockResolvedValue(STATS)
  api.adminGetDocuments.mockResolvedValue({ documents: DOCS })
})

describe('AdminDashboard', () => {
  it('labels which stats reset on restart and which are all time', async () => {
    render(<AdminDashboard onClose={() => {}} />)
    expect(await screen.findByText('1,048')).toBeInTheDocument()
    expect(screen.getByText('resets on restart')).toBeInTheDocument()
    expect(screen.getByText('4.2 s')).toBeInTheDocument()
    expect(screen.getByText('82%')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: '1st attempt: 3, 2nd attempt: 2, 3rd attempt: 0' })).toBeInTheDocument()
    expect(screen.getByText('Best effort, not fully verified').querySelector('b')).toHaveTextContent('1')
    expect(screen.getByText('Retried with a new search').querySelector('b')).toHaveTextContent('2')
    // No logged answers yet: a dash, not a misleading "0%".
    const tile = screen.getByText('Not flagged by validator').closest('.st-tile')
    expect(within(tile).getByText('—')).toBeInTheDocument()
  })

  it('hides removed documents until asked, and removes in two steps', async () => {
    const onDocsChanged = vi.fn()
    const toast = vi.fn()
    render(<AdminDashboard onClose={() => {}} onDocsChanged={onDocsChanged} toast={toast} />)
    fireEvent.click(screen.getByRole('tab', { name: /Documents/ }))
    expect(await screen.findByText('report.pdf')).toBeInTheDocument()
    expect(screen.queryByText('old.txt')).not.toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Show 1 removed'))
    expect(screen.getByText('old.txt')).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText('Remove report.pdf'))
    expect(api.adminDeleteDocument).not.toHaveBeenCalled()
    fireEvent.click(within(screen.getByRole('group', { name: 'Remove report.pdf?' })).getByText('Remove'))
    await waitFor(() => expect(api.adminDeleteDocument).toHaveBeenCalledWith('a'))
    await waitFor(() => expect(onDocsChanged).toHaveBeenCalled())
    expect(toast).toHaveBeenCalledWith('report.pdf removed', 'info')
  })

  it('clears the library only after confirming', async () => {
    const toast = vi.fn()
    const onDocsChanged = vi.fn()
    render(<AdminDashboard onClose={() => {}} toast={toast} onDocsChanged={onDocsChanged} docCount={4} />)
    fireEvent.click(screen.getByRole('tab', { name: /Maintenance/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Remove all' }))
    expect(api.adminDeleteAllDocuments).not.toHaveBeenCalled()
    fireEvent.click(within(screen.getByRole('group', { name: 'Remove every document?' })).getByText('Remove all'))
    await waitFor(() => expect(toast).toHaveBeenCalledWith('Library cleared: 4 documents removed', 'info'))
    expect(onDocsChanged).toHaveBeenCalled()
  })

  it('cannot clear an empty library', () => {
    render(<AdminDashboard onClose={() => {}} docCount={0} />)
    fireEvent.click(screen.getByRole('tab', { name: /Maintenance/ }))
    expect(screen.getByRole('button', { name: 'Remove all' })).toBeDisabled()
  })

  it('re-indexes the server folder and reports what /reload did', async () => {
    api.apiReload.mockResolvedValue({
      status: 'done', loaded: 1, skipped: 4, failed: 1, total_files: 5, total_chunks: 1050,
      log: [{ file: 'new.md', status: 'ok', chunks: 2 }, { file: 'broken.pdf', status: 'error', msg: 'parse: bad xref' }],
    })
    const onDocsChanged = vi.fn()
    render(<AdminDashboard onClose={() => {}} onDocsChanged={onDocsChanged} docCount={4} />)
    fireEvent.click(screen.getByRole('tab', { name: /Maintenance/ }))
    fireEvent.click(screen.getByRole('button', { name: /Re-index/ }))
    const res = await screen.findByRole('status')
    expect(res).toHaveTextContent('1 new file indexed, 4 already in the library, 1 failed.')
    expect(res).toHaveTextContent('broken.pdf parse: bad xref')
    expect(onDocsChanged).toHaveBeenCalled()
  })

  it('does not refresh the library when a re-index adds nothing', async () => {
    api.apiReload.mockResolvedValue({ status: 'done', loaded: 0, skipped: 4, failed: 0, log: [] })
    const onDocsChanged = vi.fn()
    render(<AdminDashboard onClose={() => {}} onDocsChanged={onDocsChanged} docCount={4} />)
    fireEvent.click(screen.getByRole('tab', { name: /Maintenance/ }))
    fireEvent.click(screen.getByRole('button', { name: /Re-index/ }))
    expect(await screen.findByRole('status')).toHaveTextContent('No new files, 4 already in the library.')
    expect(onDocsChanged).not.toHaveBeenCalled()
  })

  it('says when stats fail to load', async () => {
    api.adminGetStats.mockRejectedValue(new Error('Failed to fetch stats'))
    render(<AdminDashboard onClose={() => {}} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch stats')
  })
})
