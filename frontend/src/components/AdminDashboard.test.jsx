import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import AdminDashboard from './AdminDashboard.jsx'
import * as api from '../api.js'

vi.mock('../api.js', () => ({
  adminGetStats: vi.fn(),
  adminGetDocuments: vi.fn(),
  adminDeleteDocument: vi.fn(() => Promise.resolve({ deleted: true })),
  adminDeleteAllDocuments: vi.fn(() => Promise.resolve({ deleted: true, count: 99 })),
}))

const STATS = {
  store_files: 4, store_chunks: 1048, total_queries: 5, blocked_queries: 0, flagged_responses: 1,
  avg_latency_ms: 4200, latency_samples: 5,
  model_usage: { fast: 3, strong: 1, none: 1 },
  reflection_stats: { avg_confidence: 0.82, accepted_attempt: { 1: 3, 2: 2, 3: 0 } },
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

  it('clears the library only after confirming, without quoting the endpoint count', async () => {
    const toast = vi.fn()
    render(<AdminDashboard onClose={() => {}} toast={toast} />)
    fireEvent.click(screen.getByRole('tab', { name: /Documents/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'Remove all' }))
    expect(api.adminDeleteAllDocuments).not.toHaveBeenCalled()
    fireEvent.click(within(screen.getByRole('group', { name: 'Remove every document?' })).getByText('Remove all'))
    await waitFor(() => expect(toast).toHaveBeenCalledWith('Library cleared', 'info'))
  })

  it('says when stats fail to load', async () => {
    api.adminGetStats.mockRejectedValue(new Error('Failed to fetch stats'))
    render(<AdminDashboard onClose={() => {}} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch stats')
  })
})
