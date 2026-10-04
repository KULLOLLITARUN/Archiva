import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import HistoryPanel from './HistoryPanel.jsx'
import * as api from '../api.js'

vi.mock('../api.js', () => ({
  apiListConversations: vi.fn(),
  apiDeleteConversation: vi.fn(() => Promise.resolve({ deleted: true })),
}))

const ROWS = [
  { session_id: 's1', title: 'What is the invoice total?', turns: 3, updated_at: new Date().toISOString() },
  { session_id: 's2', title: 'How do I clone the VM?', turns: 1, updated_at: '2026-01-02T10:00:00Z' },
]

beforeEach(() => {
  vi.clearAllMocks()
  api.apiListConversations.mockResolvedValue(ROWS)
})

describe('HistoryPanel', () => {
  it('lists conversations and marks the open one', async () => {
    render(<HistoryPanel currentId="s1" onOpen={vi.fn()} onClose={() => {}} />)
    expect(await screen.findByText('What is the invoice total?')).toBeInTheDocument()
    expect(screen.getByText(/3 questions · just now/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^What is the invoice total/ })).toHaveAttribute('aria-current', 'true')
  })

  it('opens a conversation, then closes', async () => {
    const onOpen = vi.fn(() => Promise.resolve())
    const onClose = vi.fn()
    render(<HistoryPanel onOpen={onOpen} onClose={onClose} />)
    fireEvent.click(await screen.findByRole('button', { name: /^How do I clone the VM/ }))
    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(onOpen).toHaveBeenCalledWith('s2')
  })

  it('deletes only after confirming, and reports it', async () => {
    const onDeleted = vi.fn()
    const toast = vi.fn()
    render(<HistoryPanel onOpen={vi.fn()} onClose={() => {}} onDeleted={onDeleted} toast={toast} />)
    fireEvent.click(await screen.findByLabelText('Delete "How do I clone the VM?"'))
    expect(api.apiDeleteConversation).not.toHaveBeenCalled()
    fireEvent.click(within(screen.getByRole('group', { name: 'Delete "How do I clone the VM?"?' })).getByText('Delete'))
    await waitFor(() => expect(onDeleted).toHaveBeenCalledWith('s2'))
    expect(screen.queryByText('How do I clone the VM?')).not.toBeInTheDocument()
    expect(toast).toHaveBeenCalledWith('Conversation deleted', 'info')
  })

  it('says when there is nothing saved yet', async () => {
    api.apiListConversations.mockResolvedValue([])
    render(<HistoryPanel onOpen={vi.fn()} onClose={() => {}} />)
    expect(await screen.findByText(/No saved conversations yet/)).toBeInTheDocument()
  })
})
