import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Evidence from './Evidence.jsx'
import MessageBubble from './MessageBubble.jsx'

const answer = {
  id: 'a1', role: 'bot', content: 'The bank is ICICI.', streaming: false,
  model_used: 'openai/gpt-oss-120b', latency_ms: 6100, attempts: 2, reflected: true,
  reflection_reason: 'passed_all_checks', confidence: 0.78, flagged: false, intent: 'qa',
  sources: [{ filename: 'invoice.pdf', page: 1, text: 'Bank Details · Name ICICI', score: 0.03 }],
}

describe('Evidence', () => {
  it('renders the verdict, confidence, checks and the cited passage', () => {
    render(<Evidence message={answer} />)
    expect(screen.getByText('Grounded after self-healing')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: '78% confidence' })).toBeInTheDocument()
    expect(screen.getByText('Answered by Archiva Ultra')).toBeInTheDocument()
    expect(screen.getByText('invoice.pdf')).toBeInTheDocument()
    expect(screen.getByText('ICICI', { selector: 'mark' })).toBeInTheDocument()
  })
})

describe('MessageBubble selection', () => {
  it('selects a finished answer on click and on Enter', () => {
    const onSelect = vi.fn()
    render(<MessageBubble message={answer} onSelect={onSelect} />)
    const card = screen.getByRole('button', { name: 'Show evidence for this answer' })
    fireEvent.click(card)
    fireEvent.keyDown(card, { key: 'Enter' })
    expect(onSelect).toHaveBeenCalledTimes(2)
    expect(onSelect).toHaveBeenCalledWith('a1')
  })

  it('is not selectable while still streaming', () => {
    render(<MessageBubble message={{ ...answer, streaming: true }} onSelect={() => {}} />)
    expect(screen.queryByRole('button', { name: 'Show evidence for this answer' })).not.toBeInTheDocument()
  })
})
