import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import MessageBubble from './MessageBubble.jsx'

function botMessage(overrides = {}) {
  return {
    id: '1', role: 'bot', content: 'The answer is 42.',
    sources: [], intent: 'qa', model_used: 'llama-3.1-8b-instant',
    latency_ms: 100, flagged: false, streaming: false,
    attempts: 1, reflected: false,
    reflection_reason: 'passed_all_checks', confidence: 0.9,
    ...overrides,
  }
}

describe('MessageBubble — user messages', () => {
  it('renders user content as plain text, not markdown', () => {
    render(<MessageBubble message={{ role: 'user', content: '**not bold**' }} />)
    expect(screen.getByText('**not bold**')).toBeInTheDocument()
  })
})

describe('MessageBubble — bot messages', () => {
  it('renders markdown content as HTML', () => {
    const { container } = render(
      <MessageBubble message={botMessage({ content: 'this is **bold** text' })} />
    )
    expect(container.querySelector('strong')).toHaveTextContent('bold')
  })

  it('sanitizes injected script tags out of LLM-generated content', () => {
    const { container } = render(
      <MessageBubble message={botMessage({ content: 'hello <script>window.pwned = true</script> world' })} />
    )
    expect(container.querySelector('script')).toBeNull()
    expect(container.innerHTML).not.toContain('<script>')
  })

  it('groups multiple chunks from the same file into one source badge with a page range', () => {
    const message = botMessage({
      sources: [
        { filename: 'contract.pdf', page: 1 },
        { filename: 'contract.pdf', page: 2 },
        { filename: 'contract.pdf', page: 3 },
      ],
    })
    render(<MessageBubble message={message} />)
    // One badge for the file, not three
    expect(screen.getAllByText('contract.pdf')).toHaveLength(1)
    expect(screen.getByText('pp. 1-3')).toBeInTheDocument()
  })

  it('keeps sources from different files as separate badges', () => {
    const message = botMessage({
      sources: [
        { filename: 'a.pdf', page: 1 },
        { filename: 'b.pdf', page: 1 },
      ],
    })
    render(<MessageBubble message={message} />)
    expect(screen.getByText('a.pdf')).toBeInTheDocument()
    expect(screen.getByText('b.pdf')).toBeInTheDocument()
  })

  it('hides the source list when the answer is "not found in the document"', () => {
    const message = botMessage({
      content: 'Not found in the document.',
      sources: [{ filename: 'contract.pdf', page: 1 }],
    })
    render(<MessageBubble message={message} />)
    expect(screen.queryByText('contract.pdf')).not.toBeInTheDocument()
  })

  it('shows the reflection badge with confidence when reflected', () => {
    const message = botMessage({
      attempts: 2, reflected: true,
      reflection_reason: 'low_overlap', confidence: 0.75,
    })
    render(<MessageBubble message={message} />)
    expect(screen.getByText('Expanded search')).toBeInTheDocument()
    expect(screen.getByText(/75% confident/)).toBeInTheDocument()
  })

  it('hides the reflection badge on the first successful attempt', () => {
    const message = botMessage({ attempts: 1, reflected: false })
    render(<MessageBubble message={message} />)
    expect(screen.queryByText(/confident/)).not.toBeInTheDocument()
  })

  it('hides sources and badges while still streaming', () => {
    const message = botMessage({
      streaming: true, attempts: 2, reflected: true,
      sources: [{ filename: 'contract.pdf', page: 1 }],
    })
    render(<MessageBubble message={message} />)
    expect(screen.queryByText('contract.pdf')).not.toBeInTheDocument()
    expect(screen.queryByText('Expanded search')).not.toBeInTheDocument()
  })

  it('never renders the raw model id, only the branded label', () => {
    render(<MessageBubble message={botMessage({ model_used: 'llama-3.3-70b-versatile' })} />)
    expect(screen.queryByText(/llama/i)).not.toBeInTheDocument()
  })
})
