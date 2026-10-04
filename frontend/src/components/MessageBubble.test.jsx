import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import MessageBubble from './MessageBubble.jsx'

function botMessage(overrides = {}) {
  return {
    id: 'a1', role: 'bot', content: 'The answer is 42.',
    sources: [{ filename: 'facts.pdf', page: 3, text: 'The answer is 42.', score: 0.03 }],
    intent: 'qa', model_used: 'llama-3.1-8b-instant',
    latency_ms: 2400, flagged: false, streaming: false,
    attempts: 1, reflected: false,
    reflection_reason: 'passed_all_checks', confidence: 0.9,
    ...overrides,
  }
}

describe('MessageBubble — questions', () => {
  it('renders the question as plain text, not markdown', () => {
    render(<MessageBubble message={{ id: 'q1', role: 'user', content: '**not bold**' }} />)
    expect(screen.getByText('**not bold**')).toBeInTheDocument()
  })
})

describe('MessageBubble — answers', () => {
  it('renders markdown content as HTML', () => {
    const { container } = render(<MessageBubble message={botMessage({ content: 'this is **bold** text' })} />)
    expect(container.querySelector('.ans-body strong')).toHaveTextContent('bold')
  })

  it('sanitizes injected script tags out of LLM-generated content', () => {
    const { container } = render(
      <MessageBubble message={botMessage({ content: 'hello <script>window.pwned = true</script> world' })} />
    )
    expect(container.querySelector('script')).toBeNull()
  })

  it('shows the intent, a Grounded pill, confidence, sources and latency', () => {
    render(<MessageBubble message={botMessage()} onSelect={() => {}} />)
    expect(screen.getByText('Q&A')).toBeInTheDocument()
    expect(screen.getByText('Grounded')).toBeInTheDocument()
    expect(screen.getByText('90%')).toBeInTheDocument()
    expect(screen.getByText('1 source')).toBeInTheDocument()
    expect(screen.getByText('2.4 s')).toBeInTheDocument()
  })

  it('marks an answer that needed a retry as self-healed', () => {
    render(<MessageBubble message={botMessage({ attempts: 2, reflected: true })} />)
    expect(screen.getByText('Self-healed')).toBeInTheDocument()
  })

  it('never renders the raw model id', () => {
    const { container } = render(<MessageBubble message={botMessage()} />)
    expect(container.textContent).not.toMatch(/llama|8b/i)
  })

  it('turns an inline source tag into a numbered citation', () => {
    const onCite = vi.fn()
    const onSelect = vi.fn()
    render(<MessageBubble onSelect={onSelect} onCite={onCite}
      message={botMessage({ content: 'The answer is 42. [Source: facts.pdf, page 3]' })} />)
    expect(screen.queryByText(/Source:/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Source 1' }))
    expect(onCite).toHaveBeenCalledWith('a1', [1])
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('shows a refusal in the not-found style, with no pills or footer', () => {
    const { container } = render(<MessageBubble docCount={4} message={botMessage({
      content: 'Not found in the document.', reflection_reason: 'explicit_not_found', sources: [], confidence: 0,
    })} />)
    expect(container.querySelector('.ans--nf')).not.toBeNull()
    expect(screen.getByText(/none of your 4 documents/)).toBeInTheDocument()
    expect(container.querySelector('.ans-foot')).toBeNull()
  })

  it('shows the waiting placeholder until the first token', () => {
    render(<MessageBubble message={botMessage({ content: '', streaming: true })} status="Searching your documents…" />)
    expect(screen.getByRole('status')).toHaveTextContent('Searching your documents…')
  })

  it('hides pills and footer while the answer streams in', () => {
    const { container } = render(<MessageBubble message={botMessage({ content: 'The ans', streaming: true })} />)
    expect(container.querySelector('.ans-head')).toBeNull()
    expect(container.querySelector('.ans-foot')).toBeNull()
  })
})
