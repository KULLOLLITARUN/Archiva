import { describe, expect, it } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { nextQuestions } from './suggest.js'
import ChatWindow from './components/ChatWindow.jsx'

const TOPICS = [
  { label: 'Azure VM Cloning', prompts: ['How do I snapshot the VM disk?', 'Why is the specialized VM clone safe?', 'Which ports does the cloned VM open?'] },
  { label: 'GST Invoice Details', prompts: ['Which bank and UPI ID are on the invoice?', 'What is the invoice total?'] },
  { label: 'Transformer Architecture', prompts: ['How does self-attention differ from RNNs?'] },
]
const convo = (q, a) => [
  { id: 'q1', role: 'user', content: q },
  { id: 'a1', role: 'bot', content: a, streaming: false, sources: [] },
]

describe('nextQuestions', () => {
  it('ranks questions that share words with the last exchange first', () => {
    const got = nextQuestions(TOPICS, convo('What is the invoice total?', 'The invoice total is 1,250 rupees.'))
    expect(got[0]).toEqual({ label: 'GST Invoice Details', prompt: 'Which bank and UPI ID are on the invoice?' })
  })

  it('never repeats a question already asked in the conversation', () => {
    const got = nextQuestions(TOPICS, convo('What is the invoice total?', 'It is 1,250.')).map(x => x.prompt)
    expect(got).not.toContain('What is the invoice total?')
  })

  it('takes at most two from one topic and returns at most three', () => {
    const got = nextQuestions(TOPICS, convo('How do I clone the VM?', 'Snapshot the VM disk, then create the cloned VM.'))
    expect(got).toHaveLength(3)
    expect(got.filter(x => x.label === 'Azure VM Cloning')).toHaveLength(2)
  })

  it('is empty without topics', () => {
    expect(nextQuestions([], convo('q', 'a'))).toEqual([])
  })
})

describe('Ask next in the thread', () => {
  const props = { streamStatus: '', topics: TOPICS, topicsLoaded: true, docCount: 3 }

  it('appears under a finished answer and sends the picked question', () => {
    const sent = []
    render(<ChatWindow {...props} messages={convo('What is the invoice total?', 'The invoice total is 1,250.')}
      onSend={q => sent.push(q)} inputDisabled={false} />)
    const group = screen.getByRole('group', { name: 'Ask next' })
    fireEvent.click(group.querySelector('.next-q'))
    expect(sent).toEqual(['Which bank and UPI ID are on the invoice?'])
  })

  it('is hidden while an answer is still coming', () => {
    const msgs = convo('q', '')
    msgs[1].streaming = true
    render(<ChatWindow {...props} messages={msgs} onSend={() => {}} inputDisabled />)
    expect(screen.queryByRole('group', { name: 'Ask next' })).not.toBeInTheDocument()
  })
})
