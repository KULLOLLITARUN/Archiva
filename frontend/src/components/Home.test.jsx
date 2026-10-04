import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Home, { topicIcon } from './Home.jsx'
import { BookOpen, Cloud } from 'lucide-react'

const topics = [{ label: 'Azure VM Cloning', icon: '☁️', color: '#000', prompts: ['First question?', 'Second question?'] }]

function home(props = {}) {
  const onSend = vi.fn()
  render(<Home topics={[]} topicsLoaded docCount={4} onSend={onSend} inputDisabled={false} onAddDocs={() => {}} {...props} />)
  return onSend
}

describe('Home', () => {
  it('says topics are on the way instead of implying there are none', () => {
    home({ topicsLoaded: false })
    expect(screen.getByText('Finding topics…')).toBeInTheDocument()
    expect(screen.queryByText(/Upload documents/)).not.toBeInTheDocument()
  })

  it('asks a starter question and can offer another from the same topic', () => {
    const onSend = home({ topics })
    fireEvent.click(screen.getByRole('button', { name: 'Another question about Azure VM Cloning' }))
    fireEvent.click(screen.getByText('Second question?'))
    expect(onSend).toHaveBeenCalledWith('Second question?')
  })

  it('offers to add a document when the library is empty', () => {
    const onAddDocs = vi.fn()
    home({ docCount: 0, onAddDocs })
    fireEvent.click(screen.getByText('Add your first document'))
    expect(onAddDocs).toHaveBeenCalled()
    expect(screen.queryByText('Start from your documents')).not.toBeInTheDocument()
  })

  it('picks a line icon from the topic name', () => {
    expect(topicIcon('Azure VM Cloning')).toBe(Cloud)
    expect(topicIcon('Medieval poetry')).toBe(BookOpen)
  })
})
