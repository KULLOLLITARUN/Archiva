import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Modal from './Modal.jsx'
import PlaybookPanel from './PlaybookPanel.jsx'

describe('Modal', () => {
  it('is a labelled dialog that takes focus and returns it on close', () => {
    const opener = document.createElement('button')
    document.body.appendChild(opener)
    opener.focus()
    const { unmount } = render(<Modal title="Playbook" onClose={() => {}}><button>inside</button></Modal>)
    const dialog = screen.getByRole('dialog', { name: 'Playbook' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveFocus()
    unmount()
    expect(opener).toHaveFocus()
    opener.remove()
  })

  it('closes on Escape without letting keys reach the page behind it', () => {
    const onClose = vi.fn()
    const behind = vi.fn()
    document.addEventListener('keydown', behind)
    render(<Modal title="Playbook" onClose={onClose}><p>hi</p></Modal>)
    const dialog = screen.getByRole('dialog')
    fireEvent.keyDown(dialog, { key: 'n' })
    fireEvent.keyDown(dialog, { key: 'Escape' })
    document.removeEventListener('keydown', behind)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(behind).not.toHaveBeenCalled()
  })

  it('closes from its close button', () => {
    const onClose = vi.fn()
    render(<Modal title="Playbook" onClose={onClose}><p>hi</p></Modal>)
    fireEvent.click(screen.getByLabelText('Close Playbook'))
    expect(onClose).toHaveBeenCalled()
  })
})

describe('PlaybookPanel', () => {
  it('describes the current UI, not the old one', () => {
    render(<PlaybookPanel onClose={() => {}} />)
    const text = screen.getByRole('dialog').textContent
    for (const label of ['Grounded', 'Self-healed', 'Not fully verified', 'Another question', 'Evidence', 'Markdown, CSV and HTML']) {
      expect(text).toContain(label)
    }
    // JSX drops a line break between an element and text; these need their spaces.
    expect(text).toContain('Enter (Shift + Enter for a new line)')
    expect(text).toContain('New conversation (or N) starts fresh')
    for (const stale of ['Smart Suggestions', 'source chip', 'Documents button', 'top-right']) {
      expect(text).not.toContain(stale)
    }
  })
})
