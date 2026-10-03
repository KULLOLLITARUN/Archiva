import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import MessageBubble, { stripInlineCitations } from './MessageBubble.jsx'

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

describe('stripInlineCitations', () => {
  it('removes [Source: …] tags', () => {
    expect(stripInlineCitations('It uses Disk-Clone-1. [Source: plan.docx, page 1]'))
      .toBe('It uses Disk-Clone-1.')
  })

  it('removes the full-width 【Source: …】 variant', () => {
    expect(stripInlineCitations('It fails.【Source: plan.docx, page 1】')).toBe('It fails.')
  })

  it('removes a tag in the middle of a sentence without leaving a double space', () => {
    expect(stripInlineCitations('Zone 1 [Source: a.docx, page 2] is required.'))
      .toBe('Zone 1 is required.')
  })

  it('removes every tag in a bulleted answer', () => {
    const text = '- one [Source: a.docx, page 1]\n- two [Source: a.docx, page 1]\n- three [Source: a.docx, page 1]'
    expect(stripInlineCitations(text)).toBe('- one\n- two\n- three')
  })

  it('leaves ordinary brackets and links alone', () => {
    const text = 'See [step 1] and [the docs](https://example.com) for [1] details.'
    expect(stripInlineCitations(text)).toBe(text)
  })

  it('handles empty and missing text', () => {
    expect(stripInlineCitations('')).toBe('')
    expect(stripInlineCitations(undefined)).toBe('')
  })

  it('collapses the blank lines a removed tag leaves behind', () => {
    expect(stripInlineCitations('Answer.\n\n[Source: a.docx, page 1]\n\nNext.')).toBe('Answer.\n\nNext.')
  })

  describe('while streaming', () => {
    it('hides a citation that has started but not finished', () => {
      expect(stripInlineCitations('Zone 1. [Source: plan.do', true)).toBe('Zone 1.')
      expect(stripInlineCitations('Zone 1. [Sou', true)).toBe('Zone 1.')
    })

    it('does not hide an unrelated open bracket', () => {
      expect(stripInlineCitations('Press [Enter', true)).toBe('Press [Enter')
    })

    it('leaves a half-written tag alone when not streaming', () => {
      expect(stripInlineCitations('Zone 1. [Source: plan.do', false)).toBe('Zone 1. [Source: plan.do')
    })
  })
})

describe('MessageBubble — inline citations', () => {
  const cited = 'It uses Disk-Clone-1. [Source: plan.docx, page 1]'

  it('hides inline tags when source pills are shown', () => {
    const { container } = render(
      <MessageBubble message={botMessage({ content: cited, sources: [{ filename: 'plan.docx', page: 1 }] })} />
    )
    expect(container.querySelector('.markdown-body').textContent).not.toContain('[Source')
    expect(container.querySelector('.markdown-body').textContent).toContain('Disk-Clone-1')
  })

  it('keeps inline tags when there are no pills, since then they are the only provenance', () => {
    const { container } = render(<MessageBubble message={botMessage({ content: cited, sources: [] })} />)
    expect(container.querySelector('.markdown-body').textContent).toContain('[Source: plan.docx, page 1]')
  })

  it('still shows the source pill', () => {
    const { container } = render(
      <MessageBubble message={botMessage({ content: cited, sources: [{ filename: 'plan.docx', page: 1 }] })} />
    )
    expect(container.textContent).toContain('plan.docx')
  })
})
