import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { initialTheme, THEME_KEY, useTheme } from './theme.js'

function stubSystem(dark) {
  vi.stubGlobal('matchMedia', q => ({
    matches: q.includes('dark') ? dark : false,
    addEventListener: () => {},
    removeEventListener: () => {},
  }))
}

describe('theme', () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => vi.unstubAllGlobals())

  it('follows the system setting when nothing is saved', () => {
    stubSystem(true)
    expect(initialTheme()).toBe('ink')
    stubSystem(false)
    expect(initialTheme()).toBe('paper')
  })

  it('prefers a saved choice over the system setting', () => {
    stubSystem(true)
    localStorage.setItem(THEME_KEY, 'paper')
    expect(initialTheme()).toBe('paper')
  })

  it('ignores a corrupt saved value', () => {
    stubSystem(true)
    localStorage.setItem(THEME_KEY, 'neon')
    expect(initialTheme()).toBe('ink')
  })

  it('applies and remembers an explicit pick', () => {
    stubSystem(false)
    const { result } = renderHook(() => useTheme())
    act(() => result.current[1]('ink'))
    expect(result.current[0]).toBe('ink')
    expect(document.documentElement.dataset.theme).toBe('ink')
    expect(localStorage.getItem(THEME_KEY)).toBe('ink')
  })
})
