import { describe, expect, it } from 'vitest'
import { maskModel, modelTierStyle } from './brand.js'

describe('maskModel', () => {
  it('never leaks the raw model id', () => {
    const { label } = maskModel('llama-3.3-70b-versatile')
    expect(label).not.toContain('llama')
    expect(label).not.toContain('70b')
  })

  it('maps large/reasoning models to the Ultra tier', () => {
    expect(maskModel('some-120b-model')).toEqual({ label: 'Archiva Ultra', tier: 'ultra' })
    expect(maskModel('qwen3-something')).toEqual({ label: 'Archiva Ultra', tier: 'ultra' })
  })

  it('maps 70b/pro/strong models to the Pro tier', () => {
    expect(maskModel('llama-3.3-70b-versatile')).toEqual({ label: 'Archiva Pro', tier: 'pro' })
  })

  it('maps small/fast models to the Swift tier', () => {
    // Deliberately avoids "llama" — the Pro rule's `/llama/i` matches
    // before Swift's `/8b|.../` ever gets checked (first-match-wins), so
    // GROQ_FAST's actual default id ("llama-3.1-8b-instant") is Pro, not
    // Swift, despite being the "fast" model. See the dedicated test below.
    expect(maskModel('some-fast-mini-model')).toEqual({ label: 'Archiva Swift', tier: 'swift' })
  })

  it('documents that an 8b llama id resolves to Pro, not Swift, due to rule order', () => {
    // GROQ_FAST defaults to "llama-3.1-8b-instant" (config.py). The Pro
    // rule's bare `/llama/i` match fires before the Swift rule's `/8b/`
    // ever runs, so the "fast" model is badged as "Archiva Pro" in the
    // UI. Not necessarily wrong, but worth knowing — pinned here so a
    // future MODEL_RULES reorder is a deliberate choice, not a surprise.
    expect(maskModel('llama-3.1-8b-instant')).toEqual({ label: 'Archiva Pro', tier: 'pro' })
  })

  it('falls back to the base label for null, undefined, or "none"', () => {
    expect(maskModel(null)).toEqual({ label: 'Archiva', tier: 'base' })
    expect(maskModel(undefined)).toEqual({ label: 'Archiva', tier: 'base' })
    expect(maskModel('none')).toEqual({ label: 'Archiva', tier: 'base' })
  })

  it('falls back to the base label for an unrecognized id', () => {
    expect(maskModel('totally-unknown-model-xyz')).toEqual({ label: 'Archiva', tier: 'base' })
  })

  it('first matching rule wins for ids matching multiple patterns', () => {
    // Contains both an Ultra signal ("large") and a Pro signal ("llama") —
    // Ultra is checked first in MODEL_RULES, so it should win.
    expect(maskModel('llama-large-custom')).toEqual({ label: 'Archiva Ultra', tier: 'ultra' })
  })
})

describe('modelTierStyle', () => {
  it('returns a distinct color for each known tier', () => {
    const colors = new Set(
      ['ultra', 'pro', 'swift', 'base'].map(tier => modelTierStyle(tier).color)
    )
    expect(colors.size).toBe(4)
  })

  it('falls back to the base color for an unknown tier', () => {
    expect(modelTierStyle('unknown-tier')).toEqual(modelTierStyle('base'))
  })
})
