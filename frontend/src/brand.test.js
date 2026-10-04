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
    expect(maskModel('some-fast-mini-model')).toEqual({ label: 'Archiva Swift', tier: 'swift' })
  })

  it('maps the default Groq models by size, not by vendor name', () => {
    // GROQ_FAST / GROQ_STRONG defaults (config.py). "gpt" is a Pro family
    // name, but the parameter count decides: 20b is Swift, 120b Ultra.
    expect(maskModel('openai/gpt-oss-20b')).toEqual({ label: 'Archiva Swift', tier: 'swift' })
    expect(maskModel('openai/gpt-oss-120b')).toEqual({ label: 'Archiva Ultra', tier: 'ultra' })
  })

  it('ranks a small parameter count above a Pro family name', () => {
    expect(maskModel('llama-3.1-8b-instant')).toEqual({ label: 'Archiva Swift', tier: 'swift' })
  })

  it('does not read a version number as a parameter count', () => {
    // "3.1-8b" is 8b; the "1" of "3.1" must not combine into something else,
    // and "llama-3.3-70b" stays Pro rather than tripping the 7b/8b rule.
    expect(maskModel('llama-3.3-70b-versatile')).toEqual({ label: 'Archiva Pro', tier: 'pro' })
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
