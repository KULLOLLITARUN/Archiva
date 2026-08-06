/**
 * brand.js — Single source of truth for Archiva branding and model display names.
 *
 * DESIGN PRINCIPLE:
 *   No internal model identifiers (provider names, parameter counts, version
 *   strings) are ever shown to end users. Everything maps to Archiva-branded
 *   tier names that communicate capability without leaking infrastructure.
 *
 * Tier names:
 *   Archiva Swift   — fast, lightweight responses
 *   Archiva Pro     — default high-quality responses
 *   Archiva Ultra   — maximum reasoning, used for hard queries
 *   Archiva         — generic fallback label
 */

// ── Brand constants ────────────────────────────────────────────────────────────

export const BRAND = {
  name:    'Archiva',
  tagline: 'Ask Your Documents Anything',
  sub:     'AI Document Intelligence',
  version: '2.0',
}

// ── Model tier rules (ordered — first match wins) ─────────────────────────────
//
// Each rule: { test: fn(rawModelId) => bool, label: string, tier: string }
//
// "tier" controls badge colour in the UI:
//   swift  → teal/green
//   pro    → purple/violet
//   ultra  → gold/amber
//   base   → grey (fallback)

const MODEL_RULES = [
  // Ultra / Reasoning tier — large or reasoning-class models
  {
    test:  id => /120b|qwen3|32b|reasoning|ultra|large/i.test(id),
    label: 'Archiva Ultra',
    tier:  'ultra',
  },
  // Pro tier — default quality models
  {
    test:  id => /70b|pro|strong|gpt|claude|gemini|llama/i.test(id),
    label: 'Archiva Pro',
    tier:  'pro',
  },
  // Swift tier — fast / small models
  {
    test:  id => /8b|13b|7b|mini|swift|fast|small|turbo|20b/i.test(id),
    label: 'Archiva Swift',
    tier:  'swift',
  },
]

const FALLBACK_MODEL = { label: 'Archiva', tier: 'base' }

/**
 * maskModel(rawModelId) → { label: string, tier: string }
 *
 * Converts any raw model identifier into a branded display label.
 * Never returns the raw id. Safe to call with null / undefined / 'none'.
 *
 * @param {string|null|undefined} rawModelId
 * @returns {{ label: string, tier: string }}
 */
export function maskModel(rawModelId) {
  if (!rawModelId || rawModelId === 'none') return FALLBACK_MODEL
  for (const rule of MODEL_RULES) {
    if (rule.test(rawModelId)) return { label: rule.label, tier: rule.tier }
  }
  return FALLBACK_MODEL
}

/**
 * Tier colour map for the meta row badge.
 * Returns inline style properties.
 */
export function modelTierStyle(tier) {
  switch (tier) {
    case 'ultra': return { color: '#fbbf24' }  // gold
    case 'pro':   return { color: '#a78bfa' }  // violet
    case 'swift': return { color: '#22d3a5' }  // teal
    default:      return { color: '#6b7280' }  // grey
  }
}
