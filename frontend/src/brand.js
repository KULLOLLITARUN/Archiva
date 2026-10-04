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
  name: 'Archiva',
  tagline: 'Ask Your Documents Anything',
  // Was "AI Document Intelligence" — generic AI-product boilerplate that
  // could describe literally any RAG tool. This names the thing that
  // actually differentiates Archiva: answers are refused rather than
  // guessed, retried against a deterministic quality check, and every
  // claim traces to a cited page.
  sub: 'Every Answer, Sourced',
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
    test: id => /120b|qwen3|32b|reasoning|ultra|large/i.test(id),
    label: 'Archiva Ultra',
    tier: 'ultra',
  },
  // Swift tier by parameter count. Checked before the family names below:
  // the size says more about the tier than the vendor does, and family
  // names are too broad — "gpt" matched openai/gpt-oss-20b (the default
  // fast model) and badged it Pro.
  {
    test: id => /(^|[^\d.])(7|8|13|20)b\b/i.test(id),
    label: 'Archiva Swift',
    tier: 'swift',
  },
  // Pro tier — default quality models
  {
    test: id => /70b|pro|strong|gpt|claude|gemini|llama/i.test(id),
    label: 'Archiva Pro',
    tier: 'pro',
  },
  // Swift tier — fast / small models named without a size
  {
    test: id => /mini|swift|fast|small|turbo/i.test(id),
    label: 'Archiva Swift',
    tier: 'swift',
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
    case 'ultra': return { color: '#8b4a12' }  // deep terracotta — matches --accent
    case 'pro': return { color: '#6b3910' }  // deeper terracotta — matches --accent2
    case 'swift': return { color: '#3f7350' }  // deep sage — matches --green
    default: return { color: '#6b5f4d' }  // warm grey — matches --text2
  }
}
