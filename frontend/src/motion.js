/**
 * motion.js — thin wrapper over anime.js so every animation honours
 * prefers-reduced-motion in one place.
 *
 * Motion is reserved for entrances and direct interaction; nothing loops.
 * With reduced motion, `go` jumps straight to each property's end value
 * instead of animating, so layouts that depend on the final state (an
 * element animated in from opacity 0) still end up visible.
 */

import { animate, utils } from 'animejs'

export function reducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}

const KEYS_NOT_PROPS = new Set(['duration', 'delay', 'ease', 'loop', 'alternate', 'autoplay', 'onComplete', 'onBegin', 'onUpdate'])

function endState(props) {
  const out = {}
  for (const [k, v] of Object.entries(props)) {
    if (KEYS_NOT_PROPS.has(k)) continue
    out[k] = Array.isArray(v) ? v[v.length - 1] : v
  }
  return out
}

export function go(targets, props) {
  if (reducedMotion()) {
    utils.set(targets, endState(props))
    return null
  }
  return animate(targets, props)
}
