import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// globals: false in vite.config.js's test block means Testing Library's
// own auto-cleanup (which only registers if it finds a *global* afterEach)
// never fires, so a render() from one test leaks into the next. Register
// it explicitly instead of flipping on vitest's global injection.
afterEach(() => cleanup())

// jsdom has no ResizeObserver; anime.js's splitText (Home's headline) uses
// one to re-split on resize. Layout never changes in jsdom, so a no-op does.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// splitText also waits on document.fonts before splitting; jsdom has none.
if (!document.fonts) {
  Object.defineProperty(document, 'fonts', {
    configurable: true,
    value: { status: 'loaded', ready: Promise.resolve(), addEventListener() {}, removeEventListener() {} },
  })
}
