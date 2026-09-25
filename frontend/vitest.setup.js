import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// globals: false in vite.config.js's test block means Testing Library's
// own auto-cleanup (which only registers if it finds a *global* afterEach)
// never fires, so a render() from one test leaks into the next. Register
// it explicitly instead of flipping on vitest's global injection.
afterEach(() => cleanup())
