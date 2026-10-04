/**
 * Sidebar.jsx — the Library column: brand, new conversation, documents,
 * and the low-frequency tools (Playbook, stats and library maintenance,
 * theme).
 *
 * A fixed column from 900px up; below that it is a slide-in menu, `open`
 * controlled by App.
 */

import { useLayoutEffect, useRef } from 'react'
import { BookOpen, Gauge, Moon, Plus, Sun, X } from 'lucide-react'
import { stagger } from 'animejs'
import BrandMark from './BrandMark.jsx'
import Library from './Library.jsx'
import { BRAND } from '../brand.js'
import { go } from '../motion.js'
import { matches, NARROW } from '../useMedia.js'

const ICON = { size: 16, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

export default function Sidebar({
  docsInfo, open, inert, onClose, onNewChat, onDocsChanged, toast, uploadRef, onPlaybook, onStats, theme, onTheme,
}) {
  const { total_files = 0, total_chunks = 0 } = docsInfo
  const ref = useRef(null)

  // The library slides in once on first paint. Not on phones, where it
  // starts off-screen and would animate unseen.
  useLayoutEffect(() => {
    if (matches(NARROW)) return
    // Document rows animate themselves as they load (see Library).
    const items = ref.current.querySelectorAll('.side-top, .new-chat, .side-h, .drop, .side-foot')
    go(items, { opacity: [0, 1], translateX: [-10, 0], delay: stagger(40), duration: 600, ease: 'outExpo' })
  }, [])

  return (
    <aside
      ref={ref}
      id="library"
      className={`side${open ? ' open' : ''}`}
      aria-label="Library"
      inert={inert ? '' : undefined}
    >
      <div className="side-top">
        <BrandMark />
        <span className="wordmark">{BRAND.name}<small>{BRAND.sub}</small></span>
        <button type="button" className="ibtn side-close" onClick={onClose} aria-label="Close library">
          <X {...ICON} />
        </button>
      </div>

      <button type="button" className="new-chat" onClick={onNewChat}>
        <Plus {...ICON} />New conversation<kbd aria-hidden="true">N</kbd>
      </button>

      <div className="side-h">
        Library
        <span className="tabular">
          {total_files} doc{total_files === 1 ? '' : 's'} · {total_chunks.toLocaleString()} passages
        </span>
      </div>
      <Library docsInfo={docsInfo} onDocsChanged={onDocsChanged} toast={toast} uploadRef={uploadRef} />

      <div className="side-foot">
        <button type="button" className="side-link" onClick={onPlaybook}><BookOpen {...ICON} />Playbook</button>
        <button type="button" className="side-link" onClick={onStats}><Gauge {...ICON} />Stats &amp; maintenance</button>
        <div className="theme" role="group" aria-label="Theme">
          <button type="button" aria-pressed={theme === 'paper'} onClick={() => onTheme('paper')}>
            <Sun {...ICON} size={14} />Paper
          </button>
          <button type="button" aria-pressed={theme === 'ink'} onClick={() => onTheme('ink')}>
            <Moon {...ICON} size={14} />Ink
          </button>
        </div>
      </div>
    </aside>
  )
}
