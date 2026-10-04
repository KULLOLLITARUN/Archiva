/**
 * Home.jsx — the empty conversation: headline, composer, and starter cards
 * built from /suggestions (topics the backend found in the documents).
 */

import { useLayoutEffect, useRef, useState } from 'react'
import {
  BookOpen, Cloud, Cpu, Landmark, MessageSquare, Receipt, RefreshCw,
  Scale, ShieldCheck, Upload,
} from 'lucide-react'
import { createTimeline, spring, splitText, stagger, utils } from 'animejs'
import InputBar from './InputBar.jsx'
import { go, reducedMotion } from '../motion.js'

const ICON = { size: 16, strokeWidth: 1.75, className: 'ico', 'aria-hidden': true }

// Topics arrive with an emoji; the design uses line icons, so pick one from
// the topic's words, with a neutral book for anything unrecognised.
const TOPIC_ICONS = [
  [/cloud|azure|aws|vm\b|server|deploy|network/i, Cloud],
  [/invoice|gst|tax|payment|bill|receipt|price|cost/i, Receipt],
  [/prompt|llm|chat|conversation|message/i, MessageSquare],
  [/transformer|neural|model|architecture|attention|machine|ai\b|cpu|compute/i, Cpu],
  [/contract|legal|policy|law|compliance|terms/i, Scale],
  [/bank|finance|account|loan/i, Landmark],
  [/security|safety|risk/i, ShieldCheck],
]
export function topicIcon(label = '') {
  return (TOPIC_ICONS.find(([re]) => re.test(label)) || [null, BookOpen])[1]
}

function Starter({ topic, onAsk, disabled }) {
  const [i, setI] = useState(0)
  const prompts = topic.prompts || []
  const Icon = topicIcon(topic.label)
  const prompt = prompts[i % Math.max(prompts.length, 1)]
  if (!prompt) return null

  // The pointer spotlight follows the cursor inside the card.
  const onPointerMove = e => {
    const r = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
    e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
  }

  return (
    <div className="starter" onPointerMove={onPointerMove}>
      <button type="button" className="starter-main" onClick={() => onAsk(prompt)} disabled={disabled}>
        <span className="starter-ico"><Icon {...ICON} /></span>
        <span><b>{topic.label}</b><p>{prompt}</p></span>
      </button>
      {prompts.length > 1 && (
        <button type="button" className="starter-more" onClick={() => setI(n => n + 1)}
          aria-label={`Another question about ${topic.label}`}>
          <RefreshCw {...ICON} size={13} />Another question
          <span className="tabular">{(i % prompts.length) + 1}/{prompts.length}</span>
        </button>
      )}
    </div>
  )
}

export default function Home({ topics, topicsLoaded, docCount, onSend, inputDisabled, onAddDocs }) {
  const ref = useRef(null)
  const shown = topics.filter(t => t.prompts?.length).slice(0, 4)

  // Entrance, once per visit to the home view: the headline's words rise
  // out of a clip, then the rest settles in behind them.
  useLayoutEffect(() => {
    if (reducedMotion()) return undefined
    const root = ref.current
    const q = s => root.querySelectorAll(s)
    const split = splitText(root.querySelector('h1'), { words: { wrap: 'clip', class: 'w' } })
    utils.set(q('.kicker, .lede, .composer, .composer-note, .starters-h'), { opacity: 0 })
    utils.set(split.words, { translateY: '108%' })
    createTimeline({ defaults: { ease: 'outExpo', duration: 850 } })
      .add(q('.kicker'), { opacity: [0, 1], translateY: [8, 0], duration: 500 })
      .add(split.words, { translateY: ['108%', '0%'], delay: stagger(60) }, '-=300')
      .add(q('.lede'), { opacity: [0, 1], translateY: [10, 0] }, '-=600')
      .add(q('.composer'), { opacity: [0, 1], translateY: [16, 0], scale: [0.985, 1] }, '-=650')
      .add(q('.composer-note, .starters-h'), { opacity: [0, 1] }, '-=600')
    return () => split.revert()
  }, [])

  // Starter cards arrive whenever /suggestions answers, often after the
  // headline has played; they spring in on their own.
  const cardsKey = shown.map(t => t.label).join('|')
  useLayoutEffect(() => {
    const cards = ref.current.querySelectorAll('.starter')
    if (cards.length) {
      go(cards, { opacity: [0, 1], translateY: [14, 0], delay: stagger(70), ease: spring({ bounce: 0.25, duration: 700 }) })
    }
  }, [cardsKey])

  return (
    <div className="home" ref={ref}>
      <span className="kicker"><ShieldCheck {...ICON} size={14} />Verified against your documents</span>
      <h1>What do you want <em>to know?</em></h1>
      <p className="lede">
        {docCount
          ? <>Every answer is checked against its sources and cited to the page &mdash; or honestly refused.</>
          : <>Add a document and ask it anything. Every answer is checked against its sources and cited to the page.</>}
      </p>

      <InputBar id="ask-home" onSend={onSend} isLoading={inputDisabled} />
      <p className="composer-note"><kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line</p>

      {docCount > 0 && (shown.length > 0 || !topicsLoaded) && (
        <>
          <div className="starters-h">
            <b>Start from your documents</b>
            <span>{shown.length ? `${shown.length} topic${shown.length === 1 ? '' : 's'} found` : 'Finding topics…'}</span>
          </div>
          <div className="starters" aria-busy={!topicsLoaded || undefined}>
            {shown.length
              ? shown.map(t => <Starter key={t.label} topic={t} onAsk={onSend} disabled={inputDisabled} />)
              : [0, 1, 2, 3].map(k => <div key={k} className="starter starter--ghost" aria-hidden="true" />)}
          </div>
        </>
      )}

      {docCount === 0 && (
        <button type="button" className="starter starter--add" onClick={onAddDocs}>
          <span className="starter-ico"><Upload {...ICON} /></span>
          <span><b>Add your first document</b>
            <p>PDF, Word, text, Markdown, CSV, HTML, or a scanned page.</p></span>
        </button>
      )}
    </div>
  )
}
