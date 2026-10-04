/**
 * PlaybookPanel.jsx — how to use Archiva, opened from "Playbook" in the
 * library footer.
 *
 * Describes the UI as it is, and claims nothing the app doesn't do: the
 * pills and citation shown here are the real classes, and what each check
 * means is the same wording evidence.js uses.
 */

import { useLayoutEffect, useRef } from 'react'
import { AlertTriangle, BookOpen, Check, ChevronDown, RefreshCw } from 'lucide-react'
import { stagger } from 'animejs'
import Modal from './Modal.jsx'
import { BRAND } from '../brand.js'
import { go } from '../motion.js'

const ICON = { size: 13, strokeWidth: 2, className: 'ico ico-sm', 'aria-hidden': true }

const STEPS = [
  {
    title: 'Add documents',
    body: <>Drop files onto the library on the left, or click <b>browse</b>. On a phone, open the menu
      and tap <b>Add documents</b>. PDF, Word (.docx), text, Markdown, CSV and HTML are indexed straight
      away; scanned PDFs are read with OCR in the background, and their row shows the page it&rsquo;s on.</>,
  },
  {
    title: 'Ask a question',
    body: <>Type it in plain words and send it<span className="pb-kbd"> with <kbd>Enter</kbd>{' '}
      (<kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line)</span>. Not sure where to start? The cards on the home page are built from topics found in your
      documents; <b>Another question</b> shows the next one for that topic.</>,
  },
  {
    title: 'Read the answer',
    body: <>Numbered citations <span className="cite" aria-hidden="true">1</span> mark where each statement
      came from. Hover one to light up its passage, or click it to jump there. The pill above the answer
      says how it was checked:</>,
    legend: true,
  },
  {
    title: 'Check the evidence',
    body: <>Click an answer to open <b>Evidence</b>: its confidence, the checks it passed, and the cited
      passages, with the words they share with the answer highlighted.</>,
  },
  {
    title: 'Keep going',
    body: <>Follow-up questions are read in the context of the conversation. <b>New conversation</b>{' '}
      <span className="pb-kbd">(or <kbd>N</kbd>) </span>starts fresh, and <b>Export</b> saves the conversation as Markdown or PDF.</>,
  },
]

const PILLS = [
  { cls: 'pill-ok', Icon: Check, label: 'Grounded', desc: 'passed every check on the first attempt.' },
  { cls: 'pill-heal', Icon: RefreshCw, label: 'Self-healed', desc: 'a first draft failed a check, so it was retried before you saw it.' },
  { cls: 'pill-heal', Icon: AlertTriangle, label: 'Not fully verified', desc: 'the best answer still failed a check after retrying. Read it against the passages.' },
]

const FAQ = [
  {
    q: 'Why does it say "Not in your documents"?',
    a: `No passage supported an answer, so ${BRAND.name} declined instead of guessing. Try the words your document uses, or add a document that covers the question.`,
  },
  {
    q: 'What are the checks?',
    a: 'Every number in the answer must appear in its passages, the answer must not contradict them, and its wording must overlap them. An answer that fails is retried with a different search or a stricter prompt.',
  },
  {
    q: 'What does the confidence score mean?',
    a: 'How much of the answer’s wording appears in the passages it used, less a little for each retry. Below 60% the ring in Evidence turns amber.',
  },
  {
    q: 'How do I remove a document?',
    a: 'Click the × on its row in the library (it appears on hover; touch screens always show it), then confirm. Removing a scan that is still being read stops its OCR.',
  },
  {
    q: 'What are Pipeline stats and Manage library?',
    a: 'Pipeline stats shows what is indexed, how questions went since the server started (attempts, response time, confidence), and the share of all answers the output validator didn’t flag. Manage library re-indexes files placed in the server’s test_docs/ folder, or clears the whole library.',
  },
  {
    q: 'Can I change the colours?',
    a: 'Switch between Paper and Ink at the bottom of the library. Until you pick one, it follows your system setting.',
  },
]

export default function PlaybookPanel({ onClose }) {
  const ref = useRef(null)

  useLayoutEffect(() => {
    go(ref.current.querySelectorAll('.pb-step'), { opacity: [0, 1], translateY: [8, 0], delay: stagger(50, { start: 120 }), duration: 500, ease: 'outExpo' })
  }, [])

  return (
    <Modal title="Playbook" sub={`How to get answers you can check from ${BRAND.name}`} icon={BookOpen} onClose={onClose} className="pb">
      <div ref={ref}>
        <p className="pb-lede">
          {BRAND.name} answers only from the documents in your library. Each answer is checked against
          the passages it came from before you see it, and when your documents don&rsquo;t cover a
          question, it says so instead of guessing.
        </p>

        <ol className="pb-steps">
          {STEPS.map((s, i) => (
            <li className="pb-step" key={s.title}>
              <span className="pb-num" aria-hidden="true">{i + 1}</span>
              <div>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
                {s.legend && (
                  <ul className="pb-legend">
                    {PILLS.map(p => (
                      <li key={p.label}>
                        <span className={`pill ${p.cls}`}><p.Icon {...ICON} />{p.label}</span>
                        <span>{p.desc}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          ))}
        </ol>

        <h3 className="pb-h">Common questions</h3>
        <div className="pb-faq">
          {FAQ.map(item => (
            <details key={item.q}>
              <summary>{item.q}<ChevronDown size={15} strokeWidth={1.75} className="ico" aria-hidden="true" /></summary>
              <p>{item.a}</p>
            </details>
          ))}
        </div>

        <section className="pb-kb">
          <h3 className="pb-h">Keyboard</h3>
          <dl className="pb-keys">
            <div><dt><kbd>Enter</kbd></dt><dd>Send</dd></div>
            <div><dt><kbd>Shift</kbd> + <kbd>Enter</kbd></dt><dd>New line</dd></div>
            <div><dt><kbd>N</kbd></dt><dd>New conversation</dd></div>
            <div><dt><kbd>Esc</kbd></dt><dd>Close a panel</dd></div>
          </dl>
        </section>
      </div>
    </Modal>
  )
}
