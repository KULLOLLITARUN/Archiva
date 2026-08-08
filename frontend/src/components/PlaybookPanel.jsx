/**
 * PlaybookPanel.jsx — Side panel that teaches users how to use Archiva.
 * Designed to be so simple even a child can follow it.
 */

import { useState } from 'react'
import {
  BookOpen, FileText, FolderOpen, HelpCircle, MessageCircle,
  RefreshCw, Sparkles, Upload, X, Zap,
} from 'lucide-react'
import { BRAND } from '../brand.js'

const STEPS = [
  {
    num: 1,
    icon: <FolderOpen size={16} />,
    title: 'Open Documents panel',
    desc: `Click the "Documents" button in the top-right corner to open the file manager.`,
    tip: 'You can upload PDF, Word (.docx), or plain text (.txt) files.',
  },
  {
    num: 2,
    icon: <Upload size={16} />,
    title: 'Upload your file',
    desc: 'Drag your file into the dotted box, or click it to open a file picker. Wait for the green "indexed" badge to appear.',
    tip: 'You can upload multiple files at the same time!',
  },
  {
    num: 3,
    icon: <MessageCircle size={16} />,
    title: 'Ask a question',
    desc: 'Type your question in the chat box at the bottom and press Enter (or the send button).',
    tip: 'Ask in plain English — no special commands needed.',
  },
  {
    num: 4,
    icon: <Zap size={16} />,
    title: 'Get your answer',
    desc: `${BRAND.name} reads your documents and types back the answer with the exact source so you can verify it.`,
    tip: 'Look for the source chips below each answer to see where the info came from.',
  },
  {
    num: 5,
    icon: <RefreshCw size={16} />,
    title: 'Self-healing AI',
    desc: `If the first answer isn't confident enough, ${BRAND.name} automatically retries with smarter settings. You'll see a small badge under the answer when this happens.`,
    tip: 'No badge = confident on the first try. Amber badge = it had to try harder. Red badge = it retried several times — worth double-checking.',
  },
  {
    num: 6,
    icon: <Sparkles size={16} />,
    title: 'Use Smart Suggestions',
    desc: 'On the home screen, click any topic chip (Explore, Analyze, Issues…) to see ready-made questions — then click any question to send it instantly.',
    tip: 'Smart Suggestions are great when you don\'t know where to start!',
  },
]

const FAQ = [
  {
    q: 'Why does it say "Not found in the document"?',
    a: 'The answer isn\'t in any uploaded file. Try uploading a document that contains the information you\'re looking for.',
  },
  {
    q: 'Can I ask follow-up questions?',
    a: 'Yes! Archiva remembers the conversation. Ask naturally, just like talking to a person.',
  },
  {
    q: 'What file types does it support?',
    a: 'PDF, Word DOCX, and plain Text TXT. One file can be any size.',
  },
  {
    q: 'How do I remove a file?',
    a: 'Open the Documents panel and click the close button next to the file you want to delete.',
  },
  {
    q: 'What do the badges under an answer mean?',
    a: 'A short label like "Expanded search" or "Used stronger AI" means the AI checked its own answer and tried again to give you something better before showing it to you. No badge at all means it was confident on the very first try.',
  },
  {
    q: 'What is "Re-index Documents"?',
    a: `If you added files directly to the server's test_docs/ folder, click Re-index to make ${BRAND.name} aware of them.`,
  },
]

export default function PlaybookPanel({ isOpen, onClose }) {
  const [activeTopic, setActiveTopic] = useState(null)

  return (
    <>
      {/* Backdrop */}
      <div
        className={`up-overlay ${isOpen ? 'up-overlay--open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
        style={{ zIndex: 103 }}
      />

      <aside
        className={`play-panel ${isOpen ? 'play-panel--open' : ''}`}
        aria-label="Playbook guide"
        style={{ zIndex: 104 }}
      >
        {/* Header */}
        <div className="play-header">
          <div className="play-header-left">
            <span className="play-header-icon"><BookOpen size={16} /></span>
            <span className="play-header-title">How to Use {BRAND.name}</span>
          </div>
          <button className="play-close-btn" onClick={onClose} aria-label="Close playbook"><X size={14} /></button>
        </div>

        <div className="play-body">

          {/* Intro card */}
          <div className="play-intro">
            <strong>{BRAND.name}</strong> lets you upload any document and ask questions about it
            in plain English. The AI finds the answer and tells you exactly where it came from.
            Follow these 5 simple steps!
          </div>

          {/* Steps */}
          <div className="play-section-title">Step-by-Step Guide</div>

          {STEPS.map((s, i) => (
            <div
              className="play-step"
              key={s.num}
              style={{ animationDelay: `${i * 0.07}s` }}
            >
              <div className="play-step-header">
                <div className="play-step-num">{s.num}</div>
                <div className="play-step-title">{s.title}</div>
                <div className="play-step-icon">{s.icon}</div>
              </div>
              <div className="play-step-desc">{s.desc}</div>
              <div className="play-step-tip">{s.tip}</div>
            </div>
          ))}

          {/* FAQ */}
          <div className="play-section-title" style={{ marginTop: 4 }}>Common Questions</div>

          {FAQ.map((item) => (
            <div className="play-qa" key={item.q} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <HelpCircle size={15} style={{ flexShrink: 0, marginTop: 1, color: 'var(--text3)' }} />
              <div>
                <div className="play-qa-q" style={{ marginBottom: 2 }}>{item.q}</div>
                <div className="play-qa-a">{item.a}</div>
              </div>
            </div>
          ))}

          {/* ── Smart Suggestions Explorer ─────────────────────── */}
          <div className="play-section-title" style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sparkles size={12} /> Smart Suggestions
          </div>

          <div className="play-intro" style={{ fontSize: 12 }}>
            After you upload a document, {BRAND.name} uses its most powerful AI
            to analyse your content and create <strong>custom topic chips</strong> —
            each with 4 clickable questions specific to your document.
          </div>

          {[
            { icon: <FileText size={16} />, label: 'Your Docs chips', desc: 'Appear automatically once documents are indexed. Click a chip to see 4 questions about that topic.' },
            { icon: <Sparkles size={16} />, label: 'AI-generated topics', desc: 'Topics are created by the AI from your actual content — not generic templates.' },
            { icon: <MessageCircle size={16} />, label: 'One-click asking', desc: 'Click any suggested question and it is sent instantly to the chat.' },
            { icon: <RefreshCw size={16} />, label: 'Auto-refreshes', desc: 'Upload a new file and topics update to reflect all your documents.' },
          ].map((item) => (
            <div key={item.label} className="play-qa" style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <span style={{ flexShrink: 0, color: 'var(--accent2)' }}>{item.icon}</span>
              <div>
                <div className="play-qa-q" style={{ marginBottom: 2 }}>{item.label}</div>
                <div className="play-qa-a">{item.desc}</div>
              </div>
            </div>
          ))}

          {/* Colour legend */}
          <div className="play-section-title" style={{ marginTop: 4 }}>Answer Quality Colours</div>

          {[
            { color: 'var(--green)', label: 'Green dot', desc: 'Server is online and documents are loaded' },
            { color: '#8b4a12', label: 'Amber badge', desc: 'The AI retried before answering — still a valid answer, just double-check it' },
            { color: '#a1483a', label: 'Red badge', desc: 'The AI retried several times and is less certain — worth verifying against the source' },
            { color: '#6b5f4d', label: 'Grey badge', desc: 'Answer not found in your documents' },
          ].map((item, i) => (
            <div key={i} className="play-qa" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: item.color, flexShrink: 0 }} />
              <div>
                <div className="play-qa-q" style={{ marginBottom: 2 }}>{item.label}</div>
                <div className="play-qa-a">{item.desc}</div>
              </div>
            </div>
          ))}

          {/* Footer */}
          <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text3)', padding: '8px 0' }}>
            {BRAND.name} · {BRAND.sub} · v{BRAND.version}
          </div>
        </div>
      </aside>
    </>
  )
}
