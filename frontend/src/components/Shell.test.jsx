import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import TopBar from './TopBar.jsx'
import Sidebar from './Sidebar.jsx'

vi.mock('../api.js', () => ({ apiHealth: () => Promise.resolve({}) }))

const docsInfo = { files: [], total_files: 2, total_chunks: 1048 }

function topBar(props = {}) {
  return render(
    <TopBar title="New conversation" subtitle="" docsInfo={docsInfo} backendStatus=""
      hasMessages={false} evOpen={false} onMenu={() => {}} onToggleEvidence={() => {}}
      onExport={() => {}} {...props} />
  )
}

describe('TopBar', () => {
  it('hides export and the evidence toggle before the first message', () => {
    topBar()
    expect(screen.queryByLabelText('Export conversation')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Show evidence')).not.toBeInTheDocument()
  })

  it('exports as Markdown or PDF from the export menu', () => {
    const onExport = vi.fn()
    topBar({ hasMessages: true, onExport })
    fireEvent.click(screen.getByLabelText('Export conversation'))
    fireEvent.click(screen.getByRole('menuitem', { name: /PDF/ }))
    expect(onExport).toHaveBeenCalledWith('pdf')
    expect(screen.queryByRole('menu')).not.toBeInTheDocument()
  })

  it('shows the backend connection message instead of the ready count', () => {
    topBar({ backendStatus: 'Connecting to backend…' })
    expect(screen.getByRole('status')).toHaveTextContent('Connecting to backend…')
  })
})

describe('Sidebar', () => {
  it('shows library totals and switches theme', () => {
    const onTheme = vi.fn()
    render(<Sidebar docsInfo={docsInfo} theme="paper" onTheme={onTheme} onClose={() => {}}
      onNewChat={() => {}} onUpload={() => {}} onPlaybook={() => {}} onStats={() => {}} />)
    expect(screen.getByText('2 docs · 1,048 passages')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Paper/ })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: /Ink/ }))
    expect(onTheme).toHaveBeenCalledWith('ink')
  })
})
