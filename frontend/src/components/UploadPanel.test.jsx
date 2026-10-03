import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import UploadPanel, { fileMeta } from './UploadPanel.jsx'
import DocsStrip from './DocsStrip.jsx'

const noop = () => {}

function panel(files, totals = {}) {
  return render(
    <UploadPanel
      isOpen
      docsInfo={{ files, total_files: 0, total_chunks: 0, ...totals }}
      onClose={noop}
      onClearChat={noop}
      onDocsChanged={noop}
    />
  )
}

describe('fileMeta', () => {
  it('shows chunk count for ready documents', () => {
    expect(fileMeta({ status: 'ready', chunk_count: 12 })).toBe('12 chunks')
  })

  it('shows OCR page progress while processing', () => {
    expect(fileMeta({ status: 'processing', progress: [3, 10] })).toBe('OCR in progress · page 3/10')
  })

  it('says queued before any page has been processed', () => {
    expect(fileMeta({ status: 'processing', progress: null })).toBe('Queued for OCR')
  })

  it('surfaces the failure reason', () => {
    expect(fileMeta({ status: 'failed', message: 'OCR found no readable text' }))
      .toBe('OCR found no readable text')
  })

  it('falls back to a generic message when a failure has no reason', () => {
    expect(fileMeta({ status: 'failed' })).toBe('Processing failed')
  })
})

describe('UploadPanel — document list states', () => {
  it('lists a processing scan with progress and a cancel button', () => {
    panel([{
      file_id: 'f1', filename: 'scan.pdf', chunk_count: 0,
      status: 'processing', progress: [2, 8],
    }])
    expect(screen.getByText('scan.pdf')).toBeInTheDocument()
    expect(screen.getByText('OCR in progress · page 2/8')).toBeInTheDocument()
    expect(screen.getByText('OCR running')).toBeInTheDocument()
    expect(screen.getByTitle('Cancel and remove scan.pdf')).toBeInTheDocument()
  })

  it('lists a failed document with its reason and a remove button', () => {
    panel([{
      file_id: 'f2', filename: 'blurry.pdf', chunk_count: 0,
      status: 'failed', message: 'OCR found no readable text',
    }])
    expect(screen.getByText('OCR found no readable text')).toBeInTheDocument()
    expect(screen.getByText('failed')).toBeInTheDocument()
    expect(screen.getByTitle('Remove blurry.pdf')).toBeInTheDocument()
  })

  it('does not show a status pill for ready documents', () => {
    panel([{ file_id: 'f3', filename: 'ok.pdf', chunk_count: 7, status: 'ready' }],
      { total_files: 1, total_chunks: 7 })
    expect(screen.getByText('7 chunks')).toBeInTheDocument()
    expect(screen.queryByText('OCR running')).not.toBeInTheDocument()
    expect(screen.queryByText('failed')).not.toBeInTheDocument()
  })
})

describe('DocsStrip — pending documents', () => {
  it('marks a processing document as not searchable yet', () => {
    render(<DocsStrip onManage={noop} docsInfo={{
      files: [{ filename: 'scan.pdf', chunk_count: 0, status: 'processing' }],
    }} />)
    expect(screen.getByText('OCR…')).toBeInTheDocument()
    expect(screen.getByTitle(/OCR in progress, not searchable yet/)).toBeInTheDocument()
  })

  it('flags a failed document', () => {
    render(<DocsStrip onManage={noop} docsInfo={{
      files: [{ filename: 'bad.pdf', chunk_count: 0, status: 'failed', message: 'no text' }],
    }} />)
    expect(screen.getByText('!')).toBeInTheDocument()
    expect(screen.getByTitle(/no text/)).toBeInTheDocument()
  })
})
