import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import Library from './Library.jsx'
import { docBadge, docStatus, isAccepted } from '../docs.js'
import * as api from '../api.js'

vi.mock('../api.js', () => ({ apiUpload: vi.fn(), apiDeleteFile: vi.fn() }))

const ready = { file_id: 'f1', filename: 'Plan.docx', chunk_count: 49, status: 'ready' }
const reading = { file_id: 'f2', filename: 'scan.pdf', chunk_count: 0, status: 'processing', progress: [2, 8] }
const queued = { file_id: 'f3', filename: 'scan2.pdf', chunk_count: 0, status: 'processing', progress: null }
const failed = { file_id: 'f4', filename: 'blurry.pdf', chunk_count: 0, status: 'failed', message: 'OCR found no readable text' }

function lib(files, props = {}) {
  const onDocsChanged = vi.fn(() => Promise.resolve())
  const toast = vi.fn()
  render(<Library docsInfo={{ files }} onDocsChanged={onDocsChanged} toast={toast} {...props} />)
  return { onDocsChanged, toast }
}

afterEach(() => vi.clearAllMocks())

describe('docs helpers', () => {
  it('describes each document state', () => {
    expect(docStatus(ready).text).toBe('49 passages')
    expect(docStatus({ ...ready, chunk_count: 1 }).text).toBe('1 passage')
    expect(docStatus(reading)).toMatchObject({ tone: 'busy', text: 'Reading page 3 of 8', progress: 0.25 })
    expect(docStatus(queued)).toMatchObject({ text: 'Queued for OCR', progress: null })
    expect(docStatus(failed)).toMatchObject({ tone: 'bad', text: 'OCR found no readable text' })
    expect(docStatus({ status: 'failed' }).text).toBe('Processing failed')
  })

  it('badges by extension, and scans in progress as OCR', () => {
    expect(docBadge(ready)).toEqual({ type: 'doc', label: 'DOC' })
    expect(docBadge({ filename: 'notes.md', status: 'ready' })).toEqual({ type: 'txt', label: 'MD' })
    expect(docBadge(reading)).toEqual({ type: 'ocr', label: 'OCR' })
    expect(docBadge({ filename: 'scan.pdf', status: 'ready', ocr: true })).toEqual({ type: 'ocr', label: 'OCR' })
  })

  it('accepts the types the backend parses', () => {
    for (const f of ['a.pdf', 'a.DOCX', 'a.txt', 'a.md', 'a.csv', 'a.html', 'a.htm']) expect(isAccepted(f)).toBe(true)
    expect(isAccepted('a.exe')).toBe(false)
  })
})

describe('Library', () => {
  it('lists documents with status and OCR progress', () => {
    lib([ready, reading, queued, failed])
    expect(screen.getByText('49 passages')).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'OCR progress for scan.pdf' })).toHaveAttribute('aria-valuenow', '25')
    expect(screen.getByRole('progressbar', { name: 'OCR progress for scan2.pdf' })).not.toHaveAttribute('aria-valuenow')
    expect(screen.getByText('OCR found no readable text')).toBeInTheDocument()
    expect(screen.getByLabelText('Cancel and remove scan.pdf')).toBeInTheDocument()
  })

  it('says so when the library is empty', () => {
    lib([])
    expect(screen.getByText(/No documents yet/)).toBeInTheDocument()
  })

  it('asks before removing, and can be told to keep the document', () => {
    lib([ready])
    fireEvent.click(screen.getByLabelText('Remove Plan.docx'))
    expect(screen.getByText('Remove from library?')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Keep' }))
    expect(api.apiDeleteFile).not.toHaveBeenCalled()
    expect(screen.getByText('49 passages')).toBeInTheDocument()
  })

  it('removes a document and refreshes the list', async () => {
    api.apiDeleteFile.mockResolvedValue({})
    const { onDocsChanged, toast } = lib([ready])
    fireEvent.click(screen.getByLabelText('Remove Plan.docx'))
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(onDocsChanged).toHaveBeenCalled())
    expect(api.apiDeleteFile).toHaveBeenCalledWith('f1')
    expect(toast).toHaveBeenCalledWith('Plan.docx removed', 'info')
  })

  it('uploads dropped files and rejects unsupported ones', async () => {
    api.apiUpload.mockResolvedValue({ status: 'ok', chunk_count: 12 })
    const { onDocsChanged, toast } = lib([])
    const good = new File(['x'], 'report.md')
    const bad = new File(['x'], 'tool.exe')
    fireEvent.drop(screen.getByRole('list', { name: 'Documents' }), { dataTransfer: { files: [good, bad], types: ['Files'] } })
    expect(toast).toHaveBeenCalledWith('tool.exe: unsupported type', 'err')
    await waitFor(() => expect(onDocsChanged).toHaveBeenCalled())
    expect(api.apiUpload).toHaveBeenCalledWith(good)
    expect(toast).toHaveBeenCalledWith('report.md: 12 passages indexed', 'ok')
  })
})
