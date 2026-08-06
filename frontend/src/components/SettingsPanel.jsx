export default function SettingsPanel({ isOpen, docsInfo, onClose, onClearChat }) {
  const { files = [], total_files = 0, total_chunks = 0 } = docsInfo

  return (
    <aside className={`settings-panel ${isOpen ? 'settings-panel--open' : ''}`}>
      <div className="settings-header">
        <span className="settings-title">⚙ Settings</span>
        <button className="icon-btn" onClick={onClose} aria-label="Close settings">✕</button>
      </div>

      <section className="settings-section">
        <h3 className="settings-section-title">Documents</h3>
        {files.length === 0 ? (
          <p className="settings-empty">No documents loaded.<br />Run <code>python load_docs.py</code></p>
        ) : (
          <ul className="file-list">
            {files.map((f, i) => (
              <li key={i} className="file-item">
                <div className="file-icon">📄</div>
                <div className="file-info">
                  <span className="file-name">{f.filename}</span>
                  <span className="file-chunks">{f.chunk_count} chunks</span>
                </div>
              </li>
            ))}
          </ul>
        )}
        <div className="settings-stat-row">
          <span className="stat-label">Total files</span>
          <span className="stat-value">{total_files}</span>
        </div>
        <div className="settings-stat-row">
          <span className="stat-label">Total chunks</span>
          <span className="stat-value">{total_chunks}</span>
        </div>
      </section>

      <section className="settings-section">
        <h3 className="settings-section-title">Models</h3>
        <div className="settings-stat-row">
          <span className="stat-label">Fast</span>
          <span className="stat-value stat-value--mono">gpt-oss-20b</span>
        </div>
        <div className="settings-stat-row">
          <span className="stat-label">Strong</span>
          <span className="stat-value stat-value--mono">gpt-oss-120b</span>
        </div>
        <div className="settings-stat-row">
          <span className="stat-label">Safety</span>
          <span className="stat-value stat-value--mono">qwen3-32b</span>
        </div>
        <div className="settings-stat-row">
          <span className="stat-label">Embeddings</span>
          <span className="stat-value stat-value--mono">text-embedding-004</span>
        </div>
      </section>

      <section className="settings-section">
        <h3 className="settings-section-title">Session</h3>
        <button className="clear-btn" onClick={onClearChat}>
          🗑 Clear Chat History
        </button>
      </section>
    </aside>
  )
}
