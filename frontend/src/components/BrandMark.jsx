/**
 * BrandMark.jsx — the Archiva mark: a bound book on a warm gradient tile.
 * Used in the library header and (from the conversation stage on) as the
 * answer avatar, so the two read as the same identity.
 */
export default function BrandMark({ className = '' }) {
  return (
    <span className={`mark ${className}`.trim()} aria-hidden="true">
      <svg className="ico" viewBox="0 0 24 24">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
        <path d="M8 7h8M8 11h6" />
      </svg>
    </span>
  )
}
