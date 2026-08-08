/**
 * ArchivaLogo.jsx — Shared brand mark, three stacked parallelogram "pages"
 * in solid ink-gold. Fully original, no trademark conflicts.
 *
 * Used in the header (as the primary brand mark) and as the assistant
 * avatar in chat (replacing a generic sparkle/star icon) so the two
 * reinforce the same identity instead of one being custom and the other
 * a stock icon-library glyph.
 */
export default function ArchivaLogo({ size = 34, opacity = 1 }) {
  return (
    <svg
      className="archiva-logo"
      width={size}
      height={size}
      viewBox="0 0 38 38"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Archiva logo"
      style={{ opacity }}
    >
      {/* Bottom page (farthest back) */}
      <rect x="6" y="22" width="22" height="10" rx="3" fill="var(--accent)" opacity="0.35" />

      {/* Middle page */}
      <rect x="4" y="15" width="22" height="10" rx="3" fill="var(--accent)" opacity="0.65" />

      {/* Top page (foreground) */}
      <rect x="2" y="8" width="22" height="10" rx="3" fill="var(--accent)" />
    </svg>
  )
}
