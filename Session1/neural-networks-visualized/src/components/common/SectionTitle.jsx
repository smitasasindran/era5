/**
 * Reusable heading block: small eyebrow label + title + optional description.
 */
function SectionTitle({ eyebrow, title, description, className = "" }) {
  return (
    <div className={`max-w-2xl ${className}`}>
      {eyebrow && (
        <p className="text-xs font-semibold uppercase tracking-widest text-accent-400">
          {eyebrow}
        </p>
      )}
      <h2 className="mt-2 text-2xl font-bold text-slate-50 sm:text-3xl">{title}</h2>
      {description && <p className="mt-3 text-base text-slate-400">{description}</p>}
    </div>
  );
}

export default SectionTitle;
