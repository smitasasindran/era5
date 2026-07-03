import { Link } from "react-router-dom";

/**
 * Generic surface used across the app (experiment cards, content panels, etc).
 * Renders as a <Link> when `to` is provided, otherwise a plain <div>.
 */
function Card({ to, icon: Icon, title, description, children, className = "" }) {
  const baseClasses = `group relative flex flex-col rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-sm backdrop-blur-sm transition-all duration-200 ${className}`;

  const content = (
    <>
      {Icon && (
        <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-lg bg-accent-500/10 text-accent-400 ring-1 ring-accent-500/20 transition-colors group-hover:bg-accent-500/15 group-hover:text-accent-300">
          <Icon className="h-5 w-5" />
        </div>
      )}
      {title && <h3 className="text-lg font-semibold text-slate-100">{title}</h3>}
      {description && (
        <p className="mt-2 text-sm leading-relaxed text-slate-400">{description}</p>
      )}
      {children}
    </>
  );

  if (to) {
    return (
      <Link
        to={to}
        className={`${baseClasses} hover:-translate-y-0.5 hover:border-accent-500/40 hover:shadow-lg hover:shadow-accent-950/40`}
      >
        {content}
      </Link>
    );
  }

  return <div className={baseClasses}>{content}</div>;
}

export default Card;
