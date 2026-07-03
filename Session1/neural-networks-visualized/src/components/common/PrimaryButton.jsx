import { Link } from "react-router-dom";

/**
 * Reusable call-to-action button. Renders as a <Link> when `to` is provided,
 * otherwise as a regular <button>.
 */
function PrimaryButton({
  to,
  onClick,
  type = "button",
  icon: Icon,
  disabled = false,
  children,
  className = "",
}) {
  const classes = `inline-flex items-center justify-center gap-2 rounded-lg bg-accent-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm shadow-accent-900/40 transition-colors hover:bg-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-accent-600 ${className}`;

  if (to) {
    return (
      <Link to={to} className={classes}>
        {children}
        {Icon && <Icon className="h-4 w-4" />}
      </Link>
    );
  }

  return (
    <button type={type} onClick={onClick} disabled={disabled} className={classes}>
      {children}
      {Icon && <Icon className="h-4 w-4" />}
    </button>
  );
}

export default PrimaryButton;
