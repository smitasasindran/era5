import { Link } from "react-router-dom";

/**
 * Outlined counterpart to PrimaryButton, for secondary actions that
 * shouldn't compete visually with a page's main call to action. Renders as
 * a <Link> when `to` is provided, otherwise as a regular <button>.
 */
function SecondaryButton({
  to,
  onClick,
  type = "button",
  icon: Icon,
  disabled = false,
  children,
  className = "",
}) {
  const classes = `inline-flex items-center justify-center gap-2 rounded-lg border border-accent-500/40 px-5 py-2.5 text-sm font-semibold text-accent-300 transition-colors hover:bg-accent-500/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent ${className}`;

  if (to) {
    return (
      <Link to={to} className={classes}>
        {Icon && <Icon className="h-4 w-4" />}
        {children}
      </Link>
    );
  }

  return (
    <button type={type} onClick={onClick} disabled={disabled} className={classes}>
      {Icon && <Icon className="h-4 w-4" />}
      {children}
    </button>
  );
}

export default SecondaryButton;
