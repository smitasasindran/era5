import { NavLink } from "react-router-dom";
import { FiHome, FiActivity, FiLayers, FiBox, FiTrendingUp, FiX } from "react-icons/fi";

// Central place to define the app's navigation structure.
const navItems = [
  { to: "/", label: "Home", icon: FiHome, end: true },
  { to: "/activations", label: "Activation Functions", icon: FiActivity },
  { to: "/depth", label: "Network Depth", icon: FiLayers },
  { to: "/embeddings", label: "Embeddings", icon: FiBox },
  { to: "/generalization", label: "Generalization", icon: FiTrendingUp },
];

function NavItem({ to, label, icon: Icon, end, onNavigate }) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onNavigate}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
          isActive
            ? "bg-accent-500/15 text-accent-300"
            : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-100"
        }`
      }
    >
      <Icon className="h-5 w-5 shrink-0" />
      {label}
    </NavLink>
  );
}

/**
 * Left navigation sidebar. Fixed and always visible on desktop (md+);
 * becomes a slide-in drawer with a backdrop on smaller screens, controlled
 * by `isOpen`/`onClose` from the parent layout.
 */
function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/70 backdrop-blur-sm md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-slate-800 bg-slate-900/95 transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <NavLink to="/" className="flex items-center gap-2.5" onClick={onClose}>
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-sm font-bold tracking-tight text-slate-100">
              NN Visualized
            </span>
          </NavLink>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100 md:hidden"
            aria-label="Close menu"
          >
            <FiX className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-2">
          {navItems.map((item) => (
            <NavItem key={item.to} {...item} onNavigate={onClose} />
          ))}
        </nav>

        <div className="border-t border-slate-800 px-5 py-4 text-xs text-slate-500">
          ERA V5 · Session 1 Assignment
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
