import { useLocation } from "react-router-dom";
import { FiMenu } from "react-icons/fi";

// Maps each route to the title shown in the top bar.
const pageTitles = {
  "/": "Home",
  "/activations": "Activation Functions",
  "/depth": "Network Depth",
  "/embeddings": "Embeddings",
  "/generalization": "Generalization",
};

/**
 * Sticky top navbar. Shows a hamburger button on mobile (to open the
 * sidebar drawer) and the current page's title.
 */
function Navbar({ onMenuClick }) {
  const { pathname } = useLocation();
  const title = pageTitles[pathname] ?? "Neural Networks Visualized";

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-slate-800 bg-slate-950/80 px-4 backdrop-blur-sm sm:px-6 lg:px-8">
      <button
        type="button"
        onClick={onMenuClick}
        className="rounded-md p-2 text-slate-400 hover:bg-slate-800 hover:text-slate-100 md:hidden"
        aria-label="Open menu"
      >
        <FiMenu className="h-5 w-5" />
      </button>

      <h1 className="text-sm font-semibold text-slate-200 sm:text-base">{title}</h1>

      <span className="ml-auto hidden rounded-full border border-accent-500/30 bg-accent-500/10 px-3 py-1 text-xs font-medium text-accent-300 sm:inline-block">
        ERA V5 · Session 1
      </span>
    </header>
  );
}

export default Navbar;
