/**
 * Simple app-wide footer shown beneath the routed page content.
 */
function Footer() {
  return (
    <footer className="border-t border-slate-800 px-4 py-6 text-center text-xs text-slate-500 sm:px-6 lg:px-8">
      <p>
        Built with React, React Router &amp; Tailwind CSS —{" "}
        <span className="text-slate-400">ERA V5, Session 1</span>
      </p>
    </footer>
  );
}

export default Footer;
