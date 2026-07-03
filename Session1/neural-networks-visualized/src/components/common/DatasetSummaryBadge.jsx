import { FiDatabase } from "react-icons/fi";

/**
 * Compact summary shown next to an experiment page's claim banner —
 * a few short description lines under a title, e.g. dataset size and how
 * it was generated. Generic across chapters (point datasets, sentence
 * corpora, etc.) — pass whatever lines make sense for the experiment.
 */
function DatasetSummaryBadge({ title = "Dataset", lines, icon: Icon = FiDatabase }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 lg:w-72">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-500/10 text-accent-400">
        <Icon className="h-4.5 w-4.5" />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-100">{title}</p>
        {lines.map((line) => (
          <p key={line} className="text-xs text-slate-500">
            {line}
          </p>
        ))}
      </div>
    </div>
  );
}

export default DatasetSummaryBadge;
