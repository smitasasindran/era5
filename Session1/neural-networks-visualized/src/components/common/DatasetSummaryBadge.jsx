import { FiDatabase } from "react-icons/fi";

/**
 * Compact "Dataset" summary shown next to an experiment page's claim
 * banner: point count, class count, and a short description of how the
 * data was generated.
 */
function DatasetSummaryBadge({ numPoints, description = "Concentric rings with noise" }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 lg:w-72">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-500/10 text-accent-400">
        <FiDatabase className="h-4.5 w-4.5" />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-100">Dataset</p>
        <p className="text-xs text-slate-500">{numPoints} points · 2 classes</p>
        <p className="text-xs text-slate-500">{description}</p>
      </div>
    </div>
  );
}

export default DatasetSummaryBadge;
