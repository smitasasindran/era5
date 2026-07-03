import { FiCheckCircle, FiLoader } from "react-icons/fi";

const STATUS_STYLES = {
  idle: "bg-slate-800 text-slate-400",
  training: "bg-accent-500/15 text-accent-300",
  completed: "bg-emerald-500/15 text-emerald-300",
  error: "bg-red-500/15 text-red-300",
};

const STATUS_LABELS = {
  idle: "Idle",
  training: "Training…",
  completed: "Completed",
  error: "Error",
};

/**
 * Small colored pill for a training run's status ("idle" | "training" |
 * "completed" | "error"), used by any card that trains a model.
 */
function StatusBadge({ status }) {
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLES[status]}`}
    >
      {status === "training" && <FiLoader className="h-3.5 w-3.5 animate-spin" />}
      {status === "completed" && <FiCheckCircle className="h-3.5 w-3.5" />}
      {STATUS_LABELS[status]}
    </span>
  );
}

export default StatusBadge;
