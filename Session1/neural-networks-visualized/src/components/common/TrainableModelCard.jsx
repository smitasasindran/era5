import { FiInfo } from "react-icons/fi";

import Card from "./Card";
import PrimaryButton from "./PrimaryButton";
import DecisionBoundaryPlot from "./DecisionBoundaryPlot";
import StatusBadge from "./StatusBadge";
import StatBlock from "./StatBlock";

/**
 * Compact card for a single trainable model: icon/title/architecture with a
 * status badge, a decision-boundary plot with Accuracy/Loss stats and the
 * Train button, and a one-line insight footer. Pairs directly with
 * `useModelTrainer` — pass its returned status/metrics/boundary/error/
 * train/isTraining straight through.
 *
 * `layout="split"` (default) puts the plot and stats side by side, used
 * when comparing two models (e.g. Activation Functions). `layout="stacked"`
 * puts the plot on top and a compact stats row below, which reads better
 * when several narrower cards sit side by side (e.g. Network Depth).
 */
function TrainableModelCard({
  icon: Icon,
  title,
  architecture,
  insight,
  dataset,
  status,
  metrics,
  boundary,
  error,
  onTrain,
  isTraining,
  layout = "split",
}) {
  const isStacked = layout === "stacked";
  const accuracyValue = metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—";
  const lossValue = metrics ? metrics.loss.toFixed(4) : "—";
  const trainLabel = isTraining ? "Training…" : "Train Model";

  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          {Icon && (
            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-500/10 text-accent-400 ring-1 ring-accent-500/20">
              <Icon className="h-4.5 w-4.5" />
            </div>
          )}
          <div>
            <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
            <p className="text-xs text-slate-500">Architecture: {architecture}</p>
          </div>
        </div>
        <StatusBadge status={status} />
      </div>

      <div className={isStacked ? "mt-4 flex flex-col gap-4" : "mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3"}>
        <div className={isStacked ? "" : "sm:col-span-2"}>
          <DecisionBoundaryPlot
            data={dataset}
            boundary={boundary}
            height={isStacked ? 260 : 300}
            xLabel="x₁"
            yLabel="x₂"
          />
        </div>

        {isStacked ? (
          <div className="grid grid-cols-3 items-end gap-3">
            <StatBlock label="Accuracy" value={accuracyValue} size="sm" />
            <StatBlock label="Loss" value={lossValue} size="sm" />
            <PrimaryButton onClick={onTrain} disabled={isTraining} className="w-full px-3 py-2 text-xs">
              {trainLabel}
            </PrimaryButton>
          </div>
        ) : (
          <div className="flex flex-col justify-between gap-4">
            <div className="space-y-4">
              <StatBlock label="Accuracy" value={accuracyValue} />
              <StatBlock label="Loss" value={lossValue} />
            </div>
            <PrimaryButton onClick={onTrain} disabled={isTraining} className="w-full">
              {trainLabel}
            </PrimaryButton>
          </div>
        )}
      </div>

      {status === "error" && <p className="mt-3 text-sm text-red-400">{error}</p>}

      {insight && (
        <p className="mt-4 flex items-start gap-2 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs leading-relaxed text-slate-400">
          <FiInfo className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent-400" />
          {insight}
        </p>
      )}
    </Card>
  );
}

export default TrainableModelCard;
