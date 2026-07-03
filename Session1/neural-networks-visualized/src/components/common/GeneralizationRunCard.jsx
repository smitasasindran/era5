import Card from "./Card";
import PrimaryButton from "./PrimaryButton";
import StatusBadge from "./StatusBadge";
import StatBlock from "./StatBlock";
import DecisionBoundaryPlot from "./DecisionBoundaryPlot";
import LearningCurvePlot from "./LearningCurvePlot";

function gapToneClass(gap) {
  if (gap === null) return "text-slate-100";
  if (gap >= 0.15) return "text-red-400";
  if (gap >= 0.05) return "text-amber-400";
  return "text-emerald-400";
}

/**
 * Card for one "same model, different training-set size" run: train vs
 * test accuracy/loss, the generalization gap, the decision boundary
 * (with that run's own training points overlaid), and a train-vs-test
 * loss curve. Pairs with `useModelTrainer`'s `validationDataset` option —
 * pass its status/metrics/validationMetrics/boundary/history/train/
 * isTraining straight through.
 */
function GeneralizationRunCard({
  title,
  trainSize,
  dataset,
  status,
  metrics,
  validationMetrics,
  boundary,
  history,
  error,
  onTrain,
  isTraining,
}) {
  const gap = metrics && validationMetrics ? Math.abs(metrics.accuracy - validationMetrics.accuracy) : null;

  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
          <p className="text-xs text-slate-500">{trainSize} training samples</p>
        </div>
        <StatusBadge status={status} />
      </div>

      <div className="mt-4">
        <DecisionBoundaryPlot data={dataset} boundary={boundary} height={260} xLabel="x₁" yLabel="x₂" />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <StatBlock
          label="Train Accuracy"
          value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}
          size="sm"
        />
        <StatBlock
          label="Test Accuracy"
          value={validationMetrics ? `${(validationMetrics.accuracy * 100).toFixed(1)}%` : "—"}
          size="sm"
        />
        <StatBlock label="Train Loss" value={metrics ? metrics.loss.toFixed(4) : "—"} size="sm" />
        <StatBlock
          label="Test Loss"
          value={validationMetrics ? validationMetrics.loss.toFixed(4) : "—"}
          size="sm"
        />
      </div>

      <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Generalization Gap</p>
        <p className={`text-xl font-bold ${gapToneClass(gap)}`}>{gap === null ? "—" : `${(gap * 100).toFixed(1)}%`}</p>
      </div>

      {history && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Learning Curve</p>
          <LearningCurvePlot trainLoss={history.loss} validationLoss={history.val_loss} height={200} />
        </div>
      )}

      {status === "error" && <p className="mt-3 text-sm text-red-400">{error}</p>}

      <PrimaryButton onClick={onTrain} disabled={isTraining} className="mt-4 w-full">
        {isTraining ? "Training…" : "Train Model"}
      </PrimaryButton>
    </Card>
  );
}

export default GeneralizationRunCard;
