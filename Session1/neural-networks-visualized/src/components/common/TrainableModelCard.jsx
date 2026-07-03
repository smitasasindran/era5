import { FiCpu, FiTarget, FiTrendingDown } from "react-icons/fi";

import Card from "./Card";
import PrimaryButton from "./PrimaryButton";
import MetricCard from "./MetricCard";
import DecisionBoundaryPlot from "./DecisionBoundaryPlot";

const STATUS_LABELS = {
  idle: "Idle",
  training: "Training…",
  completed: "Completed",
  error: "Error",
};

/**
 * Reusable card for a single trainable model: an architecture summary, a
 * Train button, live status/metric readouts, and a scatter plot with its
 * decision boundary overlaid once trained. Pairs directly with
 * `useModelTrainer` — pass its returned status/metrics/boundary/error/
 * train/isTraining straight through.
 */
function TrainableModelCard({
  title,
  architecture,
  dataset,
  status,
  metrics,
  boundary,
  error,
  onTrain,
  isTraining,
}) {
  return (
    <Card title={title}>
      <div className="mt-4 flex flex-col gap-5">
        <p className="text-sm text-slate-400">
          <span className="font-medium text-slate-300">Architecture: </span>
          {architecture}
        </p>

        <DecisionBoundaryPlot data={dataset} boundary={boundary} height={340} />

        <PrimaryButton onClick={onTrain} disabled={isTraining} className="self-start">
          {isTraining ? "Training…" : "Train Model"}
        </PrimaryButton>

        {status === "error" && <p className="text-sm text-red-400">{error}</p>}

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <MetricCard icon={FiCpu} label="Status" value={STATUS_LABELS[status]} />
          <MetricCard
            icon={FiTrendingDown}
            label="Loss"
            value={metrics ? metrics.loss.toFixed(4) : "—"}
          />
          <MetricCard
            icon={FiTarget}
            label="Accuracy"
            value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}
          />
        </div>
      </div>
    </Card>
  );
}

export default TrainableModelCard;
