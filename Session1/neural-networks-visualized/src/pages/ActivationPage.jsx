import { FiCpu, FiTarget, FiTrendingDown } from "react-icons/fi";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import PrimaryButton from "../components/common/PrimaryButton";
import ScatterPlot from "../components/common/ScatterPlot";
import MetricCard from "../components/common/MetricCard";
import { useActivationExperiment } from "../hooks/useActivationExperiment";

const STATUS_LABELS = {
  idle: "Idle",
  training: "Training…",
  completed: "Completed",
  error: "Error",
};

function ActivationPage() {
  const {
    dataset,
    numPoints,
    setNumPoints,
    noise,
    setNoise,
    regenerateDataset,
    epochs,
    setEpochs,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    status,
    metrics,
    error,
    train,
    isTraining,
  } = useActivationExperiment();

  return (
    <ExperimentLayout
      title="Activation Functions"
      claim="Non-linear activations let networks learn boundaries a linear model cannot."
    >
      <Card title="Overview">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Linear models can only draw straight decision boundaries. The dataset
          below — two concentric, noisy rings — has no straight line that
          separates its two classes, which is exactly the kind of problem
          non-linear activations are built to solve.
        </p>
      </Card>

      <Card title="Interactive Demo">
        <div className="mt-4 flex flex-col gap-6">
          <ScatterPlot data={dataset} />

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Points</span>
                <span className="text-slate-200">{numPoints}</span>
              </span>
              <input
                type="range"
                min={50}
                max={600}
                step={10}
                value={numPoints}
                onChange={(event) => setNumPoints(Number(event.target.value))}
                disabled={isTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Noise</span>
                <span className="text-slate-200">{noise.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.02}
                value={noise}
                onChange={(event) => setNoise(Number(event.target.value))}
                disabled={isTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>

          <PrimaryButton onClick={regenerateDataset} disabled={isTraining} className="self-start">
            Regenerate dataset
          </PrimaryButton>
        </div>
      </Card>

      <Card title="Train a Linear Model">
        <div className="mt-4 flex flex-col gap-6">
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Epochs</span>
                <span className="text-slate-200">{epochs}</span>
              </span>
              <input
                type="range"
                min={10}
                max={300}
                step={10}
                value={epochs}
                onChange={(event) => setEpochs(Number(event.target.value))}
                disabled={isTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Learning rate</span>
                <span className="text-slate-200">{learningRate.toFixed(3)}</span>
              </span>
              <input
                type="range"
                min={0.001}
                max={0.3}
                step={0.001}
                value={learningRate}
                onChange={(event) => setLearningRate(Number(event.target.value))}
                disabled={isTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Batch size</span>
                <span className="text-slate-200">{batchSize}</span>
              </span>
              <input
                type="range"
                min={8}
                max={128}
                step={8}
                value={batchSize}
                onChange={(event) => setBatchSize(Number(event.target.value))}
                disabled={isTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>

          <PrimaryButton onClick={train} disabled={isTraining} className="self-start">
            {isTraining ? "Training…" : "Train Linear Model"}
          </PrimaryButton>

          {status === "error" && (
            <p className="text-sm text-red-400">{error}</p>
          )}

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <MetricCard icon={FiCpu} label="Status" value={STATUS_LABELS[status]} />
            <MetricCard
              icon={FiTrendingDown}
              label="Final Loss"
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

      <Card title="What to Observe">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          A single sigmoid layer is still a linear model in disguise. Train
          it on these rings and watch the accuracy plateau well short of
          100% — no matter how long you train or how small the learning
          rate, a straight-line boundary can't separate one ring from the
          other.
        </p>
      </Card>

      <Card title="Why It Matters">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Every experiment on this page will try to separate these same two
          classes. How well that works depends heavily on the activation
          function a network uses between its layers.
        </p>
      </Card>

      <Card title="Summary">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          This dataset — not linearly separable, but clearly structured — is
          the shared starting point for exploring activation functions in
          this experiment.
        </p>
      </Card>
    </ExperimentLayout>
  );
}

export default ActivationPage;
