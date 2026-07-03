import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import PrimaryButton from "../components/common/PrimaryButton";
import ScatterPlot from "../components/common/ScatterPlot";
import MetricCard from "../components/common/MetricCard";
import TrainableModelCard from "../components/common/TrainableModelCard";
import { useActivationExperiment } from "../hooks/useActivationExperiment";

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
    linear,
    relu,
    isAnyTraining,
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
          non-linear activations are built to solve. Train a linear model and a
          ReLU network on the same data below and compare what each one learns.
        </p>
      </Card>

      <Card title="Dataset">
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
                disabled={isAnyTraining}
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
                disabled={isAnyTraining}
                className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>

          <PrimaryButton onClick={regenerateDataset} disabled={isAnyTraining} className="self-start">
            Regenerate dataset
          </PrimaryButton>

          <p className="text-xs text-slate-500">
            This exact dataset — {numPoints} points — is shared by both models
            below so their results are directly comparable.
          </p>
        </div>
      </Card>

      <Card title="Hyperparameters">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Both models below train with these same settings by default, so any
          difference in their results comes from architecture alone.
        </p>
        <div className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-3">
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
              disabled={isAnyTraining}
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
              disabled={isAnyTraining}
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
              disabled={isAnyTraining}
              className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </label>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <TrainableModelCard
          title="Linear Model"
          architecture="Dense(1) → Sigmoid"
          dataset={dataset}
          status={linear.status}
          metrics={linear.metrics}
          boundary={linear.boundary}
          error={linear.error}
          onTrain={linear.train}
          isTraining={linear.isTraining}
        />

        <TrainableModelCard
          title="ReLU Network"
          architecture="Dense(8) → ReLU → Dense(1) → Sigmoid"
          dataset={dataset}
          status={relu.status}
          metrics={relu.metrics}
          boundary={relu.boundary}
          error={relu.error}
          onTrain={relu.train}
          isTraining={relu.isTraining}
        />
      </div>

      <Card title="Comparison">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          The only architectural difference is the addition of a ReLU
          activation layer. The linear model is constrained to a single
          straight decision boundary, while the ReLU network learns a
          nonlinear boundary that separates the concentric rings.
        </p>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <MetricCard
            label="Linear accuracy"
            value={linear.metrics ? `${(linear.metrics.accuracy * 100).toFixed(1)}%` : "—"}
          />
          <MetricCard
            label="ReLU accuracy"
            value={relu.metrics ? `${(relu.metrics.accuracy * 100).toFixed(1)}%` : "—"}
          />
        </div>
      </Card>
    </ExperimentLayout>
  );
}

export default ActivationPage;
