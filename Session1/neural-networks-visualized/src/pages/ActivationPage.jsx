import { FiCheck, FiShare2, FiTrendingUp } from "react-icons/fi";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import DatasetSummaryBadge from "../components/common/DatasetSummaryBadge";
import DatasetTrainingSettingsCard from "../components/common/DatasetTrainingSettingsCard";
import TrainableModelCard from "../components/common/TrainableModelCard";
import ComparisonTable from "../components/common/ComparisonTable";
import IconList from "../components/common/IconList";
import { useActivationExperiment } from "../hooks/useActivationExperiment";

function ActivationPage() {
  const {
    dataset,
    numPoints,
    setNumPoints,
    noise,
    setNoise,
    regenerateDataset,
    resetResults,
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
      claim="Non-linear activations let networks learn boundaries a linear model cannot. We train two models on the exact same dataset of concentric rings and compare what they learn."
      headerActions={<DatasetSummaryBadge numPoints={numPoints} />}
    >
      <DatasetTrainingSettingsCard
        numPoints={numPoints}
        setNumPoints={setNumPoints}
        noise={noise}
        setNoise={setNoise}
        epochs={epochs}
        setEpochs={setEpochs}
        learningRate={learningRate}
        setLearningRate={setLearningRate}
        batchSize={batchSize}
        setBatchSize={setBatchSize}
        onRegenerateDataset={regenerateDataset}
        onResetResults={resetResults}
        disabled={isAnyTraining}
        footnote="Both models use the same settings above for a fair comparison."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <TrainableModelCard
          icon={FiTrendingUp}
          title="Linear Model"
          architecture="Dense(1) → Sigmoid"
          insight="The linear model can only produce a single straight decision boundary."
          dataset={dataset}
          status={linear.status}
          metrics={linear.metrics}
          boundary={linear.boundary}
          error={linear.error}
          onTrain={linear.train}
          isTraining={linear.isTraining}
        />

        <TrainableModelCard
          icon={FiShare2}
          title="ReLU Network"
          architecture="Dense(8) → ReLU → Dense(1) → Sigmoid"
          insight="The ReLU network learns a nonlinear boundary that separates the rings."
          dataset={dataset}
          status={relu.status}
          metrics={relu.metrics}
          boundary={relu.boundary}
          error={relu.error}
          onTrain={relu.train}
          isTraining={relu.isTraining}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <Card title="Comparison">
          <ComparisonTable
            rows={[
              {
                model: "Linear",
                boundary: "Straight line",
                accuracy: linear.metrics ? `${(linear.metrics.accuracy * 100).toFixed(1)}%` : "—",
              },
              {
                model: "ReLU",
                boundary: "Nonlinear curve",
                accuracy: relu.metrics ? `${(relu.metrics.accuracy * 100).toFixed(1)}%` : "—",
              },
            ]}
          />
          <p className="mt-3 text-xs text-slate-500">
            <span className="font-semibold text-emerald-400">Result:</span> Non-linearity makes all
            the difference.
          </p>
        </Card>

        <Card title="What to Observe">
          <IconList
            icon={FiCheck}
            items={[
              "Linear model fails: it misclassifies many points because a line can't separate the rings.",
              "ReLU model succeeds: it forms a closed boundary around the inner ring.",
              "Same data, same settings, different outcomes due to non-linearity.",
            ]}
          />
        </Card>

        <Card title="Why It Matters">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Real-world problems are rarely linearly separable. Non-linear activations (like ReLU,
            tanh, etc.) let neural networks model complex, nonlinear relationships.
          </p>
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
            This is the key reason why deep learning works so well.
          </p>
        </Card>

        <Card title="Key Takeaway">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            A single linear transformation can only draw a straight boundary. Adding a non-linear
            activation like ReLU lets the network bend that boundary and solve problems linear
            models cannot.
          </p>
          <p className="mt-3 text-sm font-semibold text-accent-300">
            Non-linearity = Expressive Power
          </p>
        </Card>
      </div>
    </ExperimentLayout>
  );
}

export default ActivationPage;
