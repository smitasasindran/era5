import { FiCheck, FiHelpCircle, FiLayers, FiTrendingUp, FiZap } from "react-icons/fi";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import DatasetSummaryBadge from "../components/common/DatasetSummaryBadge";
import DatasetTrainingSettingsCard from "../components/common/DatasetTrainingSettingsCard";
import TrainableModelCard from "../components/common/TrainableModelCard";
import ComparisonTable from "../components/common/ComparisonTable";
import IconList from "../components/common/IconList";
import MatrixCollapseDiagram from "../components/common/MatrixCollapseDiagram";
import { useDepthExperiment } from "../hooks/useDepthExperiment";

function DepthPage() {
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
    fiveLinear,
    fiveRelu,
    isAnyTraining,
  } = useDepthExperiment();

  const accuracyOf = (trainer) => (trainer.metrics ? `${(trainer.metrics.accuracy * 100).toFixed(1)}%` : "—");

  return (
    <ExperimentLayout
      title="Is Deeper Always Better?"
      subtitle="Depth Without Nonlinearity is a Lie"
      claim="If we simply make a network deeper, will it automatically become more powerful? No — five linear layers are mathematically equivalent to one. Depth only becomes useful once nonlinear activations separate those layers. We train three models on the exact same dataset to prove it."
      headerActions={<DatasetSummaryBadge numPoints={numPoints} />}
    >
      <Card title="Overview">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          A single Dense layer does one simple thing: it takes every input feature, multiplies it
          by a set of weights, adds them together, and produces a new set of numbers.
          Geometrically, that's just a stretch, a rotation, and a shift of space — nothing more
          elaborate.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          Stack a second Dense layer directly after the first, with nothing in between, and you
          might expect the network to get more flexible. It doesn't. A stretch-and-shift applied to
          another stretch-and-shift is still just a stretch-and-shift — five of them collapse into
          one equivalent transformation, so the network can only ever draw a single straight
          boundary, no matter how many linear layers you stack.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          ReLU breaks this pattern. It looks at every number leaving a layer and zeroes out the
          negative ones. That small "kink" means the transformation is no longer purely
          straight-line — and stacking several of these kinks lets the network bend, fold, and wrap
          its boundary around shapes a straight line never could.
        </p>
      </Card>

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
        footnote="All three models use the same settings above for a fair comparison. Changing the dataset resets every trained model."
      />

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
        <TrainableModelCard
          layout="stacked"
          icon={FiTrendingUp}
          title="Single Linear Layer"
          architecture="Dense(1) → Sigmoid"
          insight="A single linear layer can only draw a straight boundary."
          dataset={dataset}
          status={linear.status}
          metrics={linear.metrics}
          boundary={linear.boundary}
          error={linear.error}
          onTrain={linear.train}
          isTraining={linear.isTraining}
        />

        <TrainableModelCard
          layout="stacked"
          icon={FiLayers}
          title="Five Linear Layers"
          architecture="5 × Dense (no activation) → Sigmoid"
          insight="Five layers, zero activations between them — still just a straight line."
          dataset={dataset}
          status={fiveLinear.status}
          metrics={fiveLinear.metrics}
          boundary={fiveLinear.boundary}
          error={fiveLinear.error}
          onTrain={fiveLinear.train}
          isTraining={fiveLinear.isTraining}
        />

        <TrainableModelCard
          layout="stacked"
          icon={FiZap}
          title="Five Layers + ReLU"
          architecture="4 × (Dense → ReLU) → Dense → Sigmoid"
          insight="ReLU between every layer lets the network bend its boundary around the rings."
          dataset={dataset}
          status={fiveRelu.status}
          metrics={fiveRelu.metrics}
          boundary={fiveRelu.boundary}
          error={fiveRelu.error}
          onTrain={fiveRelu.train}
          isTraining={fiveRelu.isTraining}
        />
      </div>

      <Card title="Comparison">
        <ComparisonTable
          rows={[
            { model: "One Layer", boundary: "Straight line", accuracy: accuracyOf(linear) },
            { model: "Five Linear Layers", boundary: "Straight line", accuracy: accuracyOf(fiveLinear) },
            { model: "Five Layers + ReLU", boundary: "Nonlinear boundary", accuracy: accuracyOf(fiveRelu) },
          ]}
        />
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          The first two models learn almost identical decision boundaries despite one having five
          times as many layers. Only adding ReLU changes what the network is capable of learning.
        </p>
      </Card>

      <Card title="Why This Happens">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Every Dense layer performs a matrix multiplication: it takes the input vector x and
          produces W·x (plus a bias). Chain several linear layers with nothing in between, and
          you're really just multiplying their weight matrices together.
        </p>

        <MatrixCollapseDiagram />

        <p className="text-sm leading-relaxed text-slate-400">
          So passing x through five linear layers is mathematically identical to multiplying the
          five matrices together first and applying the result once — that combined matrix is just
          one larger linear layer:
        </p>
        <div className="mt-3 overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/50 px-4 py-3 font-mono text-xs text-slate-300 sm:text-sm">
          W₅(W₄(W₃(W₂(W₁x))))&nbsp;&nbsp;=&nbsp;&nbsp;(W₅W₄W₃W₂W₁)x
        </div>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          No amount of stacking linear layers can escape this collapse — you need a nonlinearity
          like ReLU between them to stop the chain from folding back into one.
        </p>
      </Card>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Card title="What to Observe">
          <IconList
            icon={FiCheck}
            items={[
              "One linear layer fails.",
              "Five linear layers fail in almost exactly the same way.",
              "ReLU immediately changes the learned boundary.",
            ]}
          />
        </Card>

        <Card title="Why It Matters">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Depth is not what makes deep learning powerful. Nonlinear activation functions create
            expressive models.
          </p>
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
            Without activations, adding more layers accomplishes almost nothing.
          </p>
        </Card>

        <Card title="Key Takeaway">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Depth without nonlinearity is just one larger linear equation. ReLU transforms a stack
            of linear layers into a flexible nonlinear function approximator.
          </p>
          <p className="mt-3 text-sm font-semibold text-accent-300">
            This is why activation functions are essential to deep learning.
          </p>
        </Card>
      </div>

      <Card title="Challenge Yourself">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Head back up to the settings and try to predict the outcome before you hit train:
        </p>
        <IconList
          icon={FiHelpCircle}
          iconClassName="text-accent-400"
          items={[
            "What happens with 10 linear layers instead of 5?",
            "Does increasing the number of neurons per layer help, without ReLU?",
            "What if only one hidden layer has ReLU and the rest stay linear?",
            "Can you predict the shape of a boundary before training the model?",
          ]}
        />
      </Card>
    </ExperimentLayout>
  );
}

export default DepthPage;
