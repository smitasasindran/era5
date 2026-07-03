import { FiCheck, FiHelpCircle, FiRefreshCw, FiTrash2 } from "react-icons/fi";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import PrimaryButton from "../components/common/PrimaryButton";
import SecondaryButton from "../components/common/SecondaryButton";
import SliderField from "../components/common/SliderField";
import IconList from "../components/common/IconList";
import DatasetSummaryBadge from "../components/common/DatasetSummaryBadge";
import GeneralizationRunCard from "../components/common/GeneralizationRunCard";
import GeneralizationGapChart from "../components/common/GeneralizationGapChart";
import ComparisonTable from "../components/common/ComparisonTable";
import { useGeneralizationExperiment } from "../hooks/useGeneralizationExperiment";

function gapOf(run) {
  return run.metrics && run.validationMetrics
    ? Math.abs(run.metrics.accuracy - run.validationMetrics.accuracy)
    : null;
}

function formatPercent(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function GeneralizationPage() {
  const {
    testSet,
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
    trainSizes,
    small,
    medium,
    large,
    trainAll,
    isAnyTraining,
  } = useGeneralizationExperiment();

  const runs = [
    { key: "small", title: "Tiny Training Set", trainSize: trainSizes[0], trainer: small },
    { key: "medium", title: "Medium Training Set", trainSize: trainSizes[1], trainer: medium },
    { key: "large", title: "Large Training Set", trainSize: trainSizes[2], trainer: large },
  ];

  const gapChartRuns = runs
    .map((run) => ({ trainSize: run.trainSize, gap: gapOf(run.trainer) }))
    .filter((run) => run.gap !== null);

  return (
    <ExperimentLayout
      title="Why More Data Beats Bigger Models"
      claim="Does a bigger model always learn better, or does it just memorize more? A sufficiently powerful network can drive training loss to nearly zero on a handful of examples while doing barely better than chance on data it hasn't seen. We train the exact same over-parameterized network three times, changing only how much data it gets, and watch the gap between memorizing and understanding close."
      headerActions={
        <DatasetSummaryBadge lines={["20 / 200 / 2000 train samples", "300 test samples (fixed)"]} />
      }
    >
      <Card title="Overview">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          A neural network with far more parameters than training examples has more than enough
          flexibility to draw a decision boundary through every single point it's shown — including
          the noisy, outlier-looking ones. When that happens, training loss drops close to zero, but
          the model hasn't learned the underlying pattern. It has memorized the specific dots it saw.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          You can only tell the difference by checking performance on data the model never saw
          during training — the entire reason for a held-out test set. A big gap between training
          accuracy and test accuracy is the signature of memorization; a small gap means the model
          learned something that actually generalizes.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          The model and every hyperparameter below stay exactly the same across all three runs —
          only the number of training examples changes. With 20 points, the boundary can contort
          itself around every single dot. With 2000, there are simply too many overlapping, noisy
          examples to memorize individually, so the network falls back on the one thing that fits
          all of them reasonably well: the true underlying ring pattern.
        </p>
      </Card>

      <Card title="Dataset & Training Settings">
        <div className="mt-4 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="grid flex-1 grid-cols-2 gap-5 sm:grid-cols-4">
            <SliderField
              label="Noise"
              value={noise}
              displayValue={noise.toFixed(2)}
              onChange={setNoise}
              min={0}
              max={1}
              step={0.02}
              disabled={isAnyTraining}
            />
            <SliderField
              label="Epochs"
              value={epochs}
              displayValue={epochs}
              onChange={setEpochs}
              min={10}
              max={300}
              step={10}
              disabled={isAnyTraining}
            />
            <SliderField
              label="Learning rate"
              value={learningRate}
              displayValue={learningRate.toFixed(3)}
              onChange={setLearningRate}
              min={0.001}
              max={0.3}
              step={0.001}
              disabled={isAnyTraining}
            />
            <SliderField
              label="Batch size"
              value={batchSize}
              displayValue={batchSize}
              onChange={setBatchSize}
              min={4}
              max={64}
              step={4}
              disabled={isAnyTraining}
            />
          </div>

          <div className="flex shrink-0 flex-col gap-2 lg:w-48">
            <SecondaryButton icon={FiRefreshCw} onClick={regenerateDataset} disabled={isAnyTraining}>
              Regenerate Dataset
            </SecondaryButton>
            <SecondaryButton icon={FiTrash2} onClick={resetResults} disabled={isAnyTraining}>
              Reset Results
            </SecondaryButton>
          </div>
        </div>

        <p className="mt-3 text-xs text-slate-500">
          All three runs below use the exact same model, the exact same settings above, and the
          exact same {testSet.length}-sample test set. Only the training-set size changes.
        </p>
      </Card>

      <PrimaryButton onClick={trainAll} disabled={isAnyTraining} className="self-start">
        {isAnyTraining ? "Training…" : "Train All Runs"}
      </PrimaryButton>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {runs.map((run) => (
          <GeneralizationRunCard
            key={run.key}
            title={run.title}
            trainSize={run.trainSize}
            dataset={run.trainer.dataset ?? []}
            status={run.trainer.status}
            metrics={run.trainer.metrics}
            validationMetrics={run.trainer.validationMetrics}
            boundary={run.trainer.boundary}
            history={run.trainer.history}
            error={run.trainer.error}
            onTrain={run.trainer.train}
            isTraining={run.trainer.isTraining}
          />
        ))}
      </div>

      <Card title="Generalization Gap">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          The chart below is this chapter's whole argument in one picture: the same
          over-parameterized model, trained on progressively more data, shows a generalization gap
          that shrinks dramatically as the training set grows.
        </p>
        {gapChartRuns.length >= 2 ? (
          <div className="mt-4">
            <GeneralizationGapChart runs={gapChartRuns} />
          </div>
        ) : (
          <div className="mt-4 flex h-40 items-center justify-center rounded-lg border border-dashed border-slate-800 text-sm text-slate-500">
            Train at least two runs to see the gap shrink.
          </div>
        )}
      </Card>

      <Card title="Comparison">
        <ComparisonTable
          columns={[
            { key: "size", label: "Training Set Size" },
            { key: "trainAcc", label: "Train Accuracy" },
            { key: "testAcc", label: "Test Accuracy" },
            { key: "gap", label: "Generalization Gap", align: "right" },
          ]}
          rows={runs.map((run) => ({
            id: run.key,
            size: run.trainSize,
            trainAcc: formatPercent(run.trainer.metrics?.accuracy),
            testAcc: formatPercent(run.trainer.validationMetrics?.accuracy),
            gap: formatPercent(gapOf(run.trainer)),
          }))}
        />
        <p className="mt-3 text-xs text-slate-500">
          <span className="font-semibold text-emerald-400">Result:</span> The 20-sample and
          2000-sample runs use the identical model and settings — only the data changed, and the
          gap tells the whole story.
        </p>
      </Card>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Card title="What to Observe">
          <IconList
            icon={FiCheck}
            items={[
              "Tiny datasets are easily memorized.",
              "Training loss approaches zero while test performance remains poor.",
              "Increasing the amount of data reduces overfitting.",
              "The model begins learning the true underlying pattern.",
            ]}
          />
        </Card>

        <Card title="Why It Matters">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Modern AI systems often succeed not because they use enormous neural networks, but
            because they're trained on enormous datasets. Scaling up data is frequently a more
            reliable path to a model that generalizes than scaling up parameters alone.
          </p>
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
            More data doesn't just help — for a fixed model, it's often the difference between
            memorizing and actually learning.
          </p>
        </Card>

        <Card title="Key Takeaway">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            A larger model increases learning capacity. More data improves understanding.
          </p>
          <p className="mt-3 text-sm font-semibold text-accent-300">
            Generalization comes from learning the underlying distribution rather than memorizing
            individual examples.
          </p>
        </Card>
      </div>

      <Card title="Challenge Yourself">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Before you touch a slider, predict what will happen — then check if you were right.
        </p>
        <IconList
          icon={FiHelpCircle}
          iconClassName="text-accent-400"
          items={[
            "Increase the dataset noise — how much worse does the 20-sample run get?",
            "Reduce the model size — does the tiny dataset still get memorized as easily?",
            "Train for more epochs — can 2000 samples eventually be memorized too?",
            "Try different train/test splits by regenerating the dataset a few times.",
          ]}
        />
      </Card>
    </ExperimentLayout>
  );
}

export default GeneralizationPage;
