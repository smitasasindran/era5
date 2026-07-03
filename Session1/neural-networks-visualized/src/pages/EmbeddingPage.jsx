import { useEffect, useMemo, useRef, useState } from "react";
import {
  FiCheck,
  FiCpu,
  FiHelpCircle,
  FiPlay,
  FiRefreshCw,
  FiTarget,
  FiTrash2,
  FiTrendingDown,
} from "react-icons/fi";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import MetricCard from "../components/common/MetricCard";
import PrimaryButton from "../components/common/PrimaryButton";
import SecondaryButton from "../components/common/SecondaryButton";
import SliderField from "../components/common/SliderField";
import IconList from "../components/common/IconList";
import DatasetSummaryBadge from "../components/common/DatasetSummaryBadge";
import EmbeddingScatterPlot from "../components/common/EmbeddingScatterPlot";
import { useEmbeddingExperiment } from "../hooks/useEmbeddingExperiment";
import { VOCABULARY, getCategoryOf, toEmbeddingPoints } from "../utils/languageCorpus";
import { findNearestNeighbors } from "../ml/similarity";

const STATUS_LABELS = {
  idle: "Idle",
  training: "Training…",
  completed: "Completed",
  error: "Error",
};

const REPLAY_FRAME_MS = 120;

function PlotPlaceholder({ height = 300 }) {
  return (
    <div
      style={{ height }}
      className="flex items-center justify-center rounded-lg border border-dashed border-slate-800 text-sm text-slate-500"
    >
      Train the model to see this plot.
    </div>
  );
}

function EmbeddingPage() {
  const {
    corpus,
    numSentences,
    setNumSentences,
    regenerateCorpus,
    epochs,
    setEpochs,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    status,
    metrics,
    embeddingHistory,
    rawEmbeddings,
    error,
    train,
    reset,
    isTraining,
  } = useEmbeddingExperiment();

  const [selectedWord, setSelectedWord] = useState("cat");

  // Which frame of embeddingHistory to display: null means "follow the
  // latest frame live"; a number means the user (or the replay animation)
  // has pinned it to a specific epoch.
  const [manualIndex, setManualIndex] = useState(null);
  const [isReplaying, setIsReplaying] = useState(false);
  const replayIntervalRef = useRef(null);

  const latestIndex = embeddingHistory.length - 1;
  const displayedIndex = manualIndex === null ? latestIndex : manualIndex;
  const currentFrame = displayedIndex >= 0 ? embeddingHistory[displayedIndex] : null;

  // A fresh training run should always start out following the live frame.
  useEffect(() => {
    if (isTraining) setManualIndex(null);
  }, [isTraining]);

  useEffect(() => {
    return () => {
      if (replayIntervalRef.current) clearInterval(replayIntervalRef.current);
    };
  }, []);

  const handleReplay = () => {
    if (embeddingHistory.length < 2 || isTraining) return;
    if (replayIntervalRef.current) clearInterval(replayIntervalRef.current);

    setIsReplaying(true);
    setManualIndex(0);
    let frame = 0;
    replayIntervalRef.current = setInterval(() => {
      frame += 1;
      if (frame > latestIndex) {
        clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
        setIsReplaying(false);
        return;
      }
      setManualIndex(frame);
    }, REPLAY_FRAME_MS);
  };

  const neighbors = useMemo(
    () => (rawEmbeddings ? findNearestNeighbors(selectedWord, VOCABULARY, rawEmbeddings, 3) : []),
    [selectedWord, rawEmbeddings],
  );

  const currentPoints = currentFrame ? toEmbeddingPoints(currentFrame.points) : null;
  const beforePoints = embeddingHistory[0] ? toEmbeddingPoints(embeddingHistory[0].points) : null;
  const afterPoints =
    latestIndex >= 0 && status === "completed" ? toEmbeddingPoints(embeddingHistory[latestIndex].points) : null;

  const scrubberDisabled = isTraining || isReplaying || embeddingHistory.length < 2;

  return (
    <ExperimentLayout
      title="How Embeddings Learn Meaning"
      claim="Can a model learn that 'cat' and 'dog' are similar without ever being told so? Yes — if two words tend to appear in the same contexts, gradient descent naturally pulls their embeddings toward each other. We train a tiny next-word predictor on a made-up language and watch categories emerge on their own."
      headerActions={
        <DatasetSummaryBadge
          lines={[`${numSentences} sentences · 8 words`, "3 categories, never labeled"]}
        />
      }
    >
      <Card title="Overview">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          An embedding is just a lookup table: each word in the vocabulary gets its own list of
          numbers (a vector), initialized completely at random. At the start there is nothing
          meaningful about these vectors — "cat" and "mango" are just as likely to start out close
          together as "cat" and "dog".
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          We train a tiny model to do one simple job: given the current word, predict which word
          comes next. To get good at that job, the model has to notice that "cat", "dog", and "cow"
          are always followed by the same kind of word — so it can reuse the same
          "this predicts a verb next" machinery for all three. The cheapest way to reuse that
          machinery is to make their embeddings land in similar places in vector space.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          Nobody ever tells the model "cat and dog are animals" — that structure isn't in the
          training data at all. You'll notice some generated sentences sound a little odd (like
          "cow chase mango") — that's intentional. Our toy language only cares about statistical
          patterns, not real-world meaning, which is exactly the point: the model can't lean on
          meaning it was never given.
        </p>
      </Card>

      <Card title="Corpus & Training Settings">
        <div className="mt-4 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="grid flex-1 grid-cols-2 gap-5 sm:grid-cols-4">
            <SliderField
              label="Sentences"
              value={numSentences}
              displayValue={numSentences}
              onChange={setNumSentences}
              min={10}
              max={150}
              step={10}
              disabled={isTraining}
            />
            <SliderField
              label="Epochs"
              value={epochs}
              displayValue={epochs}
              onChange={setEpochs}
              min={50}
              max={300}
              step={10}
              disabled={isTraining}
            />
            <SliderField
              label="Learning rate"
              value={learningRate}
              displayValue={learningRate.toFixed(3)}
              onChange={setLearningRate}
              min={0.001}
              max={0.3}
              step={0.001}
              disabled={isTraining}
            />
            <SliderField
              label="Batch size"
              value={batchSize}
              displayValue={batchSize}
              onChange={setBatchSize}
              min={4}
              max={64}
              step={4}
              disabled={isTraining}
            />
          </div>

          <div className="flex shrink-0 flex-col gap-2 lg:w-48">
            <SecondaryButton icon={FiRefreshCw} onClick={regenerateCorpus} disabled={isTraining}>
              Regenerate Sentences
            </SecondaryButton>
            <SecondaryButton icon={FiTrash2} onClick={reset} disabled={isTraining}>
              Reset Results
            </SecondaryButton>
          </div>
        </div>

        <p className="mt-3 text-xs text-slate-500">
          Sample sentences: {corpus.sentences.slice(0, 3).map((s) => `"${s.join(" ")}"`).join(", ")}…
        </p>
      </Card>

      <Card title="Trained Embedding Space">
        <div className="mt-4 flex flex-col gap-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <MetricCard icon={FiCpu} label="Status" value={STATUS_LABELS[status]} />
            <MetricCard
              icon={FiTrendingDown}
              label="Loss"
              value={metrics ? metrics.loss.toFixed(4) : "—"}
            />
            <MetricCard
              icon={FiTarget}
              label="Next-word Accuracy"
              value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}
            />
          </div>

          <PrimaryButton onClick={train} disabled={isTraining} className="self-start">
            {isTraining ? "Training…" : "Train Embedding Model"}
          </PrimaryButton>

          {status === "error" && <p className="text-sm text-red-400">{error}</p>}

          {currentPoints ? (
            <EmbeddingScatterPlot
              points={currentPoints}
              selectedWord={selectedWord}
              onSelectWord={setSelectedWord}
              height={440}
            />
          ) : (
            <PlotPlaceholder height={440} />
          )}

          {embeddingHistory.length > 0 && (
            <div className="flex items-center gap-3">
              <SecondaryButton icon={FiPlay} onClick={handleReplay} disabled={scrubberDisabled}>
                Replay
              </SecondaryButton>
              <input
                type="range"
                min={0}
                max={Math.max(latestIndex, 0)}
                step={1}
                value={Math.max(displayedIndex, 0)}
                onChange={(event) => setManualIndex(Number(event.target.value))}
                disabled={scrubberDisabled}
                className="flex-1 accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              />
              <span className="w-28 shrink-0 text-right text-xs text-slate-400">
                Epoch {currentFrame ? currentFrame.epoch : 0} / {epochs}
              </span>
            </div>
          )}

          {isTraining && (
            <p className="text-xs text-slate-500">
              Watch the words drift — at epoch 0 they're scattered randomly, with no notion of
              category at all.
            </p>
          )}

          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
              Click a word (on the plot or below) to inspect it
            </p>
            <div className="flex flex-wrap gap-2">
              {VOCABULARY.map((word) => (
                <button
                  key={word}
                  type="button"
                  onClick={() => setSelectedWord(word)}
                  className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                    word === selectedWord
                      ? "border-accent-400 bg-accent-500/20 text-accent-200"
                      : "border-slate-700 bg-slate-800/60 text-slate-300 hover:border-slate-600"
                  }`}
                >
                  {word}
                </button>
              ))}
            </div>
          </div>

          {rawEmbeddings && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-4 py-3">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Nearest neighbors of "{selectedWord}" ({getCategoryOf(selectedWord)}), based on the
                final trained embeddings
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {neighbors.map((neighbor) => (
                  <span
                    key={neighbor.word}
                    className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-200"
                  >
                    {neighbor.word} <span className="text-slate-500">({neighbor.similarity.toFixed(2)})</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>

      <Card title="Comparison">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Before training, embeddings sit at random positions — no category structure at all.
          After training on nothing but "what word comes next", the same words rearrange
          themselves into three tight, separate clusters.
        </p>
        <div className="mt-4 grid grid-cols-1 gap-5 md:grid-cols-2">
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Before Training
            </p>
            {beforePoints ? <EmbeddingScatterPlot points={beforePoints} height={300} /> : <PlotPlaceholder />}
          </div>
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              After Training
            </p>
            {afterPoints ? <EmbeddingScatterPlot points={afterPoints} height={300} /> : <PlotPlaceholder />}
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          <span className="font-semibold text-emerald-400">Result:</span> Category structure was
          never labeled — it emerged entirely from context statistics.
        </p>
      </Card>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Card title="What to Observe">
          <IconList
            icon={FiCheck}
            items={[
              "Cat, dog, and cow drift toward the same corner of the plot — nothing told the model they're all animals.",
              "Apple and mango end up as each other's nearest neighbor.",
              "Eat, chase, and see cluster too, even though they never sit next to each other in a sentence.",
            ]}
          />
        </Card>

        <Card title="Why It Matters">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            This is exactly how real word embeddings work, just at a scale of billions of words
            instead of eight. Word2Vec, GloVe, and the embedding layers inside every modern
            language model all learn structure the same way — from co-occurrence statistics, not
            dictionaries.
          </p>
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
            Meaning, in a neural network, is a side effect of prediction — not something you have to
            hand-label.
          </p>
        </Card>

        <Card title="Key Takeaway">
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Embeddings are never told which words are similar. Similarity is discovered — it
            emerges because words used in the same way are, statistically, interchangeable from the
            model's point of view.
          </p>
          <p className="mt-3 text-sm font-semibold text-accent-300">
            Context determines meaning — and gradient descent finds it automatically.
          </p>
        </Card>
      </div>

      <Card title="Challenge Yourself">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Retrain a few times and see what holds up, then think about these:
        </p>
        <IconList
          icon={FiHelpCircle}
          iconClassName="text-accent-400"
          items={[
            "What happens to the clusters if you shrink the corpus down to 10 sentences?",
            "Would a 4th category (say, colors) form its own cluster, or blend into the others?",
            "Is 'see' always closer to 'eat' and 'chase', or does that change between runs?",
            "Can you guess a word's category from its position alone, without the color legend?",
          ]}
        />
      </Card>
    </ExperimentLayout>
  );
}

export default EmbeddingPage;
