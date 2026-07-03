import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { VOCABULARY, buildCorpus, bigramsToIndices } from "../utils/languageCorpus";
import {
  createEmbeddingModel,
  bigramIndicesToTensors,
  trainEmbeddingModel,
  extractEmbeddings,
} from "../ml/embeddingModel";
import { disposeTensors } from "../ml/binaryClassifier";
import { projectTo2DContinuous } from "../ml/pca";

const DEFAULT_NUM_SENTENCES = 60;
const DEFAULT_EPOCHS = 150;
const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_BATCH_SIZE = 16;
const EMBEDDING_DIM = 4;

// How many snapshots to capture across a training run, at most. Capping
// this keeps the animation smooth (bounded re-renders) regardless of how
// many epochs the user configures.
const MAX_SNAPSHOTS = 60;

// Each captured frame gets a small artificial pause. Training itself
// finishes in well under a second at this model's size, which is too fast
// to actually watch the embeddings move — this deliberately paces the
// animation to a speed a person can follow, purely for visibility.
const FRAME_PACING_MS = 40;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Controller hook for the Embeddings experiment. Generates a synthetic
 * sentence corpus, trains a tiny Embedding → Softmax next-token-prediction
 * model on it, and captures the embedding space's 2D (PCA) projection at
 * a sequence of points throughout training — not just before/after — so
 * the page can animate words drifting into their category clusters as
 * training progresses. Mirrors the shape of `useModelTrainer` (status
 * machine, duplicate-run guard, tensor/model disposal) but is otherwise a
 * separate pipeline since next-token prediction is a different ML task
 * from binary classification.
 */
export function useEmbeddingExperiment() {
  // --- Shared corpus -------------------------------------------------
  const [numSentences, setNumSentences] = useState(DEFAULT_NUM_SENTENCES);
  const [corpusVersion, setCorpusVersion] = useState(0);

  const corpus = useMemo(
    () => buildCorpus(numSentences),
    // corpusVersion has no value of its own — bumping it forces a fresh
    // random draw with the same numSentences.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numSentences, corpusVersion],
  );

  const regenerateCorpus = useCallback(() => {
    setCorpusVersion((version) => version + 1);
  }, []);

  // --- Hyperparameters ---------------------------------------------------
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- Training lifecycle ------------------------------------------------
  const [status, setStatus] = useState("idle"); // "idle" | "training" | "completed" | "error"
  const [metrics, setMetrics] = useState(null); // { loss, accuracy }
  // Sequence of { epoch, points } snapshots — points[i] is the 2D position
  // of VOCABULARY[i] at that epoch. embeddingHistory[0] is epoch 0 (random
  // init); the last entry is the final trained state.
  const [embeddingHistory, setEmbeddingHistory] = useState([]);
  const [rawEmbeddings, setRawEmbeddings] = useState(null); // full-dim, for nearest-neighbor lookups
  const [error, setError] = useState(null);

  const modelRef = useRef(null);
  const isTrainingRef = useRef(false);
  const isMountedRef = useRef(true);

  const disposeModel = useCallback(() => {
    modelRef.current?.dispose();
    modelRef.current = null;
  }, []);

  const reset = useCallback(() => {
    disposeModel();
    setStatus("idle");
    setMetrics(null);
    setEmbeddingHistory([]);
    setRawEmbeddings(null);
    setError(null);
  }, [disposeModel]);

  // Any config change (new corpus or hyperparameters) makes a previously
  // trained model and its embeddings stale.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reset, [corpus, epochs, learningRate, batchSize]);

  // Free the model on unmount. If a training run is still in flight, ask it
  // to stop at the next epoch boundary instead of disposing under it.
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      disposeModel();
    };
  }, [disposeModel]);

  const train = useCallback(async () => {
    if (isTrainingRef.current) return; // guard against duplicate/overlapping runs
    isTrainingRef.current = true;

    disposeModel();
    setStatus("training");
    setError(null);
    setEmbeddingHistory([]);

    const model = createEmbeddingModel({ vocabSize: VOCABULARY.length, embeddingDim: EMBEDDING_DIM });
    modelRef.current = model;

    const history = [];
    let previousBasis = null;

    const captureFrame = async (epoch) => {
      const raw = extractEmbeddings(model);
      const { points, basis } = projectTo2DContinuous(raw, previousBasis);
      previousBasis = basis;
      history.push({ epoch, points });
      if (isMountedRef.current) setEmbeddingHistory([...history]);
      await sleep(FRAME_PACING_MS);
    };

    // Epoch 0: capture the untrained (randomly initialized) embedding
    // space, so the animation starts from a genuine "word soup".
    await captureFrame(0);

    const snapshotInterval = Math.max(1, Math.round(epochs / MAX_SNAPSHOTS));

    const bigramIndices = bigramsToIndices(corpus.bigrams);
    const { xs, ys } = bigramIndicesToTensors(bigramIndices, VOCABULARY.length);

    try {
      const { finalMetrics } = await trainEmbeddingModel(model, xs, ys, {
        epochs,
        learningRate,
        batchSize,
        onEpochEnd: async (epoch) => {
          if (!isMountedRef.current) {
            model.stopTraining = true;
            return;
          }
          const isLastEpoch = epoch === epochs - 1;
          if ((epoch + 1) % snapshotInterval === 0 || isLastEpoch) {
            await captureFrame(epoch + 1);
          }
        },
      });

      if (isMountedRef.current) {
        setMetrics(finalMetrics);
        setRawEmbeddings(extractEmbeddings(model));
        setStatus("completed");
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err.message ?? "Training failed.");
        setStatus("error");
      }
    } finally {
      disposeTensors({ xs, ys });
      isTrainingRef.current = false;
    }
  }, [corpus, epochs, learningRate, batchSize, disposeModel]);

  return {
    // corpus
    corpus,
    numSentences,
    setNumSentences,
    regenerateCorpus,
    // hyperparameters
    epochs,
    setEpochs,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    // training
    status,
    metrics,
    embeddingHistory,
    rawEmbeddings,
    error,
    train,
    reset,
    isTraining: status === "training",
  };
}
