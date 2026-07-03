import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { VOCABULARY, buildCorpus, bigramsToIndices } from "../utils/languageCorpus";
import {
  createEmbeddingModel,
  bigramIndicesToTensors,
  trainEmbeddingModel,
  extractEmbeddings,
} from "../ml/embeddingModel";
import { disposeTensors } from "../ml/binaryClassifier";
import { projectTo2D } from "../ml/pca";

const DEFAULT_NUM_SENTENCES = 60;
const DEFAULT_EPOCHS = 150;
const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_BATCH_SIZE = 16;
const EMBEDDING_DIM = 4;

/**
 * Controller hook for the Embeddings experiment. Generates a synthetic
 * sentence corpus, trains a tiny Embedding → Softmax next-token-prediction
 * model on it, and exposes both the pre-training (randomly initialized)
 * and post-training embedding spaces — projected to 2D via PCA — so the
 * page can show the "before vs after" comparison that is this chapter's
 * proof. Mirrors the shape of `useModelTrainer` (status machine,
 * duplicate-run guard, tensor/model disposal) but is otherwise a separate
 * pipeline since next-token prediction is a different ML task from binary
 * classification.
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
  const [initialEmbeddings2D, setInitialEmbeddings2D] = useState(null); // before training
  const [embeddings2D, setEmbeddings2D] = useState(null); // after training
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
    setInitialEmbeddings2D(null);
    setEmbeddings2D(null);
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

    const model = createEmbeddingModel({ vocabSize: VOCABULARY.length, embeddingDim: EMBEDDING_DIM });
    modelRef.current = model;

    // Capture the untrained (randomly initialized) embedding space first,
    // so the UI can show a genuine before-vs-after comparison.
    const initialRaw = extractEmbeddings(model);
    const initial2D = projectTo2D(initialRaw);

    const bigramIndices = bigramsToIndices(corpus.bigrams);
    const { xs, ys } = bigramIndicesToTensors(bigramIndices, VOCABULARY.length);

    try {
      const { finalMetrics } = await trainEmbeddingModel(model, xs, ys, {
        epochs,
        learningRate,
        batchSize,
        onEpochEnd: () => {
          if (!isMountedRef.current) {
            model.stopTraining = true;
          }
        },
      });

      const trainedRaw = extractEmbeddings(model);
      const trained2D = projectTo2D(trainedRaw);

      if (isMountedRef.current) {
        setMetrics(finalMetrics);
        setInitialEmbeddings2D(initial2D);
        setEmbeddings2D(trained2D);
        setRawEmbeddings(trainedRaw);
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
    initialEmbeddings2D,
    embeddings2D,
    rawEmbeddings,
    error,
    train,
    reset,
    isTraining: status === "training",
  };
}
