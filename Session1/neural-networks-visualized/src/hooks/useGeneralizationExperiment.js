import { useCallback, useMemo, useState } from "react";

import { generateConcentricRings, shuffleArray } from "../utils/datasets";
import { createFiveLayerReLUModel } from "../ml/binaryClassifier";
import { useModelTrainer } from "./useModelTrainer";

const TRAIN_SIZES = [20, 200, 2000];
const TEST_SIZE = 300;
const DEFAULT_NOISE = 0.3;
const DEFAULT_EPOCHS = 150;
const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_BATCH_SIZE = 16;
const HIDDEN_UNITS = 16;

/**
 * Controller hook for the Generalization experiment. Generates one noisy
 * dataset, splits off a fixed test set, and trains the exact same
 * over-parameterized model three times on progressively larger slices of
 * the remaining training pool (20 / 200 / 2000 samples) — architecture and
 * hyperparameters held identical throughout, so training-set size is the
 * only variable. Reuses `useModelTrainer`'s `validationDataset` option
 * (evaluated against the one fixed test set every run) rather than a
 * bespoke pipeline.
 */
export function useGeneralizationExperiment() {
  // --- Shared dataset + fixed test set -----------------------------------
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [datasetVersion, setDatasetVersion] = useState(0);

  const { trainSubsets, testSet } = useMemo(() => {
    const poolSize = Math.max(...TRAIN_SIZES) + TEST_SIZE;
    const pool = generateConcentricRings({ numPoints: poolSize, noise });
    const shuffled = shuffleArray(pool);

    const testSet = shuffled.slice(0, TEST_SIZE);
    const trainPool = shuffled.slice(TEST_SIZE);
    const trainSubsets = TRAIN_SIZES.map((size) => trainPool.slice(0, size));

    return { trainSubsets, testSet };
    // datasetVersion has no value of its own — bumping it forces a fresh
    // random draw with the same noise.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [noise, datasetVersion]);

  const regenerateDataset = useCallback(() => {
    setDatasetVersion((version) => version + 1);
  }, []);

  // --- Shared hyperparameters (identical for all three runs) -------------
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- The three training-set sizes under comparison ----------------------
  const createModel = useCallback(() => createFiveLayerReLUModel({ inputDim: 2, hiddenUnits: HIDDEN_UNITS }), []);

  const small = useModelTrainer({
    dataset: trainSubsets[0],
    createModel,
    epochs,
    learningRate,
    batchSize,
    validationDataset: testSet,
  });
  const medium = useModelTrainer({
    dataset: trainSubsets[1],
    createModel,
    epochs,
    learningRate,
    batchSize,
    validationDataset: testSet,
  });
  const large = useModelTrainer({
    dataset: trainSubsets[2],
    createModel,
    epochs,
    learningRate,
    batchSize,
    validationDataset: testSet,
  });

  const isAnyTraining = small.isTraining || medium.isTraining || large.isTraining;

  const trainAll = useCallback(async () => {
    if (isAnyTraining) return;
    await small.train();
    await medium.train();
    await large.train();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAnyTraining, small.train, medium.train, large.train]);

  const resetResults = useCallback(() => {
    small.reset();
    medium.reset();
    large.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [small.reset, medium.reset, large.reset]);

  return {
    // dataset
    testSet,
    noise,
    setNoise,
    regenerateDataset,
    resetResults,
    // hyperparameters
    epochs,
    setEpochs,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    // runs — `dataset` is attached here since useModelTrainer takes it as
    // an input but doesn't echo it back, and the page needs the actual
    // training points to overlay on each run's decision boundary plot.
    trainSizes: TRAIN_SIZES,
    small: { ...small, dataset: trainSubsets[0] },
    medium: { ...medium, dataset: trainSubsets[1] },
    large: { ...large, dataset: trainSubsets[2] },
    trainAll,
    isAnyTraining,
  };
}
