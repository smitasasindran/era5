import { useCallback, useMemo, useState } from "react";

import { generateConcentricRings } from "../utils/datasets";
import {
  createLinearModel,
  createFiveLayerLinearModel,
  createFiveLayerReLUModel,
} from "../ml/binaryClassifier";
import { useModelTrainer } from "./useModelTrainer";

const DEFAULT_NUM_POINTS = 300;
const DEFAULT_NOISE = 0.2;
const DEFAULT_EPOCHS = 80;
const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_BATCH_SIZE = 32;
const HIDDEN_UNITS = 8;

/**
 * Controller hook for the Network Depth experiment. Generates a single
 * shared dataset and trains three models against it — a single linear
 * layer, five stacked linear layers with no activation, and five layers
 * with ReLU between each — using identical hyperparameters by default, so
 * the only variable between them is architecture. Mirrors
 * `useActivationExperiment`, reusing the same `useModelTrainer` primitive
 * three times instead of two.
 */
export function useDepthExperiment() {
  // --- Shared dataset ----------------------------------------------------
  const [numPoints, setNumPoints] = useState(DEFAULT_NUM_POINTS);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [datasetVersion, setDatasetVersion] = useState(0);

  const dataset = useMemo(
    () => generateConcentricRings({ numPoints, noise }),
    // datasetVersion has no value of its own — bumping it forces a fresh
    // random draw with the same numPoints/noise, still shared by all models.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numPoints, noise, datasetVersion],
  );

  const regenerateDataset = useCallback(() => {
    setDatasetVersion((version) => version + 1);
  }, []);

  // --- Shared hyperparameters (identical for all three models by default) --
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- The three architectures under comparison ---------------------------
  const createLinear = useCallback(() => createLinearModel({ inputDim: 2 }), []);
  const createFiveLinear = useCallback(
    () => createFiveLayerLinearModel({ inputDim: 2, hiddenUnits: HIDDEN_UNITS }),
    [],
  );
  const createFiveRelu = useCallback(
    () => createFiveLayerReLUModel({ inputDim: 2, hiddenUnits: HIDDEN_UNITS }),
    [],
  );

  const linear = useModelTrainer({ dataset, createModel: createLinear, epochs, learningRate, batchSize });
  const fiveLinear = useModelTrainer({
    dataset,
    createModel: createFiveLinear,
    epochs,
    learningRate,
    batchSize,
  });
  const fiveRelu = useModelTrainer({
    dataset,
    createModel: createFiveRelu,
    epochs,
    learningRate,
    batchSize,
  });

  const resetResults = useCallback(() => {
    linear.reset();
    fiveLinear.reset();
    fiveRelu.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [linear.reset, fiveLinear.reset, fiveRelu.reset]);

  return {
    // dataset
    dataset,
    numPoints,
    setNumPoints,
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
    // models
    linear,
    fiveLinear,
    fiveRelu,
    isAnyTraining: linear.isTraining || fiveLinear.isTraining || fiveRelu.isTraining,
  };
}
