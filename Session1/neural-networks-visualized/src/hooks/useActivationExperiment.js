import { useCallback, useMemo, useState } from "react";

import { generateConcentricRings } from "../utils/datasets";
import { createLinearModel, createHiddenLayerModel } from "../ml/binaryClassifier";
import { useModelTrainer } from "./useModelTrainer";

const DEFAULT_NUM_POINTS = 300;
const DEFAULT_NOISE = 0.2;
const DEFAULT_EPOCHS = 80;
const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_BATCH_SIZE = 32;
const HIDDEN_UNITS = 8;

/**
 * Controller hook for the Activation Functions experiment. Generates a
 * single shared dataset and trains a linear model and a ReLU network
 * against it — using identical hyperparameters by default — so the two
 * architectures can be compared fairly. ActivationPage stays a thin view
 * over this state.
 */
export function useActivationExperiment() {
  // --- Shared dataset ----------------------------------------------------
  const [numPoints, setNumPoints] = useState(DEFAULT_NUM_POINTS);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [datasetVersion, setDatasetVersion] = useState(0);

  const dataset = useMemo(
    () => generateConcentricRings({ numPoints, noise }),
    // datasetVersion has no value of its own — bumping it forces a fresh
    // random draw with the same numPoints/noise, still shared by both models.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numPoints, noise, datasetVersion],
  );

  const regenerateDataset = useCallback(() => {
    setDatasetVersion((version) => version + 1);
  }, []);

  // --- Shared hyperparameters (identical for both models by default) -----
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- The two architectures under comparison ------------------------------
  const createLinear = useCallback(() => createLinearModel({ inputDim: 2 }), []);
  const createRelu = useCallback(
    () => createHiddenLayerModel({ inputDim: 2, hiddenUnits: HIDDEN_UNITS }),
    [],
  );

  const linear = useModelTrainer({ dataset, createModel: createLinear, epochs, learningRate, batchSize });
  const relu = useModelTrainer({ dataset, createModel: createRelu, epochs, learningRate, batchSize });

  return {
    // dataset
    dataset,
    numPoints,
    setNumPoints,
    noise,
    setNoise,
    regenerateDataset,
    // hyperparameters
    epochs,
    setEpochs,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    // models
    linear,
    relu,
    isAnyTraining: linear.isTraining || relu.isTraining,
  };
}
