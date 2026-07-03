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
const DEFAULT_EPOCHS = 100;
// A plain (no-activation) stack of several linear layers compounds an
// unbounded transformation at every layer; at the previous 0.1 default
// this measurably diverged (loss spiking into the single digits) in
// roughly half of all training runs even at the original 5-layer depth.
// 0.02 trains all three architectures reliably across the full range of
// layer/neuron counts exposed below.
const DEFAULT_LEARNING_RATE = 0.02;
const DEFAULT_BATCH_SIZE = 32;
const DEFAULT_HIDDEN_UNITS = 8;
const DEFAULT_NUM_LAYERS = 5;

/**
 * Controller hook for the Network Depth experiment. Generates a single
 * shared dataset and trains three models against it — a single linear
 * layer, several stacked linear layers with no activation, and several
 * layers with ReLU between each — using identical training hyperparameters
 * by default, so architecture is the only variable. Each model's depth
 * (layer count, where applicable) and width (neurons per layer) can be
 * tuned independently, letting visitors run exactly the experiments this
 * chapter's "Challenge Yourself" section suggests (more layers, more
 * neurons, etc.) without leaving the page.
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

  // --- Shared training hyperparameters (identical for all three models) --
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- Per-model architecture controls -------------------------------------
  // Model A has no layer-count control: it's always exactly one
  // (effectively linear) layer, but its width is still tunable so visitors
  // can check "does widening a linear layer help, without ReLU?".
  const [linearHiddenUnits, setLinearHiddenUnits] = useState(DEFAULT_HIDDEN_UNITS);

  const [fiveLinearHiddenUnits, setFiveLinearHiddenUnits] = useState(DEFAULT_HIDDEN_UNITS);
  const [fiveLinearNumLayers, setFiveLinearNumLayers] = useState(DEFAULT_NUM_LAYERS);

  const [fiveReluHiddenUnits, setFiveReluHiddenUnits] = useState(DEFAULT_HIDDEN_UNITS);
  const [fiveReluNumLayers, setFiveReluNumLayers] = useState(DEFAULT_NUM_LAYERS);

  const createLinear = useCallback(
    () => createLinearModel({ inputDim: 2, hiddenUnits: linearHiddenUnits }),
    [linearHiddenUnits],
  );
  const createFiveLinear = useCallback(
    () =>
      createFiveLayerLinearModel({
        inputDim: 2,
        hiddenUnits: fiveLinearHiddenUnits,
        numLayers: fiveLinearNumLayers,
      }),
    [fiveLinearHiddenUnits, fiveLinearNumLayers],
  );
  const createFiveRelu = useCallback(
    () =>
      createFiveLayerReLUModel({
        inputDim: 2,
        hiddenUnits: fiveReluHiddenUnits,
        numLayers: fiveReluNumLayers,
      }),
    [fiveReluHiddenUnits, fiveReluNumLayers],
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
    // per-model architecture controls
    linearHiddenUnits,
    setLinearHiddenUnits,
    fiveLinearHiddenUnits,
    setFiveLinearHiddenUnits,
    fiveLinearNumLayers,
    setFiveLinearNumLayers,
    fiveReluHiddenUnits,
    setFiveReluHiddenUnits,
    fiveReluNumLayers,
    setFiveReluNumLayers,
    // models
    linear,
    fiveLinear,
    fiveRelu,
    isAnyTraining: linear.isTraining || fiveLinear.isTraining || fiveRelu.isTraining,
  };
}
