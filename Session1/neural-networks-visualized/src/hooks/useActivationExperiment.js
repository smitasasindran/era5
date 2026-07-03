import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { generateConcentricRings } from "../utils/datasets";
import {
  createLinearModel,
  datasetToTensors,
  trainModel,
  disposeTensors,
} from "../ml/binaryClassifier";

const DEFAULT_NUM_POINTS = 300;
const DEFAULT_NOISE = 0.2;
const DEFAULT_EPOCHS = 50;
const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_BATCH_SIZE = 32;

/**
 * Controller hook for the Activation Functions experiment. Owns the
 * dataset, hyperparameters, and the full model training lifecycle
 * (creation, fitting, tensor/model disposal), keeping ActivationPage a
 * thin view over this state.
 */
export function useActivationExperiment() {
  // --- Dataset ---------------------------------------------------------
  const [numPoints, setNumPoints] = useState(DEFAULT_NUM_POINTS);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [datasetVersion, setDatasetVersion] = useState(0);

  const dataset = useMemo(
    () => generateConcentricRings({ numPoints, noise }),
    // datasetVersion has no value of its own — bumping it forces a fresh
    // random draw with the same numPoints/noise.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numPoints, noise, datasetVersion],
  );

  const regenerateDataset = useCallback(() => {
    setDatasetVersion((version) => version + 1);
  }, []);

  // --- Hyperparameters ---------------------------------------------------
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);

  // --- Training lifecycle ------------------------------------------------
  const [status, setStatus] = useState("idle"); // "idle" | "training" | "completed" | "error"
  const [metrics, setMetrics] = useState(null); // { loss, accuracy }
  const [error, setError] = useState(null);

  const modelRef = useRef(null);
  const isTrainingRef = useRef(false);
  const isMountedRef = useRef(true);

  const disposeModel = useCallback(() => {
    modelRef.current?.dispose();
    modelRef.current = null;
  }, []);

  // Any config change makes a previously trained model stale.
  useEffect(() => {
    disposeModel();
    setStatus("idle");
    setMetrics(null);
    setError(null);
  }, [dataset, epochs, learningRate, batchSize, disposeModel]);

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

    const model = createLinearModel({ inputDim: 2 });
    modelRef.current = model;

    const { xs, ys } = datasetToTensors({ points: dataset });

    try {
      const { finalMetrics } = await trainModel(model, xs, ys, {
        epochs,
        learningRate,
        batchSize,
        onEpochEnd: () => {
          if (!isMountedRef.current) {
            model.stopTraining = true;
          }
        },
      });

      if (isMountedRef.current) {
        setMetrics(finalMetrics);
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
  }, [dataset, epochs, learningRate, batchSize, disposeModel]);

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
    // training
    status,
    metrics,
    error,
    train,
    isTraining: status === "training",
  };
}
