import { useCallback, useEffect, useRef, useState } from "react";

import { datasetToTensors, trainModel, disposeTensors } from "../ml/binaryClassifier";
import { getDataBounds, computeDecisionBoundary } from "../ml/decisionBoundary";

const DEFAULT_BOUNDARY_RESOLUTION = 60;

/**
 * Generic controller for training a single pluggable binary classifier
 * against a shared dataset. Owns the model instance, tensor/model
 * disposal, a duplicate-run guard, and — once training completes — the
 * decision boundary grid for that model. Framework-agnostic orchestration
 * lives entirely here so experiment pages stay thin views.
 *
 * Multiple independent instances of this hook can share the same
 * `dataset` reference to train and compare several architectures fairly
 * (e.g. Activation Functions, Network Depth), or share the same
 * `validationDataset` while each getting a different `dataset` to compare
 * training-set sizes fairly (e.g. Generalization).
 *
 * @param {Object} options
 * @param {Array<{x: number, y: number, label: number}>} options.dataset
 * @param {() => import("@tensorflow/tfjs").LayersModel} options.createModel
 * @param {number} options.epochs
 * @param {number} options.learningRate
 * @param {number} options.batchSize
 * @param {number} [options.boundaryResolution=60]
 * @param {Array<{x: number, y: number, label: number}>} [options.validationDataset] -
 *   Optional fixed held-out set. When given, training also reports
 *   `validationMetrics` and per-epoch `history.val_loss`/`val_acc`.
 */
export function useModelTrainer({
  dataset,
  createModel,
  epochs,
  learningRate,
  batchSize,
  boundaryResolution = DEFAULT_BOUNDARY_RESOLUTION,
  validationDataset,
}) {
  const [status, setStatus] = useState("idle"); // "idle" | "training" | "completed" | "error"
  const [metrics, setMetrics] = useState(null); // { loss, accuracy }
  const [validationMetrics, setValidationMetrics] = useState(null); // { loss, accuracy } on validationDataset
  const [history, setHistory] = useState(null); // per-epoch { loss, acc, val_loss?, val_acc? }
  const [boundary, setBoundary] = useState(null); // { x, y, z } grid for DecisionBoundaryPlot
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
    setValidationMetrics(null);
    setHistory(null);
    setBoundary(null);
    setError(null);
  }, [disposeModel]);

  // Any config change (new dataset, validation set, hyperparameters, or a
  // different `createModel` — e.g. the architecture's layer/neuron count
  // changed) makes a previously trained model and its boundary stale.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reset, [dataset, validationDataset, epochs, learningRate, batchSize, createModel]);

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
    setBoundary(null);

    const model = createModel();
    modelRef.current = model;

    const { xs, ys } = datasetToTensors({ points: dataset });
    const validationTensors = validationDataset
      ? datasetToTensors({ points: validationDataset })
      : null;

    try {
      const { history: fitHistory, finalMetrics } = await trainModel(model, xs, ys, {
        epochs,
        learningRate,
        batchSize,
        validationData: validationTensors ?? undefined,
        onEpochEnd: () => {
          if (!isMountedRef.current) {
            model.stopTraining = true;
          }
        },
      });

      const bounds = getDataBounds(dataset);
      const nextBoundary = await computeDecisionBoundary(model, {
        ...bounds,
        resolution: boundaryResolution,
      });

      if (isMountedRef.current) {
        setMetrics(finalMetrics);
        setBoundary(nextBoundary);
        setHistory(fitHistory);
        if (finalMetrics.valLoss !== undefined) {
          setValidationMetrics({ loss: finalMetrics.valLoss, accuracy: finalMetrics.valAccuracy });
        }
        setStatus("completed");
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err.message ?? "Training failed.");
        setStatus("error");
      }
    } finally {
      disposeTensors({ xs, ys });
      if (validationTensors) disposeTensors(validationTensors);
      isTrainingRef.current = false;
    }
  }, [
    dataset,
    validationDataset,
    epochs,
    learningRate,
    batchSize,
    createModel,
    boundaryResolution,
    disposeModel,
  ]);

  return {
    status,
    metrics,
    validationMetrics,
    history,
    boundary,
    error,
    train,
    reset,
    isTraining: status === "training",
  };
}
