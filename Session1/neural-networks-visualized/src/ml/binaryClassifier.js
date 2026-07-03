import * as tf from "@tensorflow/tfjs";

// Framework-agnostic TensorFlow.js training layer for binary classification.
// No React or DOM dependencies live here — this module can be imported by
// any experiment page (or reused outside the app entirely).

const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_EPOCHS = 50;
const DEFAULT_BATCH_SIZE = 32;

/**
 * Creates a linear binary classifier: a single Dense layer with a sigmoid
 * output (logistic regression). The model is left uncompiled — `trainModel`
 * and `evaluate` compile it on first use.
 *
 * @param {Object} [options]
 * @param {number} [options.inputDim=2] - Number of input features.
 * @returns {tf.Sequential} An uncompiled Sequential model.
 */
export function createLinearModel({ inputDim = 2 } = {}) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [inputDim],
      activation: "sigmoid",
    }),
  );
  return model;
}

/**
 * Converts a labeled 2D point dataset into training tensors.
 *
 * @param {Object} dataset
 * @param {Array<{x: number, y: number, label: number}>} dataset.points - Labeled points.
 * @param {Object} [dataset.metadata] - Optional provenance info (e.g. noise,
 *   numPoints); not used for the conversion, just passed through so callers
 *   can keep it alongside the tensors.
 * @returns {{xs: tf.Tensor2D, ys: tf.Tensor2D, metadata: (Object|undefined)}}
 *   `xs` has shape [n, 2], `ys` has shape [n, 1]. The caller owns the
 *   returned tensors and must dispose them (see `disposeTensors`) once done.
 */
export function datasetToTensors({ points, metadata }) {
  const features = points.map((point) => [point.x, point.y]);
  const labels = points.map((point) => [point.label]);

  return {
    xs: tf.tensor2d(features),
    ys: tf.tensor2d(labels),
    metadata,
  };
}

/**
 * Compiles a model with binary cross-entropy loss and the Adam optimizer,
 * unless it has already been compiled. Exported separately so callers can
 * get a compiled model without going through `trainModel`.
 *
 * @param {tf.LayersModel} model
 * @param {Object} [options]
 * @param {number} [options.learningRate=0.05]
 */
export function compileModel(model, { learningRate = DEFAULT_LEARNING_RATE } = {}) {
  if (model.optimizer) return;

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
}

/**
 * Trains a model with binary cross-entropy loss and the Adam optimizer.
 *
 * @param {tf.LayersModel} model
 * @param {tf.Tensor2D} xs
 * @param {tf.Tensor2D} ys
 * @param {Object} [options]
 * @param {number} [options.epochs=50]
 * @param {number} [options.learningRate=0.05]
 * @param {number} [options.batchSize=32]
 * @param {Function} [options.onEpochEnd] - Forwarded to tf.js as the
 *   `onEpochEnd(epoch, logs)` fit callback, useful for progress reporting.
 * @returns {Promise<{history: Object, finalMetrics: {loss: number, accuracy: number}}>}
 *   `history` holds the per-epoch arrays tf.js records (loss, acc, ...).
 */
export async function trainModel(model, xs, ys, options = {}) {
  const {
    epochs = DEFAULT_EPOCHS,
    learningRate = DEFAULT_LEARNING_RATE,
    batchSize = DEFAULT_BATCH_SIZE,
    onEpochEnd,
  } = options;

  compileModel(model, { learningRate });

  const { history } = await model.fit(xs, ys, {
    epochs,
    batchSize,
    shuffle: true,
    callbacks: onEpochEnd ? { onEpochEnd } : undefined,
  });

  const lastEpoch = epochs - 1;
  const accuracyKey = "acc" in history ? "acc" : "accuracy";

  return {
    history,
    finalMetrics: {
      loss: history.loss[lastEpoch],
      accuracy: history[accuracyKey][lastEpoch],
    },
  };
}

/**
 * Runs inference and returns class-1 probabilities (the sigmoid output).
 *
 * @param {tf.LayersModel} model
 * @param {tf.Tensor2D} xs
 * @returns {Promise<number[]>}
 */
export async function predict(model, xs) {
  const output = model.predict(xs);
  const probabilities = await output.data();
  output.dispose();
  return Array.from(probabilities);
}

/**
 * Evaluates a model on held-out data, compiling it first if needed.
 *
 * @param {tf.LayersModel} model
 * @param {tf.Tensor2D} xs
 * @param {tf.Tensor2D} ys
 * @returns {Promise<{loss: number, accuracy: (number|undefined)}>}
 */
export async function evaluate(model, xs, ys) {
  compileModel(model);

  const result = model.evaluate(xs, ys, { batchSize: xs.shape[0] });
  const [lossTensor, accuracyTensor] = Array.isArray(result) ? result : [result, undefined];

  const [loss] = await lossTensor.data();
  const accuracy = accuracyTensor ? (await accuracyTensor.data())[0] : undefined;

  lossTensor.dispose();
  accuracyTensor?.dispose();

  return { loss, accuracy };
}

/**
 * Disposes the tensors returned by `datasetToTensors` (or any {xs, ys}
 * pair) to free WebGL/CPU memory once a model no longer needs them.
 *
 * @param {{xs?: tf.Tensor, ys?: tf.Tensor}} tensors
 */
export function disposeTensors({ xs, ys } = {}) {
  xs?.dispose();
  ys?.dispose();
}
