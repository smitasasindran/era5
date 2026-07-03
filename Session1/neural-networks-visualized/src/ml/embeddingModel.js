import * as tf from "@tensorflow/tfjs";

// Framework-agnostic TensorFlow.js training layer for next-token
// prediction. Same conventions as binaryClassifier.js (create → tensors →
// train → inspect), but for a multi-class softmax problem instead of
// binary classification, since the two tasks need different loss
// functions and output shapes.

const DEFAULT_LEARNING_RATE = 0.05;
const DEFAULT_EPOCHS = 150;
const DEFAULT_BATCH_SIZE = 16;

/**
 * Creates a tiny next-token-prediction model: an Embedding layer (one
 * learned vector per vocabulary word) feeding a softmax Dense layer that
 * predicts a probability distribution over "what word comes next". The
 * Embedding layer's weights are the thing this whole experiment is about
 * — see `extractEmbeddings`.
 *
 * @param {Object} options
 * @param {number} options.vocabSize - Number of distinct words.
 * @param {number} [options.embeddingDim=4] - Size of each word's embedding vector.
 * @returns {tf.Sequential} An uncompiled Sequential model.
 */
export function createEmbeddingModel({ vocabSize, embeddingDim = 4 }) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: embeddingDim, inputLength: 1 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: vocabSize, activation: "softmax" }));
  return model;
}

/**
 * Converts (currentWordIndex, nextWordIndex) bigrams into training
 * tensors: `xs` is [n, 1] of input word indices, `ys` is [n, vocabSize]
 * one-hot next-word targets.
 *
 * @param {Array<[number, number]>} bigramIndices
 * @param {number} vocabSize
 * @returns {{xs: tf.Tensor2D, ys: tf.Tensor2D}} Caller owns and must dispose these.
 */
export function bigramIndicesToTensors(bigramIndices, vocabSize) {
  const currentIndices = bigramIndices.map(([current]) => [current]);
  const nextIndices = bigramIndices.map(([, next]) => next);

  const xs = tf.tensor2d(currentIndices, [bigramIndices.length, 1], "int32");
  const ys = tf.oneHot(tf.tensor1d(nextIndices, "int32"), vocabSize).toFloat();

  return { xs, ys };
}

/**
 * Compiles (if needed) and trains the model with categorical
 * cross-entropy and the Adam optimizer.
 *
 * @param {tf.LayersModel} model
 * @param {tf.Tensor2D} xs
 * @param {tf.Tensor2D} ys
 * @param {Object} [options]
 * @param {number} [options.epochs=150]
 * @param {number} [options.learningRate=0.05]
 * @param {number} [options.batchSize=16]
 * @param {Function} [options.onEpochEnd] - Forwarded to tf.js's fit callbacks.
 * @returns {Promise<{history: Object, finalMetrics: {loss: number, accuracy: number}}>}
 */
export async function trainEmbeddingModel(model, xs, ys, options = {}) {
  const {
    epochs = DEFAULT_EPOCHS,
    learningRate = DEFAULT_LEARNING_RATE,
    batchSize = DEFAULT_BATCH_SIZE,
    onEpochEnd,
  } = options;

  if (!model.optimizer) {
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

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
 * Reads out the learned embedding matrix as a plain JS array of arrays —
 * one row per vocabulary word, in the same order the model was built
 * with.
 *
 * @param {tf.LayersModel} model
 * @returns {number[][]}
 */
export function extractEmbeddings(model) {
  const embeddingLayer = model.layers[0];
  const [weights] = embeddingLayer.getWeights();
  return weights.arraySync();
}
