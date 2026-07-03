import * as tf from "@tensorflow/tfjs";

import { predict } from "./binaryClassifier";

/**
 * Computes axis-aligned bounds around a set of 2D points, expanded by a
 * fractional padding so a plotted decision boundary doesn't clip points
 * sitting right at the edge of the data.
 *
 * @param {Array<{x: number, y: number}>} points
 * @param {number} [padding=0.5] - Fraction of each axis's span to pad on
 *   both sides (falls back to a flat 0.5 unit pad if a span is zero).
 * @returns {{xMin: number, xMax: number, yMin: number, yMax: number}}
 */
export function getDataBounds(points, padding = 0.5) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const xPad = (xMax - xMin) * padding || padding;
  const yPad = (yMax - yMin) * padding || padding;

  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  };
}

function linspace(min, max, steps) {
  if (steps === 1) return [min];
  const step = (max - min) / (steps - 1);
  return Array.from({ length: steps }, (_, i) => min + i * step);
}

/**
 * Evaluates any binary TF.js classifier (2 input features, sigmoid output)
 * over a regular grid spanning the given bounds, producing the probability
 * surface a contour plot needs. Works with any model shape — linear,
 * hidden-layer, or otherwise — as long as it takes 2 features and predicts
 * a single sigmoid probability.
 *
 * @param {tf.LayersModel} model
 * @param {Object} options
 * @param {number} options.xMin
 * @param {number} options.xMax
 * @param {number} options.yMin
 * @param {number} options.yMax
 * @param {number} [options.resolution=50] - Grid steps per axis.
 * @returns {Promise<{x: number[], y: number[], z: number[][]}>}
 *   `x`/`y` are the grid's axis values; `z[row][col]` is the predicted
 *   probability at (x[col], y[row]) — the shape Plotly's contour trace
 *   expects.
 */
export async function computeDecisionBoundary(
  model,
  { xMin, xMax, yMin, yMax, resolution = 50 },
) {
  const x = linspace(xMin, xMax, resolution);
  const y = linspace(yMin, yMax, resolution);

  const gridPoints = [];
  for (let row = 0; row < resolution; row++) {
    for (let col = 0; col < resolution; col++) {
      gridPoints.push([x[col], y[row]]);
    }
  }

  const gridXs = tf.tensor2d(gridPoints);
  let probabilities;
  try {
    probabilities = await predict(model, gridXs);
  } finally {
    gridXs.dispose();
  }

  const z = [];
  for (let row = 0; row < resolution; row++) {
    z.push(probabilities.slice(row * resolution, (row + 1) * resolution));
  }

  return { x, y, z };
}
