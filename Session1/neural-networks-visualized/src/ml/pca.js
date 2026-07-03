// Minimal PCA implementation for projecting small embedding matrices down
// to 2 dimensions for plotting. Framework-agnostic plain JS — the matrices
// involved here (a handful of words x a handful of embedding dimensions)
// are far too small to need TensorFlow.js or an external linear-algebra
// dependency, so this uses power iteration, the simplest approach that
// still converges reliably at this scale.

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function matVec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function normalize(vector) {
  const norm = Math.sqrt(dot(vector, vector)) || 1;
  return vector.map((value) => value / norm);
}

// Finds the dominant eigenvector of a symmetric matrix via power iteration.
function powerIteration(matrix, dimensions, iterations = 200) {
  let vector = normalize(Array.from({ length: dimensions }, () => Math.random() - 0.5));
  for (let i = 0; i < iterations; i++) {
    vector = normalize(matVec(matrix, vector));
  }
  return vector;
}

// Removes an eigenvector's contribution from a matrix (Hotelling deflation)
// so the next call to `powerIteration` finds the next-largest component.
function deflate(matrix, eigenvector, dimensions) {
  const matrixTimesVector = matVec(matrix, eigenvector);
  const eigenvalue = dot(eigenvector, matrixTimesVector);

  const deflated = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0));
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < dimensions; j++) {
      deflated[i][j] = matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j];
    }
  }
  return deflated;
}

function meanCenter(matrix) {
  const rowCount = matrix.length;
  const dimensions = matrix[0].length;

  const mean = new Array(dimensions).fill(0);
  for (const row of matrix) {
    for (let j = 0; j < dimensions; j++) mean[j] += row[j] / rowCount;
  }
  const centered = matrix.map((row) => row.map((value, j) => value - mean[j]));
  return { centered, mean, dimensions, rowCount };
}

function covarianceOf(centered, dimensions, rowCount) {
  const covariance = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0));
  for (const row of centered) {
    for (let i = 0; i < dimensions; i++) {
      for (let j = 0; j < dimensions; j++) {
        covariance[i][j] += (row[i] * row[j]) / Math.max(rowCount - 1, 1);
      }
    }
  }
  return covariance;
}

/**
 * Projects each row of `matrix` onto its top 2 principal components.
 *
 * @param {number[][]} matrix - n rows (e.g. one per word) x d columns (embedding dimensions).
 * @returns {number[][]} n rows x 2 columns — the 2D projection.
 */
export function projectTo2D(matrix) {
  const { centered, dimensions, rowCount } = meanCenter(matrix);
  const covariance = covarianceOf(centered, dimensions, rowCount);

  const firstComponent = powerIteration(covariance, dimensions);
  const secondComponent = powerIteration(deflate(covariance, firstComponent, dimensions), dimensions);

  return centered.map((row) => [dot(row, firstComponent), dot(row, secondComponent)]);
}

/**
 * Like `projectTo2D`, but for projecting a *sequence* of slowly-changing
 * matrices (e.g. an embedding matrix at successive training epochs) into a
 * visually continuous animation. PCA components are only defined up to an
 * arbitrary sign, so computing them independently for each frame can flip
 * a component's direction from one call to the next — which looks like a
 * jump cut, not smooth motion. Passing the previous call's `basis` back in
 * flips each new component's sign to match it whenever they'd otherwise
 * point opposite ways, so consecutive frames stay visually consistent.
 *
 * @param {number[][]} matrix
 * @param {{components: number[][]}|null} previousBasis - The `basis` returned by the prior call, or null for the first frame.
 * @returns {{points: number[][], basis: {components: number[][]}}}
 */
export function projectTo2DContinuous(matrix, previousBasis = null) {
  const { centered, dimensions, rowCount } = meanCenter(matrix);
  const covariance = covarianceOf(centered, dimensions, rowCount);

  let firstComponent = powerIteration(covariance, dimensions);
  let secondComponent = powerIteration(deflate(covariance, firstComponent, dimensions), dimensions);

  if (previousBasis) {
    if (dot(firstComponent, previousBasis.components[0]) < 0) {
      firstComponent = firstComponent.map((value) => -value);
    }
    if (dot(secondComponent, previousBasis.components[1]) < 0) {
      secondComponent = secondComponent.map((value) => -value);
    }
  }

  const points = centered.map((row) => [dot(row, firstComponent), dot(row, secondComponent)]);
  return { points, basis: { components: [firstComponent, secondComponent] } };
}
