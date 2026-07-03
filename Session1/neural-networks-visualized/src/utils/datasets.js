// Standard-normal noise via the Box-Muller transform.
function gaussianNoise() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Generates two concentric, noisy rings of labeled 2D points — a classic
 * dataset that is not linearly separable. Returns an array of
 * { x, y, label } points, where label is 0 (inner ring) or 1 (outer ring).
 */
export function generateConcentricRings({ numPoints = 300, noise = 0.2 } = {}) {
  const innerCount = Math.floor(numPoints / 2);
  const outerCount = numPoints - innerCount;
  const points = [];

  const addRing = (count, radius, label) => {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = radius + gaussianNoise() * noise;
      points.push({
        x: r * Math.cos(angle),
        y: r * Math.sin(angle),
        label,
      });
    }
  };

  addRing(innerCount, 1, 0);
  addRing(outerCount, 2.5, 1);

  return points;
}

/**
 * Returns a new array with the same elements in random order (Fisher-Yates).
 * Used to split a generated dataset into train/test subsets without bias.
 */
export function shuffleArray(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
