import Plot from "react-plotly.js";

import { buildLabeledScatterTraces, basePlotlyLayout } from "./plotlyTheme";

/**
 * Scatter plot for 2D-projected word embeddings: each point is labeled
 * with its word and colored by category. Pass `selectedWord` to highlight
 * one point (larger marker, brighter outline); pass `onSelectWord` to make
 * points clickable — used for nearest-neighbor lookups.
 */
function EmbeddingScatterPlot({ points, selectedWord, onSelectWord, height = 420 }) {
  const handleClick = (event) => {
    const word = event?.points?.[0]?.text;
    if (word && onSelectWord) onSelectWord(word);
  };

  return (
    <Plot
      data={buildLabeledScatterTraces(points, { selectedWord })}
      layout={basePlotlyLayout({ height, xLabel: "PC1", yLabel: "PC2" })}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
      onClick={onSelectWord ? handleClick : undefined}
    />
  );
}

export default EmbeddingScatterPlot;
