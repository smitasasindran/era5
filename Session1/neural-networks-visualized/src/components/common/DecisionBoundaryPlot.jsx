import Plot from "react-plotly.js";

import { buildScatterTraces, basePlotlyLayout, CLASS_COLORS } from "./plotlyTheme";

/**
 * Scatter plot of labeled 2D points with an optional decision-boundary
 * probability contour rendered beneath them. Pass `boundary` — the
 * {x, y, z} grid from `computeDecisionBoundary` (src/ml/decisionBoundary.js)
 * — once a model has been trained; omit it to fall back to a plain scatter.
 */
function DecisionBoundaryPlot({ data, boundary, height = 420, xLabel = "x", yLabel = "y" }) {
  const traces = [];

  // Pushed first so Plotly draws it beneath the scatter traces.
  if (boundary) {
    traces.push({
      x: boundary.x,
      y: boundary.y,
      z: boundary.z,
      type: "contour",
      showscale: false,
      opacity: 0.55,
      hoverinfo: "skip",
      contours: { start: 0, end: 1, size: 0.05, coloring: "fill" },
      line: { width: 0 },
      colorscale: [
        [0, CLASS_COLORS[0]],
        [1, CLASS_COLORS[1]],
      ],
    });
  }

  traces.push(...buildScatterTraces(data));

  return (
    <Plot
      data={traces}
      layout={basePlotlyLayout({ height, xLabel, yLabel })}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default DecisionBoundaryPlot;
