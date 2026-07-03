import Plot from "react-plotly.js";

import { buildScatterTraces, basePlotlyLayout, CLASS_COLORS } from "./plotlyTheme";

// Padding around the DATA POINTS themselves (not the boundary's prediction
// grid, which pads much wider so the contour doesn't look clipped at the
// data's edge). Fixing the view to this range keeps the plot's zoom level
// identical before and after training — without it, Plotly auto-fits to
// whatever traces are visible, so adding the wider boundary grid makes it
// zoom out and every point appears to shrink right when training finishes.
const VIEW_PADDING = 0.15;

function computeAxisRange(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * VIEW_PADDING || VIEW_PADDING;
  return [min - pad, max + pad];
}

/**
 * Scatter plot of labeled 2D points with an optional decision-boundary
 * probability contour rendered beneath them. Pass `boundary` — the
 * {x, y, z} grid from `computeDecisionBoundary` (src/ml/decisionBoundary.js)
 * — once a model has been trained; omit it to fall back to a plain scatter.
 */
function DecisionBoundaryPlot({ data, boundary, height = 420, xLabel = "x", yLabel = "y" }) {
  const traces = [];

  // Pushed first so Plotly draws it beneath the scatter traces. A hard
  // 2-band split at 0.5 (rather than a smooth gradient) keeps the boundary
  // itself crisp and legible, especially for the linear model's near-flat
  // probability surface.
  if (boundary) {
    traces.push({
      x: boundary.x,
      y: boundary.y,
      z: boundary.z,
      type: "contour",
      showscale: false,
      opacity: 0.85,
      hoverinfo: "skip",
      contours: { start: 0, end: 1, size: 0.5, coloring: "fill" },
      line: { width: 0 },
      colorscale: [
        [0, CLASS_COLORS[0]],
        [0.5, CLASS_COLORS[0]],
        [0.5, CLASS_COLORS[1]],
        [1, CLASS_COLORS[1]],
      ],
    });
  }

  traces.push(...buildScatterTraces(data));

  const layout = basePlotlyLayout({ height, xLabel, yLabel });
  layout.xaxis = {
    ...layout.xaxis,
    range: computeAxisRange(data.map((point) => point.x)),
    autorange: false,
  };
  layout.yaxis = {
    ...layout.yaxis,
    range: computeAxisRange(data.map((point) => point.y)),
    autorange: false,
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default DecisionBoundaryPlot;
