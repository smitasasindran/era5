import Plot from "react-plotly.js";

import { buildScatterTraces, basePlotlyLayout } from "./plotlyTheme";

/**
 * Reusable dark-themed scatter plot for labeled 2D points ({x, y, label}).
 * Each distinct label is rendered as its own colored trace.
 */
function ScatterPlot({ data, height = 420, xLabel = "x", yLabel = "y" }) {
  return (
    <Plot
      data={buildScatterTraces(data)}
      layout={basePlotlyLayout({ height, xLabel, yLabel })}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default ScatterPlot;
