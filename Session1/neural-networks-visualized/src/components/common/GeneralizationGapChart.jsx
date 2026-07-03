import Plot from "react-plotly.js";

import { basePlotlyLayout } from "./plotlyTheme";

/**
 * This chapter's centerpiece: generalization gap plotted against training
 * set size. Training set sizes typically span multiple orders of
 * magnitude, so the x-axis uses a log scale.
 *
 * @param {Array<{trainSize: number, gap: number}>} runs - `gap` as a fraction (e.g. 0.16 = 16%).
 */
function GeneralizationGapChart({ runs, height = 360 }) {
  const trace = {
    x: runs.map((run) => run.trainSize),
    y: runs.map((run) => run.gap * 100),
    mode: "lines+markers",
    type: "scatter",
    name: "Generalization gap",
    line: { color: "#fbbf24", width: 3 },
    marker: { color: "#fbbf24", size: 11 },
  };

  const layout = {
    ...basePlotlyLayout({
      height,
      xLabel: "Training set size (log scale)",
      yLabel: "Generalization gap (%)",
      equalAspect: false,
    }),
    xaxis: {
      title: "Training set size (log scale)",
      type: "log",
      zeroline: false,
      showgrid: false,
      tickvals: runs.map((run) => run.trainSize),
    },
  };

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default GeneralizationGapChart;
