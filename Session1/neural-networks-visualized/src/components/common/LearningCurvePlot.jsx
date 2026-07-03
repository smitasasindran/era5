import Plot from "react-plotly.js";

import { basePlotlyLayout, CLASS_COLORS } from "./plotlyTheme";

/**
 * Line chart of training vs. validation loss across epochs — the classic
 * over/underfitting diagnostic. `validationLoss` is optional so this also
 * works for runs trained without a held-out set.
 */
function LearningCurvePlot({ trainLoss, validationLoss, height = 220 }) {
  const epochs = trainLoss.map((_, index) => index + 1);

  const traces = [
    {
      x: epochs,
      y: trainLoss,
      mode: "lines",
      type: "scatter",
      name: "Train loss",
      line: { color: CLASS_COLORS[0], width: 2 },
    },
  ];

  if (validationLoss) {
    traces.push({
      x: epochs,
      y: validationLoss,
      mode: "lines",
      type: "scatter",
      name: "Test loss",
      line: { color: CLASS_COLORS[1], width: 2 },
    });
  }

  return (
    <Plot
      data={traces}
      layout={basePlotlyLayout({ height, xLabel: "Epoch", yLabel: "Loss", equalAspect: false })}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default LearningCurvePlot;
