import Plot from "react-plotly.js";

// Colors match the app's accent (violet) and a complementary cyan.
const CLASS_COLORS = ["#a06aff", "#47bfff"];

/**
 * Reusable dark-themed scatter plot for labeled 2D points ({x, y, label}).
 * Each distinct label is rendered as its own colored trace.
 */
function ScatterPlot({ data, height = 420, xLabel = "x", yLabel = "y" }) {
  const labels = [...new Set(data.map((point) => point.label))].sort();

  const traces = labels.map((label, index) => {
    const points = data.filter((point) => point.label === label);
    return {
      x: points.map((point) => point.x),
      y: points.map((point) => point.y),
      mode: "markers",
      type: "scatter",
      name: `Class ${label}`,
      marker: {
        color: CLASS_COLORS[index % CLASS_COLORS.length],
        size: 7,
        opacity: 0.85,
        line: { width: 0.5, color: "#0f172a" },
      },
    };
  });

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height,
        margin: { l: 40, r: 20, t: 20, b: 40 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#94a3b8" },
        xaxis: {
          title: xLabel,
          zeroline: false,
          gridcolor: "#1e293b",
          scaleanchor: "y",
          scaleratio: 1,
        },
        yaxis: { title: yLabel, zeroline: false, gridcolor: "#1e293b" },
        legend: { orientation: "h", y: -0.15 },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

export default ScatterPlot;
