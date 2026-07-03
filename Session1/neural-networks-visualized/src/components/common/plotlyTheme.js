// Shared Plotly building blocks so every scatter-based visualization in the
// app uses one dark theme and one set of class colors.

// Matches the app's accent (violet) and a complementary cyan.
export const CLASS_COLORS = ["#a06aff", "#47bfff"];

/**
 * Builds one colored marker trace per distinct `label` in a set of
 * {x, y, label} points.
 */
export function buildScatterTraces(points) {
  const labels = [...new Set(points.map((point) => point.label))].sort();

  return labels.map((label, index) => {
    const subset = points.filter((point) => point.label === label);
    return {
      x: subset.map((point) => point.x),
      y: subset.map((point) => point.y),
      mode: "markers",
      type: "scatter",
      name: `Class ${label}`,
      marker: {
        color: CLASS_COLORS[index % CLASS_COLORS.length],
        size: 7,
        opacity: 0.95,
        line: { width: 1, color: "#0f172a" },
      },
    };
  });
}

/**
 * Shared dark-themed Plotly layout for 2D point plots: equal-aspect axes,
 * transparent background, muted grid.
 */
export function basePlotlyLayout({ height = 420, xLabel = "x", yLabel = "y" } = {}) {
  return {
    autosize: true,
    height,
    margin: { l: 40, r: 20, t: 20, b: 40 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#94a3b8" },
    xaxis: {
      title: xLabel,
      zeroline: false,
      showgrid: false,
      scaleanchor: "y",
      scaleratio: 1,
    },
    yaxis: { title: yLabel, zeroline: false, showgrid: false },
    legend: { orientation: "h", y: -0.15 },
  };
}
