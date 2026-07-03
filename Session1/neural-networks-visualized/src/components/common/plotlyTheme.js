// Shared Plotly building blocks so every scatter-based visualization in the
// app uses one dark theme and one set of class colors.

// Matches the app's accent (violet) and a complementary cyan.
export const CLASS_COLORS = ["#a06aff", "#47bfff"];

// Extends CLASS_COLORS with a third hue (amber, already used for callouts
// elsewhere) for visualizations with 3+ categories, like word embeddings.
export const CATEGORY_COLORS = ["#a06aff", "#47bfff", "#fbbf24"];

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
 * Builds one trace per distinct `category` in a set of {x, y, word,
 * category} points, labeling each marker with its word and optionally
 * enlarging one selected word's marker (e.g. for nearest-neighbor lookups).
 */
export function buildLabeledScatterTraces(points, { colors = CATEGORY_COLORS, selectedWord } = {}) {
  const categories = [...new Set(points.map((point) => point.category))].sort();

  return categories.map((category, index) => {
    const subset = points.filter((point) => point.category === category);
    return {
      x: subset.map((point) => point.x),
      y: subset.map((point) => point.y),
      text: subset.map((point) => point.word),
      mode: "markers+text",
      type: "scatter",
      name: category.charAt(0).toUpperCase() + category.slice(1),
      textposition: "top center",
      textfont: { color: "#cbd5e1", size: 11 },
      hoverinfo: "text",
      marker: {
        color: colors[index % colors.length],
        size: subset.map((point) => (point.word === selectedWord ? 16 : 10)),
        opacity: 0.9,
        line: {
          width: subset.map((point) => (point.word === selectedWord ? 2 : 1)),
          color: subset.map((point) => (point.word === selectedWord ? "#f8fafc" : "#0f172a")),
        },
      },
    };
  });
}

/**
 * Shared dark-themed Plotly layout: transparent background, muted grid.
 * `equalAspect` (default true) locks a 1:1 x/y scale, appropriate for
 * spatial 2D point plots — turn it off for charts where the axes have
 * unrelated units (e.g. loss vs. epoch).
 */
export function basePlotlyLayout({ height = 420, xLabel = "x", yLabel = "y", equalAspect = true } = {}) {
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
      ...(equalAspect ? { scaleanchor: "y", scaleratio: 1 } : {}),
    },
    yaxis: { title: yLabel, zeroline: false, showgrid: false },
    legend: { orientation: "h", y: -0.15 },
  };
}
