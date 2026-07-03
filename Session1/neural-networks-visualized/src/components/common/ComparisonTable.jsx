// Default shape used by the Model/Boundary/Accuracy tables in earlier
// chapters — kept as the default so those call sites don't need `columns`.
const DEFAULT_COLUMNS = [
  { key: "model", label: "Model" },
  { key: "boundary", label: "Boundary" },
  { key: "accuracy", label: "Accuracy", align: "right" },
];

/**
 * Results table used by experiment pages that train several models or
 * configurations and compare outcomes. `columns` defines the header cells,
 * in order; each entry in `rows` is a plain object keyed by each column's
 * `key`. The first column is always styled as the row's identity; any
 * column with `align: "right"` is right-aligned and emphasized (typically
 * the headline metric).
 *
 * @param {Array<{key: string, label: string, align?: "right"}>} [columns]
 * @param {Array<Object>} rows
 */
function ComparisonTable({ columns = DEFAULT_COLUMNS, rows }) {
  return (
    <table className="mt-3 w-full text-left text-sm">
      <thead>
        <tr className="text-xs uppercase tracking-wide text-slate-500">
          {columns.map((column) => (
            <th
              key={column.key}
              className={`pb-2 font-medium ${column.align === "right" ? "text-right" : ""}`}
            >
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {rows.map((row, rowIndex) => (
          <tr key={row.id ?? rowIndex}>
            {columns.map((column, colIndex) => {
              const cellClassName =
                colIndex === 0
                  ? "py-2 text-slate-300"
                  : column.align === "right"
                    ? "py-2 text-right font-semibold text-slate-100"
                    : "py-2 text-slate-400";
              return (
                <td key={column.key} className={cellClassName}>
                  {row[column.key]}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default ComparisonTable;
