/**
 * "Model / Boundary / Accuracy" comparison table used by experiment pages
 * that train several models on the same data and compare outcomes.
 *
 * @param {Array<{model: string, boundary: string, accuracy: string}>} rows
 */
function ComparisonTable({ rows }) {
  return (
    <table className="mt-3 w-full text-left text-sm">
      <thead>
        <tr className="text-xs uppercase tracking-wide text-slate-500">
          <th className="pb-2 font-medium">Model</th>
          <th className="pb-2 font-medium">Boundary</th>
          <th className="pb-2 text-right font-medium">Accuracy</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {rows.map((row) => (
          <tr key={row.model}>
            <td className="py-2 text-slate-300">{row.model}</td>
            <td className="py-2 text-slate-400">{row.boundary}</td>
            <td className="py-2 text-right font-semibold text-slate-100">{row.accuracy}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default ComparisonTable;
