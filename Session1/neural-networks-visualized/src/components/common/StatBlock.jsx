/**
 * A small "LABEL / big value" stat readout, used across training result
 * cards (accuracy, loss, generalization gap, ...).
 */
function StatBlock({ label, value, size = "lg" }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={size === "lg" ? "text-2xl font-bold text-slate-100" : "text-lg font-bold text-slate-100"}>
        {value}
      </p>
    </div>
  );
}

export default StatBlock;
