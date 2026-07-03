/**
 * Labeled range input showing its current value, used for every dataset
 * and hyperparameter control across experiment pages.
 */
function SliderField({ label, value, displayValue, onChange, min, max, step, disabled }) {
  return (
    <label className="flex flex-col gap-2 text-sm text-slate-400">
      <span className="flex justify-between">
        <span>{label}</span>
        <span className="text-slate-200">{displayValue}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        disabled={disabled}
        className="accent-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
      />
    </label>
  );
}

export default SliderField;
