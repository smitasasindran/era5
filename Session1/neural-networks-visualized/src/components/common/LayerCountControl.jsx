import SliderField from "./SliderField";

// Bounds chosen so training stays fast and numerically well-behaved for
// the tiny 2D toy models used across the course — a plain (no-activation)
// stack of many linear layers is prone to exploding outputs at higher
// counts, so this caps out at 10 rather than going arbitrarily deep.
const MIN_LAYERS = 2;
const MAX_LAYERS = 10;
const STEP = 1;

/**
 * Slider controlling a stacked model's total number of Dense layers
 * (including its final output layer). A reusable building block for any
 * experiment that lets visitors resize a model's depth.
 */
function LayerCountControl({ value, onChange, disabled, label = "Number of layers" }) {
  return (
    <SliderField
      label={label}
      value={value}
      displayValue={value}
      onChange={onChange}
      min={MIN_LAYERS}
      max={MAX_LAYERS}
      step={STEP}
      disabled={disabled}
    />
  );
}

export default LayerCountControl;
