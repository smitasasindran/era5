import SliderField from "./SliderField";

// Bounds chosen so training stays fast and numerically well-behaved for
// the tiny 2D toy models used across the course — verified empirically up
// to the max on both plain and ReLU stacks.
const MIN_NEURONS = 2;
const MAX_NEURONS = 32;
const STEP = 2;

/**
 * Slider controlling a layer's width (units/neurons). A reusable building
 * block for any experiment that lets visitors resize a model's capacity.
 */
function NeuronCountControl({ value, onChange, disabled, label = "Neurons per layer" }) {
  return (
    <SliderField
      label={label}
      value={value}
      displayValue={value}
      onChange={onChange}
      min={MIN_NEURONS}
      max={MAX_NEURONS}
      step={STEP}
      disabled={disabled}
    />
  );
}

export default NeuronCountControl;
