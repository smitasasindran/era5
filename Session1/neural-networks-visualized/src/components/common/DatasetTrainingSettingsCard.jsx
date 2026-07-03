import { FiRefreshCw, FiTrash2 } from "react-icons/fi";

import Card from "./Card";
import SecondaryButton from "./SecondaryButton";
import SliderField from "./SliderField";

/**
 * Reusable "Dataset & Training Settings" card: point/noise sliders for the
 * shared dataset plus epoch/learning-rate/batch-size sliders for training,
 * and Regenerate Dataset / Reset Results actions. Every experiment that
 * compares several models trained on one generated dataset reuses this
 * instead of re-declaring the same sliders.
 */
function DatasetTrainingSettingsCard({
  numPoints,
  setNumPoints,
  noise,
  setNoise,
  epochs,
  setEpochs,
  learningRate,
  setLearningRate,
  batchSize,
  setBatchSize,
  onRegenerateDataset,
  onResetResults,
  disabled,
  footnote = "All models use the same settings above for a fair comparison.",
}) {
  return (
    <Card title="Dataset & Training Settings">
      <div className="mt-4 flex flex-col gap-5 lg:flex-row lg:items-end">
        <div className="grid flex-1 grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5">
          <SliderField
            label="Points"
            value={numPoints}
            displayValue={numPoints}
            onChange={setNumPoints}
            min={50}
            max={600}
            step={10}
            disabled={disabled}
          />
          <SliderField
            label="Noise"
            value={noise}
            displayValue={noise.toFixed(2)}
            onChange={setNoise}
            min={0}
            max={1}
            step={0.02}
            disabled={disabled}
          />
          <SliderField
            label="Epochs"
            value={epochs}
            displayValue={epochs}
            onChange={setEpochs}
            min={10}
            max={300}
            step={10}
            disabled={disabled}
          />
          <SliderField
            label="Learning rate"
            value={learningRate}
            displayValue={learningRate.toFixed(3)}
            onChange={setLearningRate}
            min={0.001}
            max={0.3}
            step={0.001}
            disabled={disabled}
          />
          <SliderField
            label="Batch size"
            value={batchSize}
            displayValue={batchSize}
            onChange={setBatchSize}
            min={8}
            max={128}
            step={8}
            disabled={disabled}
          />
        </div>

        <div className="flex shrink-0 flex-col gap-2 lg:w-48">
          <SecondaryButton icon={FiRefreshCw} onClick={onRegenerateDataset} disabled={disabled}>
            Regenerate Dataset
          </SecondaryButton>
          <SecondaryButton icon={FiTrash2} onClick={onResetResults} disabled={disabled}>
            Reset Results
          </SecondaryButton>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">{footnote}</p>
    </Card>
  );
}

export default DatasetTrainingSettingsCard;
