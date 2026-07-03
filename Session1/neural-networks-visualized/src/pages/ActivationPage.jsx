import { useMemo, useState } from "react";

import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";
import PrimaryButton from "../components/common/PrimaryButton";
import ScatterPlot from "../components/common/ScatterPlot";
import { generateConcentricRings } from "../utils/datasets";

function ActivationPage() {
  const [numPoints, setNumPoints] = useState(300);
  const [noise, setNoise] = useState(0.2);
  const [seed, setSeed] = useState(0);

  // `seed` isn't used inside the generator — bumping it just forces a fresh
  // random draw with the same numPoints/noise when "Regenerate" is clicked.
  const data = useMemo(
    () => generateConcentricRings({ numPoints, noise }),
    [numPoints, noise, seed],
  );

  return (
    <ExperimentLayout
      title="Activation Functions"
      claim="Non-linear activations let networks learn boundaries a linear model cannot."
    >
      <Card title="Overview">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Linear models can only draw straight decision boundaries. The dataset
          below — two concentric, noisy rings — has no straight line that
          separates its two classes, which is exactly the kind of problem
          non-linear activations are built to solve.
        </p>
      </Card>

      <Card title="Interactive Demo">
        <div className="mt-4 flex flex-col gap-6">
          <ScatterPlot data={data} />

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Points</span>
                <span className="text-slate-200">{numPoints}</span>
              </span>
              <input
                type="range"
                min={50}
                max={600}
                step={10}
                value={numPoints}
                onChange={(event) => setNumPoints(Number(event.target.value))}
                className="accent-accent-500"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-400">
              <span className="flex justify-between">
                <span>Noise</span>
                <span className="text-slate-200">{noise.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.02}
                value={noise}
                onChange={(event) => setNoise(Number(event.target.value))}
                className="accent-accent-500"
              />
            </label>
          </div>

          <PrimaryButton onClick={() => setSeed((s) => s + 1)} className="self-start">
            Regenerate dataset
          </PrimaryButton>
        </div>
      </Card>

      <Card title="What to Observe">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Try increasing the noise until the two rings start to overlap —
          notice how much harder the boundary between them becomes to
          describe with a simple rule.
        </p>
      </Card>

      <Card title="Why It Matters">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Every experiment on this page will try to separate these same two
          classes. How well that works depends heavily on the activation
          function a network uses between its layers.
        </p>
      </Card>

      <Card title="Summary">
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          This dataset — not linearly separable, but clearly structured — is
          the shared starting point for exploring activation functions in
          this experiment.
        </p>
      </Card>
    </ExperimentLayout>
  );
}

export default ActivationPage;
