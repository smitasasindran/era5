import { motion } from "framer-motion";

/**
 * Animated illustration of several linear transformations collapsing into
 * one: a row of small "matrix" chips connected by ×, converging into a
 * single highlighted result chip. Purely decorative — animates once when
 * it scrolls into view, no props required beyond the labels shown.
 */
function MatrixCollapseDiagram({ labels = ["W₁", "W₂", "W₃", "W₄", "W₅"], resultLabel = "One Matrix" }) {
  return (
    <div className="flex flex-col items-center gap-5 py-4">
      <div className="flex flex-wrap items-center justify-center gap-3">
        {labels.map((label, index) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: -8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.08, duration: 0.4 }}
            className="flex items-center gap-3"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-lg border border-accent-500/40 bg-accent-500/10 text-sm font-semibold text-accent-200">
              {label}
            </div>
            {index < labels.length - 1 && <span className="text-lg text-slate-500">×</span>}
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: labels.length * 0.08, duration: 0.4 }}
        className="text-2xl text-slate-600"
      >
        ↓
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.85 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ delay: labels.length * 0.08 + 0.15, duration: 0.45 }}
        className="flex h-16 w-44 items-center justify-center rounded-lg border border-accent-400/60 bg-accent-500/15 text-sm font-semibold text-accent-100 shadow-lg shadow-accent-900/30"
      >
        {resultLabel}
      </motion.div>
    </div>
  );
}

export default MatrixCollapseDiagram;
