import PerformanceBadge from './PerformanceBadge';
import { probToPercent } from '../utils/api';

/**
 * Explains XGBoost output: confidence = predict_proba for the winning class.
 */
export default function PredictionCertainty({
  label,
  confidence,
  showRing = false,
  size = 'md',
}) {
  const pct = probToPercent(confidence);

  return (
    <div className="flex flex-col sm:flex-row items-center gap-6">
      {showRing && (
        <div className="flex flex-col items-center gap-2 shrink-0">
          <div
            className="confidence-ring"
            style={{ '--pct': `${pct}%` }}
          >
            <span>{pct}%</span>
          </div>
          <span className="label-badge text-muted">Class probability</span>
        </div>
      )}

      <div className="flex-1">
        <p className="field-label mb-2">Predicted performance tier</p>
        <PerformanceBadge label={label} size={size === 'lg' ? 'lg' : 'md'} />

        <p className="text-sm text-secondary mt-4 leading-relaxed">
          The model assigns{' '}
          <span className="font-semibold text-primary">{pct}%</span> probability to{' '}
          <span className="font-semibold text-primary">{label}</span>{' '}
          — its most likely class out of Low, Medium, High, and Viral.
        </p>

        <p className="text-xs text-muted mt-2 leading-relaxed">
          This is the model&apos;s internal certainty for that class, not a guarantee
          of views or real-world performance. Use the breakdown below to see all class
          probabilities.
        </p>
      </div>
    </div>
  );
}
