const STYLES = {
  Low: {
    background: 'rgba(239,68,68,0.15)',
    color: '#ef4444',
    border: '1px solid rgba(239,68,68,0.3)',
  },
  Medium: {
    background: 'rgba(234,179,8,0.15)',
    color: '#eab308',
    border: '1px solid rgba(234,179,8,0.3)',
  },
  High: {
    background: 'rgba(34,197,94,0.15)',
    color: '#22c55e',
    border: '1px solid rgba(34,197,94,0.3)',
  },
  Viral: {
    background: 'rgba(168,85,247,0.15)',
    color: '#a855f7',
    border: '1px solid rgba(168,85,247,0.3)',
  },
};

export default function PerformanceBadge({ label, size = 'md' }) {
  const style = STYLES[label] || STYLES.Medium;
  const padding = size === 'lg' ? '12px 24px' : '6px 14px';
  const fontSize = size === 'lg' ? '18px' : '13px';

  return (
    <span
      className="label-badge inline-block rounded-lg"
      style={{ ...style, padding, fontSize }}
    >
      {label}
    </span>
  );
}
