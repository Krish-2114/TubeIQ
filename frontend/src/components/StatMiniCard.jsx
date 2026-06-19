export default function StatMiniCard({ label, value, accent }) {
  return (
    <div
      className="glass-card p-5"
      style={accent ? { borderLeft: `3px solid ${accent}` } : undefined}
    >
      <div className="text-2xl md:text-3xl font-bold text-primary">{value}</div>
      <div className="text-sm mt-1 text-secondary">{label}</div>
    </div>
  );
}
