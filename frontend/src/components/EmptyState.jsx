export default function EmptyState({ title, description }) {
  return (
    <div
      className="glass-card p-10 text-center"
      style={{ borderStyle: 'dashed' }}
    >
      <div
        className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
        style={{ background: 'rgba(255,0,0,0.1)', border: '1px solid rgba(255,0,0,0.2)' }}
      >
        <span style={{ color: '#ff0000', fontSize: 20 }}>+</span>
      </div>
      <h3 className="section-heading text-lg mb-2">{title}</h3>
      <p className="text-sm max-w-md mx-auto" style={{ color: '#a1a1aa' }}>
        {description}
      </p>
    </div>
  );
}
