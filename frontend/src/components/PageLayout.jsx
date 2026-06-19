export default function PageLayout({ children, mesh = false, className = '' }) {
  return (
    <div className={`page-layout relative min-h-screen ${className}`}>
      {mesh ? <div className="gradient-mesh" /> : <div className="page-grid-bg" />}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
