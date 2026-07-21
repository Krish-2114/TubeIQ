import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="border-t mt-20 bg-white" style={{ borderColor: '#e4e4e7' }}>
      <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="text-lg font-bold">
          <span className="text-primary">Tube</span>
          <span style={{ color: '#ff0000' }}>IQ</span>
        </div>
        <p className="text-sm text-muted">YouTube channel growth intelligence.</p>
        <div className="flex flex-wrap gap-6 justify-center">
          <Link to="/analyze" className="nav-link text-sm">Analyzer</Link>
          <Link to="/insights" className="nav-link text-sm">Insights</Link>
          <Link to="/channel" className="nav-link text-sm">Channel</Link>
          <Link to="/gaps" className="nav-link text-sm">Content Gaps</Link>
          <Link to="/ab-test" className="nav-link text-sm">A/B Test</Link>
        </div>
      </div>
    </footer>
  );
}
