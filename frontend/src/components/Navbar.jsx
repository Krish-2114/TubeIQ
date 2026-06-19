import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const links = [
  { to: '/', label: 'Home' },
  { to: '/analyze', label: 'Analyzer' },
  { to: '/improve', label: 'Inspiration' },
  { to: '/insights', label: 'Insights' },
  { to: '/channel', label: 'Channel' },
  { to: '/gaps', label: 'Content Gaps' },
  { to: '/ab-test', label: 'A/B Test' },
];

export default function Navbar() {
  const location = useLocation();
  const [open, setOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold tracking-tight">
          <span className="text-primary">Tube</span>
          <span style={{ color: '#ff0000' }}>IQ</span>
        </Link>

        <div className="hidden lg:flex items-center gap-8">
          {links.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`nav-link ${location.pathname === to ? 'active' : ''}`}
            >
              {label}
            </Link>
          ))}
          <Link to="/analyze" className="btn-primary px-4 py-2 text-sm">
            Analyze Title
          </Link>
          <Link to="/channel" className="btn-secondary px-4 py-2 text-sm hidden xl:inline-block">
            Analyze Channel
          </Link>
        </div>

        <button
          className="lg:hidden btn-secondary px-3 py-2 text-sm"
          onClick={() => setOpen(!open)}
          aria-label="Menu"
        >
          {open ? 'Close' : 'Menu'}
        </button>
      </div>

      {open && (
        <div
          className="lg:hidden px-6 pb-4 flex flex-col gap-4 border-t"
          style={{ borderColor: '#e4e4e7' }}
        >
          {links.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`nav-link ${location.pathname === to ? 'active' : ''}`}
              onClick={() => setOpen(false)}
            >
              {label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
