import { useEffect, useState } from 'react';

export default function StatCounter({ target, suffix = '', label, delay = 0 }) {
  const [count, setCount] = useState(0);
  const numericTarget = parseInt(target.replace(/\D/g, ''), 10);

  useEffect(() => {
    const timeout = setTimeout(() => {
      let start = 0;
      const duration = 1500;
      const step = numericTarget / (duration / 16);
      const timer = setInterval(() => {
        start += step;
        if (start >= numericTarget) {
          setCount(numericTarget);
          clearInterval(timer);
        } else {
          setCount(Math.floor(start));
        }
      }, 16);
      return () => clearInterval(timer);
    }, delay);
    return () => clearTimeout(timeout);
  }, [numericTarget, delay]);

  const display = target.includes('+')
    ? `${count.toLocaleString()}+`
    : count.toLocaleString();

  return (
    <div
      className="glass-card stat-counter p-4 flex-1"
      style={{ borderTop: '3px solid #ff0000', animationDelay: `${delay}ms` }}
    >
      <div className="text-2xl font-bold text-white">{display}{suffix}</div>
      <div className="text-sm mt-1" style={{ color: '#a1a1aa' }}>{label}</div>
    </div>
  );
}
