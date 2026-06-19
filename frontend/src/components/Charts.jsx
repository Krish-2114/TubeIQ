import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import { BAR_GRADIENTS, LABEL_COLORS } from '../utils';

const tooltipStyle = {
  background: '#ffffff',
  border: '1px solid #e4e4e7',
  borderRadius: 8,
  color: '#18181b',
  boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
};

const axisStyle = { fill: '#71717a', fontSize: 12 };
const gridStroke = 'rgba(0,0,0,0.06)';

export function ProbabilityChart({ probabilities, vertical = true, height = 220 }) {
  const data = Object.entries(probabilities || {}).map(([label, value]) => ({
    label,
    value: value <= 1 ? value * 100 : value,
    fill: LABEL_COLORS[label] || '#ff0000',
  }));

  if (vertical) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <defs>
            {data.map((d) => (
              <linearGradient key={d.label} id={`grad-${d.label}`} x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor={BAR_GRADIENTS[d.label]?.[0] || '#ff0000'} />
                <stop offset="100%" stopColor={BAR_GRADIENTS[d.label]?.[1] || '#990000'} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid stroke={gridStroke} horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={axisStyle} unit="%" />
          <YAxis type="category" dataKey="label" tick={axisStyle} width={60} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v) => [`${v.toFixed(1)}%`, 'Probability']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} animationDuration={800}>
            {data.map((d) => (
              <Cell key={d.label} fill={`url(#grad-${d.label})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, bottom: 5 }}>
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v) => [`${v.toFixed(1)}%`, 'Probability']}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} animationDuration={800}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RedBarChart({ data, dataKey = 'value', nameKey = 'label', height = 200 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="redGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff0000" />
            <stop offset="100%" stopColor="#990000" />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={gridStroke} vertical={false} />
        <XAxis dataKey={nameKey} tick={{ ...axisStyle, fontSize: 11 }} />
        <YAxis tick={axisStyle} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey={dataKey} fill="url(#redGrad)" radius={[4, 4, 0, 0]} animationDuration={800} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function PreviewBarChart() {
  const data = [
    { label: 'Viral', value: 15, fill: '#a855f7' },
    { label: 'High', value: 78, fill: '#22c55e' },
    { label: 'Medium', value: 5, fill: '#eab308' },
    { label: 'Low', value: 2, fill: '#ef4444' },
  ];

  return (
    <ResponsiveContainer width="100%" height={100}>
      <BarChart data={data}>
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, 'Prob']} />
        <Bar dataKey="value" radius={[3, 3, 0, 0]} animationDuration={1200}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
