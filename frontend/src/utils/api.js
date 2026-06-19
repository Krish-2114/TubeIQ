export const API_BASE = 'http://localhost:8000';

export const NICHES = [
  'gaming',
  'entertainment',
  'cooking',
  'finance',
  'fitness',
  'education',
  'tech',
  'streaming',
];

export const NICHE_GRADIENTS = {
  gaming: 'linear-gradient(90deg, #ff0000, #ff6600)',
  entertainment: 'linear-gradient(90deg, #ff0000, #ff00aa)',
  cooking: 'linear-gradient(90deg, #ff6600, #ffaa00)',
  finance: 'linear-gradient(90deg, #00aa44, #00ff88)',
  fitness: 'linear-gradient(90deg, #ff4400, #ff0000)',
  education: 'linear-gradient(90deg, #0066ff, #00aaff)',
  tech: 'linear-gradient(90deg, #aa00ff, #ff00aa)',
  streaming: 'linear-gradient(90deg, #ff0000, #aa00ff)',
};

export const DAYS = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];

export const HOURS = Array.from({ length: 19 }, (_, i) => i + 6);

export const LABEL_RANK = { Viral: 4, High: 3, Medium: 2, Low: 1 };

export const LABEL_COLORS = {
  Low: '#ef4444',
  Medium: '#eab308',
  High: '#22c55e',
  Viral: '#a855f7',
};

export const BAR_GRADIENTS = {
  Low: ['#ef4444', '#991b1b'],
  Medium: ['#eab308', '#854d0e'],
  High: ['#22c55e', '#166534'],
  Viral: ['#a855f7', '#581c87'],
};

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function fetchInsights(niche) {
  return apiGet(`/insights/${niche}`);
}

export function predictTitle(data) {
  return apiPost('/predict', data);
}

export function fetchSimilar(data) {
  return apiPost('/similar', data);
}

export function fetchGaps(niche) {
  return apiPost('/gaps', { niche });
}

export function analyzeChannel(identifier) {
  return apiPost('/channel/analyze', { identifier: identifier.trim() });
}

export function improveTitle(title, niche) {
  return apiPost('/improve', { title: title.trim(), niche });
}

export function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '—';
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

export function distributionToPercentages(distribution) {
  if (!distribution) return {};
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  if (total === 0) return {};
  const order = ['Viral', 'High', 'Medium', 'Low'];
  return Object.fromEntries(
    order
      .filter((label) => distribution[label])
      .map((label) => [label, distribution[label] / total])
  );
}

export function formatViews(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(n);
}

export function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function probToPercent(p) {
  return p <= 1 ? Math.round(p * 1000) / 10 : Math.round(p * 10) / 10;
}

export function getDurationChartData(durationPerformance) {
  if (!durationPerformance) return [];
  return Object.entries(durationPerformance)
    .filter(([label]) => label !== '0-2min')
    .map(([label, value]) => ({ label, value }));
}

export function getDisplayOptimalDuration(insights) {
  const chartData = getDurationChartData(insights?.duration_performance);
  if (chartData.length > 0) {
    return chartData.reduce((best, cur) =>
      cur.value > best.value ? cur : best
    ).label;
  }
  const dur = insights?.optimal_duration;
  return dur && dur !== '0-2min' ? dur : '2-10min';
}
