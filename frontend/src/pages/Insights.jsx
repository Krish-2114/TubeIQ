import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  NICHES,
  DAYS,
  NICHE_GRADIENTS,
  fetchInsights,
  formatViews,
  capitalize,
  getDurationChartData,
  getDisplayOptimalDuration,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import StatMiniCard from '../components/StatMiniCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { RedBarChart } from '../components/Charts';

const TITLE_LENGTH_LABELS = {
  very_short: 'Very Short',
  short: 'Short',
  medium: 'Medium',
  long: 'Long',
};

export default function Insights() {
  const [searchParams] = useSearchParams();
  const [niche, setNiche] = useState(searchParams.get('niche') || 'gaming');
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const param = searchParams.get('niche');
    if (param && NICHES.includes(param)) setNiche(param);
  }, [searchParams]);

  useEffect(() => {
    setLoading(true);
    setError('');
    fetchInsights(niche)
      .then(setInsights)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [niche]);

  const dayPerfData = insights?.upload_day_performance
    ? Object.entries(insights.upload_day_performance).map(([label, value]) => ({
        label: label.slice(0, 3),
        value,
      }))
    : [];

  const durPerfData = getDurationChartData(insights?.duration_performance);

  const titleLengthData = insights?.title_length_performance
    ? Object.entries(insights.title_length_performance).map(([key, value]) => ({
        label: TITLE_LENGTH_LABELS[key] || key,
        value,
      }))
    : [];

  const topKeywords = insights?.top_keywords
    ? Object.entries(insights.top_keywords).slice(0, 15)
    : [];

  const optimalDuration = insights ? getDisplayOptimalDuration(insights) : '';
  const optimalTitleLength = insights?.optimal_title_length
    ? capitalize(insights.optimal_title_length)
    : '—';

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="INSIGHTS"
          title="Niche Intelligence"
          subtitle="Select a niche for detailed performance patterns — upload timing, duration, title structure, and keywords."
        />

        <GlassCard className="p-6 mb-8">
          <label className="field-label">Niche</label>
          <select
            className="select-field max-w-xs"
            value={niche}
            onChange={(e) => setNiche(e.target.value)}
          >
            {NICHES.map((n) => (
              <option key={n} value={n}>{capitalize(n)}</option>
            ))}
          </select>
        </GlassCard>

        {loading && (
          <div className="flex justify-center py-20">
            <LoadingSpinner size={32} />
          </div>
        )}

        {error && <p className="error-msg">{error}</p>}

        {insights && !loading && (
          <div className="space-y-8 fade-in">
            <div className="flex items-center gap-4">
              <div
                className="h-1 flex-1 rounded-full"
                style={{ background: NICHE_GRADIENTS[niche] }}
              />
              <h2 className="section-heading text-xl shrink-0">{capitalize(niche)}</h2>
              <div
                className="h-1 flex-1 rounded-full"
                style={{ background: NICHE_GRADIENTS[niche] }}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatMiniCard
                label="Median Views"
                value={formatViews(insights.niche_median_views)}
                accent="#ff0000"
              />
              <StatMiniCard
                label="Avg Engagement"
                value={`${insights.avg_engagement_rate}%`}
                accent="#22c55e"
              />
              <StatMiniCard
                label="Best Upload Day"
                value={insights.best_upload_day}
                accent="#a855f7"
              />
              <StatMiniCard
                label="Optimal Duration"
                value={optimalDuration}
                accent="#f97316"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatMiniCard
                label="Optimal Title Length"
                value={optimalTitleLength}
                accent="#3b82f6"
              />
              <StatMiniCard
                label="Videos Analyzed"
                value={insights.total_videos_analyzed?.toLocaleString() || '—'}
                accent="#71717a"
              />
              <StatMiniCard
                label="Channels Analyzed"
                value={insights.channels_analyzed ?? '—'}
                accent="#71717a"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <GlassCard className="p-6">
                <p className="field-label mb-3">Upload Day Performance</p>
                <div className="flex flex-wrap gap-2 mb-5">
                  {DAYS.map((d) => (
                    <span
                      key={d}
                      className={`day-pill ${d === insights.best_upload_day ? 'active' : ''}`}
                    >
                      {d}
                    </span>
                  ))}
                </div>
                {dayPerfData.length > 0 && (
                  <RedBarChart data={dayPerfData} height={200} />
                )}
              </GlassCard>

              <GlassCard className="p-6">
                <p className="field-label mb-1">Duration Performance</p>
                <p className="text-sm text-secondary mb-5">
                  Best bucket: <span className="text-primary font-medium">{optimalDuration}</span>
                </p>
                {durPerfData.length > 0 && (
                  <RedBarChart data={durPerfData} height={200} />
                )}
              </GlassCard>
            </div>

            {titleLengthData.length > 0 && (
              <GlassCard className="p-6">
                <p className="field-label mb-1">Title Length Performance</p>
                <p className="text-sm text-secondary mb-5">
                  Best length:{' '}
                  <span className="text-primary font-medium">
                    {optimalTitleLength}
                  </span>
                </p>
                <RedBarChart data={titleLengthData} height={200} />
              </GlassCard>
            )}

            <GlassCard className="p-6">
              <p className="field-label mb-4">Top Keywords in High Performers</p>
              {topKeywords.length === 0 ? (
                <p className="text-sm text-secondary">No keyword data available.</p>
              ) : (
                <div className="flex flex-wrap gap-2 mb-8">
                  {topKeywords.map(([word, count]) => (
                    <span key={word} className="keyword-pill">{word} ({count})</span>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { label: 'Question Title Boost', value: insights.question_title_boost },
                  { label: 'Number in Title Boost', value: insights.number_in_title_boost },
                ].map((s) => (
                  <div key={s.label} className="p-4 inset-panel">
                    <div className="text-sm text-secondary">{s.label}</div>
                    <div
                      className="text-2xl font-bold mt-1"
                      style={{ color: s.value > 1 ? '#22c55e' : '#ff4444' }}
                    >
                      {s.value}x
                    </div>
                    <p className="text-xs text-muted mt-2">
                      {s.value > 1
                        ? 'Titles with this pattern tend to outperform the niche average.'
                        : 'This pattern underperforms relative to other titles in the niche.'}
                    </p>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>
        )}
      </div>
    </PageLayout>
  );
}
