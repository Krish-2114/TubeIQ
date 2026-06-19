import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  DAYS,
  analyzeChannel,
  formatViews,
  formatDuration,
  distributionToPercentages,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import StatMiniCard from '../components/StatMiniCard';
import PerformanceBadge from '../components/PerformanceBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';
import { ProbabilityChart, RedBarChart } from '../components/Charts';

const QUICK_CHANNELS = [
  '@mkbhd',
  '@MrBeast',
  '@CarryMinati',
  '@gordonramsay',
  '@LinusTechTips',
];

export default function ChannelAnalyzer() {
  const [searchParams] = useSearchParams();
  const [identifier, setIdentifier] = useState(searchParams.get('q') || '');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runAnalysis = async (value) => {
    const id = value.trim();
    if (!id) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await analyzeChannel(id);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setIdentifier(q);
      runAnalysis(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const handleAnalyze = () => runAnalysis(identifier);

  const { channel, performance, upload_insights, content_insights } = result || {};

  const dayPerfData = upload_insights?.day_performance
    ? Object.entries(upload_insights.day_performance).map(([label, value]) => ({
        label: label.slice(0, 3),
        value,
      }))
    : [];

  const distChart = performance?.distribution
    ? distributionToPercentages(performance.distribution)
    : {};

  const topKeywords = content_insights?.top_keywords_in_best_videos
    ? Object.entries(content_insights.top_keywords_in_best_videos)
    : [];

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="CHANNEL ANALYZER"
          title="Analyze Any Channel"
          subtitle="Enter a YouTube handle or channel ID to get personalized insights from that channel's own video history."
        />

        <GlassCard className="p-6 mb-8" accent="#ff0000">
          <label className="field-label">Channel handle or ID</label>
          <div className="flex flex-col sm:flex-row gap-3 mt-2">
            <input
              className="input-field flex-1"
              placeholder="@mkbhd or UC..."
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
            <button
              className="btn-primary px-8 py-3 text-sm flex items-center justify-center gap-2 shrink-0"
              onClick={handleAnalyze}
              disabled={loading || !identifier.trim()}
            >
              {loading && <LoadingSpinner />}
              {loading ? 'Analyzing...' : 'Analyze Channel'}
            </button>
          </div>
          <p className="text-xs text-muted mt-3 mb-3">
            Examples: @mkbhd, @MrBeast, or a channel ID starting with UC
          </p>
          <div className="flex flex-wrap gap-2">
            <span className="text-xs text-secondary mr-1 self-center">Try:</span>
            {QUICK_CHANNELS.map((handle) => (
              <button
                key={handle}
                type="button"
                className="keyword-pill text-xs cursor-pointer hover:border-red-300"
                disabled={loading}
                onClick={() => {
                  setIdentifier(handle);
                  runAnalysis(handle);
                }}
              >
                {handle}
              </button>
            ))}
          </div>
          {error && <p className="error-msg mt-4">{error}</p>}
        </GlassCard>

        {loading && (
          <GlassCard className="p-10 text-center">
            <LoadingSpinner size={32} />
            <p className="text-sm text-secondary mt-4">
              Fetching channel data from YouTube. New channels may take up to a minute.
            </p>
          </GlassCard>
        )}

        {!loading && !result && !error && (
          <EmptyState
            title="No channel analyzed yet"
            description="Enter a YouTube handle like @mkbhd and click Analyze Channel to see performance patterns, upload timing, and top content."
          />
        )}

        {result && channel && (
          <div className="space-y-8 fade-in">
            <GlassCard className="p-6" glow>
              <div className="flex flex-col sm:flex-row items-start gap-5">
                {channel.thumbnail_url && (
                  <img
                    src={channel.thumbnail_url}
                    alt={channel.title}
                    className="w-20 h-20 rounded-full border object-cover shrink-0"
                    style={{ borderColor: '#e4e4e7' }}
                  />
                )}
                <div className="flex-1">
                  <h2 className="section-heading text-2xl">{channel.title}</h2>
                  <p className="text-sm text-secondary mt-1">
                    {formatViews(channel.subscriber_count)} subscribers
                    {' · '}
                    {channel.total_videos_analyzed} videos analyzed
                  </p>
                </div>
              </div>
            </GlassCard>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatMiniCard
                label="Median Views"
                value={formatViews(performance.median_views)}
                accent="#ff0000"
              />
              <StatMiniCard
                label="Avg Engagement"
                value={`${performance.avg_engagement_rate}%`}
                accent="#22c55e"
              />
              <StatMiniCard
                label="Viral Videos"
                value={content_insights.viral_count}
                accent="#a855f7"
              />
              <StatMiniCard
                label="Uploads / Week"
                value={
                  upload_insights.uploads_per_week != null
                    ? upload_insights.uploads_per_week
                    : '—'
                }
                accent="#3b82f6"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <GlassCard className="p-6">
                <p className="field-label mb-4">Performance Distribution</p>
                {Object.keys(distChart).length > 0 ? (
                  <ProbabilityChart probabilities={distChart} height={200} />
                ) : (
                  <p className="text-sm text-secondary">No distribution data.</p>
                )}
                <div className="flex flex-wrap gap-3 mt-4 pt-4 border-t" style={{ borderColor: '#f4f4f5' }}>
                  {Object.entries(performance.distribution || {}).map(([label, count]) => (
                    <div key={label} className="flex items-center gap-2 text-sm">
                      <PerformanceBadge label={label} />
                      <span className="text-secondary">{count} videos</span>
                    </div>
                  ))}
                </div>
              </GlassCard>

              <GlassCard className="p-6">
                <p className="field-label mb-3">Upload Timing</p>
                {upload_insights.best_upload_day && (
                  <>
                    <p className="text-sm text-secondary mb-4">
                      Best day:{' '}
                      <span className="text-primary font-semibold">
                        {upload_insights.best_upload_day}
                      </span>
                      {upload_insights.optimal_duration && (
                        <>
                          {' · '}
                          Best duration:{' '}
                          <span className="text-primary font-semibold">
                            {upload_insights.optimal_duration}
                          </span>
                        </>
                      )}
                    </p>
                    <div className="flex flex-wrap gap-2 mb-5">
                      {DAYS.map((d) => (
                        <span
                          key={d}
                          className={`day-pill ${
                            d === upload_insights.best_upload_day ? 'active' : ''
                          }`}
                        >
                          {d}
                        </span>
                      ))}
                    </div>
                  </>
                )}
                {dayPerfData.length > 0 && (
                  <RedBarChart data={dayPerfData} height={180} />
                )}
              </GlassCard>
            </div>

            {topKeywords.length > 0 && (
              <GlassCard className="p-6">
                <p className="field-label mb-2">Top Keywords in Best Videos</p>
                <p className="text-sm text-secondary mb-4">
                  Words that appear most in this channel&apos;s high-performing titles.
                </p>
                <div className="flex flex-wrap gap-2">
                  {topKeywords.map(([word, count]) => (
                    <span key={word} className="keyword-pill">
                      {word} ({count})
                    </span>
                  ))}
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6">
                  <div className="p-4 inset-panel">
                    <div className="text-sm text-secondary">High-performing videos</div>
                    <div className="text-2xl font-bold text-primary mt-1">
                      {content_insights.high_performing_count}
                    </div>
                  </div>
                  <div className="p-4 inset-panel">
                    <div className="text-sm text-secondary">Viral hits</div>
                    <div className="text-2xl font-bold mt-1" style={{ color: '#a855f7' }}>
                      {content_insights.viral_count}
                    </div>
                  </div>
                </div>
              </GlassCard>
            )}

            <GlassCard className="p-6">
              <p className="field-label mb-4">Top Performing Videos</p>
              <div className="overflow-x-auto -mx-2">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Title</th>
                      <th>Views</th>
                      <th>Duration</th>
                      <th>Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(performance.top_videos || []).map((video, i) => (
                      <tr key={i}>
                        <td className="text-muted">{i + 1}</td>
                        <td className="max-w-[280px]">
                          <span className="line-clamp-2">{video.title}</span>
                        </td>
                        <td className="font-medium">{formatViews(video.view_count)}</td>
                        <td className="text-secondary">
                          {formatDuration(video.duration_seconds)}
                        </td>
                        <td>
                          <PerformanceBadge label={video.performance_label} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlassCard>
          </div>
        )}
      </div>
    </PageLayout>
  );
}
