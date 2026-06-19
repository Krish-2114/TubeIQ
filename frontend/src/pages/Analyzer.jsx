import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  NICHES,
  DAYS,
  HOURS,
  predictTitle,
  fetchSimilar,
  formatViews,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import EmptyState from '../components/EmptyState';
import PerformanceBadge from '../components/PerformanceBadge';
import PredictionCertainty from '../components/PredictionCertainty';
import LoadingSpinner from '../components/LoadingSpinner';
import { ProbabilityChart } from '../components/Charts';

export default function Analyzer() {
  const [searchParams] = useSearchParams();
  const [niche, setNiche] = useState(searchParams.get('niche') || 'gaming');
  const [title, setTitle] = useState('');
  const [duration, setDuration] = useState(10);
  const [uploadDay, setUploadDay] = useState(5);
  const [uploadHour, setUploadHour] = useState(18);

  const [prediction, setPrediction] = useState(null);
  const [similar, setSimilar] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState('');
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    const param = searchParams.get('niche');
    if (param && NICHES.includes(param)) setNiche(param);
  }, [searchParams]);

  const handleAnalyze = async () => {
    if (!title.trim()) return;
    setAnalyzing(true);
    setAnalyzeError('');
    setShowResults(false);
    try {
      const body = {
        title: title.trim(),
        niche,
        duration_seconds: duration * 60,
        upload_day: uploadDay,
        upload_hour: uploadHour,
      };
      const [pred, sim] = await Promise.all([
        predictTitle(body),
        fetchSimilar({ title: title.trim(), niche, top_n: 5 }),
      ]);
      setPrediction(pred);
      setSimilar(sim);
      setShowResults(true);
    } catch (e) {
      setAnalyzeError(e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="ANALYZER"
          title="Title Performance Analyzer"
          subtitle="Enter a title idea and get a performance prediction with similar high-performing titles."
        />

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-8">
          {/* Input */}
          <div className="xl:col-span-2">
            <GlassCard className="p-6 sticky top-24" accent="#ff0000">
              <div className="mb-5">
                <label className="field-label">Niche</label>
                <select className="select-field" value={niche} onChange={(e) => setNiche(e.target.value)}>
                  {NICHES.map((n) => (
                    <option key={n} value={n}>{capitalize(n)}</option>
                  ))}
                </select>
              </div>

              <div className="mb-5">
                <label className="field-label">Video Title</label>
                <input
                  className="input-field"
                  placeholder="Enter your video title idea..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                />
              </div>

              <div className="grid grid-cols-1 gap-4 mb-6">
                <div>
                  <label className="field-label">Duration (minutes)</label>
                  <input
                    type="number"
                    className="input-field"
                    min={1}
                    max={120}
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="field-label">Upload Day</label>
                  <select className="select-field" value={uploadDay} onChange={(e) => setUploadDay(Number(e.target.value))}>
                    {DAYS.map((d, i) => (
                      <option key={d} value={i}>{d}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="field-label">Upload Hour</label>
                  <select className="select-field" value={uploadHour} onChange={(e) => setUploadHour(Number(e.target.value))}>
                    {HOURS.map((h) => (
                      <option key={h} value={h}>
                        {h === 12 ? '12 PM' : h < 12 ? `${h} AM` : `${h - 12} PM`}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                className="btn-primary w-full py-3.5 flex items-center justify-center gap-2 text-sm"
                onClick={handleAnalyze}
                disabled={analyzing || !title.trim()}
              >
                {analyzing && <LoadingSpinner />}
                {analyzing ? 'Analyzing...' : 'Analyze Title'}
              </button>
              {analyzeError && <p className="error-msg">{analyzeError}</p>}
            </GlassCard>
          </div>

          {/* Results */}
          <div className="xl:col-span-3 space-y-6">
            {!showResults && (
              <EmptyState
                title="No analysis yet"
                description="Enter a title and click Analyze to see performance prediction, similar high-performing titles, and probability breakdown."
              />
            )}

            {showResults && prediction && (
              <div className="fade-in space-y-6">
                <GlassCard className="p-6" glow>
                  <h3 className="section-heading text-lg mb-6">Prediction Result</h3>
                  <PredictionCertainty
                    label={prediction.label}
                    confidence={prediction.confidence}
                    showRing
                    size="lg"
                  />
                  <div className="mt-8 pt-6 border-t" style={{ borderColor: '#f4f4f5' }}>
                    <p className="field-label mb-4">All class probabilities</p>
                    <ProbabilityChart probabilities={prediction.probabilities} height={200} />
                  </div>
                </GlassCard>

                <GlassCard className="p-6">
                  <h3 className="section-heading text-lg mb-6">Similar High-Performing Titles</h3>
                  {(similar?.similar_titles || []).length === 0 ? (
                    <p className="text-sm" style={{ color: '#a1a1aa' }}>No similar titles found.</p>
                  ) : (
                    <div className="overflow-x-auto -mx-2">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Title</th>
                            <th>Channel</th>
                            <th>Views</th>
                            <th>Label</th>
                            <th>Match</th>
                          </tr>
                        </thead>
                        <tbody>
                          {similar.similar_titles.map((row, i) => (
                            <tr key={i}>
                              <td className="max-w-[200px]">
                                <span className="line-clamp-2">{row.title}</span>
                              </td>
                              <td style={{ color: '#a1a1aa' }}>{row.channel}</td>
                              <td className="font-medium">{formatViews(row.views)}</td>
                              <td><PerformanceBadge label={row.performance_label} /></td>
                              <td>
                                <div className="flex items-center gap-2 min-w-[100px]">
                                  <div className="progress-bar-track flex-1">
                                    <div
                                      className="progress-bar-fill"
                                      style={{ width: `${row.similarity_score * 100}%` }}
                                    />
                                  </div>
                                  <span className="text-xs" style={{ color: '#a1a1aa' }}>
                                    {(row.similarity_score * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </GlassCard>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-12">
          <GlassCard className="p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <p className="field-label mb-1">Need inspiration?</p>
              <p className="text-sm text-secondary">
                Find top-performing titles by keyword in {capitalize(niche)}.
              </p>
            </div>
            <Link to={`/improve?niche=${niche}`} className="btn-secondary px-6 py-2.5 text-sm shrink-0">
              Title Inspiration
            </Link>
          </GlassCard>
          <GlassCard className="p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <p className="field-label mb-1">Want more on {capitalize(niche)}?</p>
              <p className="text-sm text-secondary">
                View upload timing, duration, title patterns, and keyword charts.
              </p>
            </div>
            <Link to={`/insights?niche=${niche}`} className="btn-secondary px-6 py-2.5 text-sm shrink-0">
              Niche Insights
            </Link>
          </GlassCard>
          <GlassCard className="p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <p className="field-label mb-1">Analyze a full channel</p>
              <p className="text-sm text-secondary">
                Get personalized insights from any channel&apos;s video history.
              </p>
            </div>
            <Link to="/channel" className="btn-secondary px-6 py-2.5 text-sm shrink-0">
              Channel Analyzer
            </Link>
          </GlassCard>
        </div>
      </div>
    </PageLayout>
  );
}
