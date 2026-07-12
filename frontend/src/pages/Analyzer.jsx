import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  NICHES,
  predictTitle,
  compareTitles,
  fetchSimilar,
  formatViews,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import EmptyState from '../components/EmptyState';
import PerformanceBadge from '../components/PerformanceBadge';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Analyzer() {
  const [searchParams] = useSearchParams();
  const [niche, setNiche] = useState(searchParams.get('niche') || 'gaming');
  const [title, setTitle] = useState('');
  const [titleB, setTitleB] = useState('');

  const [prediction, setPrediction] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [similar, setSimilar] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState('');
  const [compareError, setCompareError] = useState('');
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

  const handleCompare = async () => {
    if (!title.trim() || !titleB.trim()) return;
    setComparing(true);
    setCompareError('');
    setComparison(null);
    try {
      const result = await compareTitles({
        niche,
        title_a: title.trim(),
        title_b: titleB.trim(),
      });
      setComparison(result);
    } catch (e) {
      setCompareError(e.message);
    } finally {
      setComparing(false);
    }
  };

  const scorePct = (score) =>
    Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : '—';

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="ANALYZER"
          title="Title Performance Analyzer"
          subtitle="Score a title against proven High/Viral winners in your niche, or compare two title ideas side by side."
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
                <label className="field-label">Title A</label>
                <input
                  className="input-field"
                  placeholder="Enter your video title idea..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                />
              </div>

              <div className="mb-6">
                <label className="field-label">Title B (optional — for compare)</label>
                <input
                  className="input-field"
                  placeholder="Enter a second title to compare..."
                  value={titleB}
                  onChange={(e) => setTitleB(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-1 gap-3">
                <button
                  className="btn-primary w-full py-3.5 flex items-center justify-center gap-2 text-sm"
                  onClick={handleAnalyze}
                  disabled={analyzing || !title.trim()}
                >
                  {analyzing && <LoadingSpinner />}
                  {analyzing ? 'Analyzing...' : 'Analyze Title A'}
                </button>
                <button
                  className="btn-secondary w-full py-3.5 flex items-center justify-center gap-2 text-sm"
                  onClick={handleCompare}
                  disabled={comparing || !title.trim() || !titleB.trim()}
                >
                  {comparing && <LoadingSpinner />}
                  {comparing ? 'Comparing...' : 'Compare Titles'}
                </button>
              </div>
              {analyzeError && <p className="error-msg">{analyzeError}</p>}
              {compareError && <p className="error-msg">{compareError}</p>}
            </GlassCard>
          </div>

          {/* Results */}
          <div className="xl:col-span-3 space-y-6">
            {!showResults && !comparison && (
              <EmptyState
                title="No analysis yet"
                description="Enter a title and click Analyze for a niche percentile score, or add a second title and Compare to see which is closer to proven winners."
              />
            )}

            {comparison && (
              <GlassCard className="p-6 fade-in" glow>
                <h3 className="section-heading text-lg mb-6">Title Comparison</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                  <div
                    className="rounded-xl p-4 border"
                    style={{
                      borderColor:
                        comparison.winner === 'title_a' ? '#22c55e' : '#f4f4f5',
                      background:
                        comparison.winner === 'title_a'
                          ? 'rgba(34,197,94,0.06)'
                          : 'transparent',
                    }}
                  >
                    <p className="field-label mb-2">
                      Title A{comparison.winner === 'title_a' ? ' — Winner' : ''}
                    </p>
                    <p className="text-sm mb-3 line-clamp-3">{title.trim()}</p>
                    <p className="text-2xl font-semibold" style={{ color: '#18181b' }}>
                      {scorePct(comparison.title_a_score)}
                    </p>
                    <p className="text-xs mt-1" style={{ color: '#a1a1aa' }}>
                      Similarity to High/Viral titles
                    </p>
                  </div>
                  <div
                    className="rounded-xl p-4 border"
                    style={{
                      borderColor:
                        comparison.winner === 'title_b' ? '#22c55e' : '#f4f4f5',
                      background:
                        comparison.winner === 'title_b'
                          ? 'rgba(34,197,94,0.06)'
                          : 'transparent',
                    }}
                  >
                    <p className="field-label mb-2">
                      Title B{comparison.winner === 'title_b' ? ' — Winner' : ''}
                    </p>
                    <p className="text-sm mb-3 line-clamp-3">{titleB.trim()}</p>
                    <p className="text-2xl font-semibold" style={{ color: '#18181b' }}>
                      {scorePct(comparison.title_b_score)}
                    </p>
                    <p className="text-xs mt-1" style={{ color: '#a1a1aa' }}>
                      Similarity to High/Viral titles
                    </p>
                  </div>
                </div>
                <p className="text-sm text-secondary">
                  Winner is the title with higher mean cosine similarity to this
                  niche&apos;s proven High/Viral reference set.
                </p>
              </GlassCard>
            )}

            {showResults && prediction && (
              <div className="fade-in space-y-6">
                <GlassCard className="p-6" glow>
                  <h3 className="section-heading text-lg mb-6">Title Score</h3>
                  <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
                    <div>
                      <p className="field-label mb-2">Niche percentile</p>
                      <p className="text-4xl font-semibold" style={{ color: '#18181b' }}>
                        {prediction.percentile.toFixed(0)}
                        <span className="text-xl font-medium" style={{ color: '#a1a1aa' }}>
                          th
                        </span>
                      </p>
                      <p className="text-sm mt-3 text-secondary max-w-md">
                        This title scores higher than{' '}
                        <strong>{prediction.percentile.toFixed(0)}%</strong> of
                        videos in {capitalize(niche)}.
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="field-label mb-2">Similarity score</p>
                      <p className="text-2xl font-semibold">
                        {scorePct(prediction.score)}
                      </p>
                    </div>
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
