import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  NICHES,
  compareTitles,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import EmptyState from '../components/EmptyState';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Analyzer() {
  const [searchParams] = useSearchParams();
  const [niche, setNiche] = useState(searchParams.get('niche') || 'gaming');
  const [title, setTitle] = useState('');
  const [titleB, setTitleB] = useState('');

  const [comparison, setComparison] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState('');

  useEffect(() => {
    const param = searchParams.get('niche');
    if (param && NICHES.includes(param)) setNiche(param);
  }, [searchParams]);

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

  const confidencePct = (confidence) =>
    Number.isFinite(confidence)
      ? `${Math.round(confidence * 100)}%`
      : '—';

  const winnerLabel =
    comparison?.winner === 'title_a'
      ? 'Title A'
      : comparison?.winner === 'title_b'
        ? 'Title B'
        : null;

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="ANALYZER"
          title="Title Comparison"
          subtitle="Compare two title ideas with the blended ranking model for your niche — TF-IDF similarity plus structural title signals."
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
                  placeholder="Enter the first title..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
                />
              </div>

              <div className="mb-6">
                <label className="field-label">Title B</label>
                <input
                  className="input-field"
                  placeholder="Enter the second title..."
                  value={titleB}
                  onChange={(e) => setTitleB(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
                />
              </div>

              <button
                className="btn-primary w-full py-3.5 flex items-center justify-center gap-2 text-sm"
                onClick={handleCompare}
                disabled={comparing || !title.trim() || !titleB.trim()}
              >
                {comparing && <LoadingSpinner />}
                {comparing ? 'Comparing...' : 'Compare Titles'}
              </button>
              {compareError && <p className="error-msg">{compareError}</p>}
            </GlassCard>
          </div>

          {/* Results */}
          <div className="xl:col-span-3 space-y-6">
            {!comparison && (
              <EmptyState
                title="No comparison yet"
                description="Enter two titles and click Compare to see which the blend model ranks higher for this niche."
              />
            )}

            {comparison && (
              <GlassCard className="p-6 fade-in" glow>
                <h3 className="section-heading text-lg mb-2">Title Comparison</h3>
                <p className="text-sm text-secondary mb-6">
                  {winnerLabel} wins — {confidencePct(comparison.confidence)} confidence
                </p>
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
                  Winner is chosen by the niche blend model (TF-IDF reference
                  similarity plus structural title features), not raw similarity
                  alone. Similarity scores above are still shown for context.
                </p>
              </GlassCard>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-12">
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
