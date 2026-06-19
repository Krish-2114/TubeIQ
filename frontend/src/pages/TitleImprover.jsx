import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  NICHES,
  improveTitle,
  formatViews,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import PerformanceBadge from '../components/PerformanceBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';

export default function TitleImprover() {
  const [searchParams] = useSearchParams();
  const [title, setTitle] = useState(searchParams.get('title') || '');
  const [niche, setNiche] = useState(searchParams.get('niche') || 'tech');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const paramNiche = searchParams.get('niche');
    if (paramNiche && NICHES.includes(paramNiche)) setNiche(paramNiche);
    const paramTitle = searchParams.get('title');
    if (paramTitle) setTitle(paramTitle);
  }, [searchParams]);

  const handleSearch = async () => {
    if (!title.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await improveTitle(title, niche);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const copyTitle = (text) => {
    navigator.clipboard?.writeText(text);
  };

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="TITLE INSPIRATION"
          title="Find What Works"
          subtitle="Enter your title idea to see top-performing videos in your niche — grouped by keyword and similar hits for inspiration."
        />

        <GlassCard className="p-6 mb-8" accent="#ff0000">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="field-label">Niche</label>
              <select
                className="select-field"
                value={niche}
                onChange={(e) => setNiche(e.target.value)}
              >
                {NICHES.map((n) => (
                  <option key={n} value={n}>{capitalize(n)}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="field-label">Your title idea</label>
              <input
                className="input-field"
                placeholder="e.g. Laptop Review"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>
          </div>
          <button
            className="btn-primary px-8 py-3 text-sm flex items-center gap-2"
            onClick={handleSearch}
            disabled={loading || !title.trim()}
          >
            {loading && <LoadingSpinner />}
            {loading ? 'Searching...' : 'Find Inspiration'}
          </button>
          {error && <p className="error-msg mt-4">{error}</p>}
        </GlassCard>

        {!loading && !result && !error && (
          <EmptyState
            title="No results yet"
            description="Enter a title and pick a niche to see top-performing videos that match your keywords."
          />
        )}

        {result && (
          <div className="space-y-8 fade-in">
            <GlassCard className="p-6">
              <p className="field-label mb-2">Your title</p>
              <p className="text-lg text-primary font-medium">{result.original_title}</p>
              <p className="text-sm text-secondary mt-2">
                Niche: {capitalize(result.niche)}
              </p>
              {result.matched_keywords?.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {result.matched_keywords.map((kw) => (
                    <span key={kw} className="keyword-pill">{kw}</span>
                  ))}
                </div>
              )}
            </GlassCard>

            {result.top_titles_by_keyword?.length > 0 ? (
              <div className="space-y-6">
                <h2 className="section-heading text-xl">Top Titles by Keyword</h2>
                {result.top_titles_by_keyword.map(({ keyword, titles }) => (
                  <GlassCard key={keyword} className="p-6">
                    <p className="field-label mb-4">
                      Top performers with &quot;{keyword}&quot;
                    </p>
                    <div className="space-y-3">
                      {titles.map((row, i) => (
                        <div
                          key={i}
                          className="flex flex-col sm:flex-row sm:items-center gap-3 p-4 rounded-xl border"
                          style={{ borderColor: '#f4f4f5' }}
                        >
                          <span className="section-number shrink-0">#{i + 1}</span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-primary leading-snug">
                              {row.title}
                            </p>
                            <p className="text-xs text-muted mt-1">
                              {formatViews(row.views)} views
                            </p>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <PerformanceBadge label={row.label} />
                            <button
                              type="button"
                              className="btn-secondary px-3 py-1.5 text-xs"
                              onClick={() => copyTitle(row.title)}
                            >
                              Copy
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                ))}
              </div>
            ) : (
              <GlassCard className="p-8 text-center text-secondary">
                No top titles found for keywords in this title.
              </GlassCard>
            )}

            {result.top_niche_keywords?.length > 0 && (
              <GlassCard className="p-6">
                <p className="field-label mb-2">Popular Keywords in {capitalize(result.niche)}</p>
                <p className="text-sm text-secondary mb-4">
                  Words that appear most in high-performing titles in this niche.
                </p>
                <div className="flex flex-wrap gap-2">
                  {result.top_niche_keywords.map((kw) => (
                    <span key={kw} className="keyword-pill">{kw}</span>
                  ))}
                </div>
              </GlassCard>
            )}

            {result.inspiration_titles?.length > 0 && (
              <GlassCard className="p-6">
                <p className="field-label mb-2">Similar Videos for Inspiration</p>
                <p className="text-sm text-secondary mb-6">
                  High-performing titles most similar to yours in {capitalize(result.niche)}.
                </p>
                <div className="overflow-x-auto -mx-2">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Title</th>
                        <th>Views</th>
                        <th>Label</th>
                        <th>Match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.inspiration_titles.map((row, i) => (
                        <tr key={i}>
                          <td className="max-w-[320px]">
                            <span className="line-clamp-2">{row.title}</span>
                          </td>
                          <td className="font-medium">{formatViews(row.views)}</td>
                          <td><PerformanceBadge label={row.label} /></td>
                          <td>
                            {row.similarity_score != null && (
                              <span className="text-xs text-secondary">
                                {(row.similarity_score * 100).toFixed(0)}%
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            )}

            <GlassCard className="p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <p className="field-label mb-1">Ready to test a title?</p>
                <p className="text-sm text-secondary">
                  Score any title in the analyzer to predict performance.
                </p>
              </div>
              <Link
                to={`/analyze?niche=${result.niche}`}
                className="btn-secondary px-6 py-2.5 text-sm shrink-0"
              >
                Open Analyzer
              </Link>
            </GlassCard>
          </div>
        )}
      </div>
    </PageLayout>
  );
}
