import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  NICHES,
  fetchGaps,
  formatViews,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import LoadingSpinner from '../components/LoadingSpinner';

export default function ContentGaps() {
  const [searchParams] = useSearchParams();
  const [niche, setNiche] = useState(searchParams.get('niche') || 'gaming');
  const [gaps, setGaps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const param = searchParams.get('niche');
    if (param && NICHES.includes(param)) setNiche(param);
  }, [searchParams]);

  useEffect(() => {
    setLoading(true);
    setError('');
    fetchGaps(niche)
      .then((data) => setGaps(data.gaps || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [niche]);

  const highOpportunity = gaps.filter((g) => g.opportunity_score > 0.7).length;

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="CONTENT GAPS"
          title="Content Gap Report"
          subtitle="Underperforming topic clusters in your niche — areas that exist but have room to grow."
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

        {!loading && !error && (
          <div className="fade-in">
            <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
              <div>
                <h2 className="section-heading text-xl">{capitalize(niche)} Gap Clusters</h2>
                <p className="text-sm text-secondary mt-1">
                  Topics with below-average views compared to other clusters in this niche.
                </p>
              </div>
              <div className="text-sm text-secondary">
                {gaps.length} gaps found
                {highOpportunity > 0 && (
                  <span className="text-primary font-medium">
                    {' '}· {highOpportunity} high opportunity
                  </span>
                )}
              </div>
            </div>

            {gaps.length === 0 ? (
              <GlassCard className="p-8 text-center text-secondary">
                No content gaps detected for this niche.
              </GlassCard>
            ) : (
              <div className="space-y-4">
                {gaps.map((gap, index) => (
                  <GlassCard key={gap.cluster_id} className="p-6 card-hover">
                    <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <span className="section-number">#{index + 1}</span>
                          <span className="font-semibold text-primary">
                            Cluster {gap.cluster_id}
                          </span>
                          {gap.opportunity_score > 0.7 && (
                            <span
                              className="text-xs px-2 py-1 rounded font-medium"
                              style={{ background: '#fff5f5', color: '#ff0000' }}
                            >
                              High opportunity
                            </span>
                          )}
                        </div>
                        <p className="field-label mb-2">Topic keywords</p>
                        <div className="flex flex-wrap gap-2">
                          {(gap.keywords || []).map((kw) => (
                            <span key={kw} className="keyword-pill text-xs">{kw}</span>
                          ))}
                        </div>
                      </div>

                      <div className="lg:w-56 shrink-0 space-y-4">
                        <div>
                          <p className="field-label">Avg Views</p>
                          <p className="text-2xl font-bold text-primary">
                            {formatViews(gap.avg_views)}
                          </p>
                        </div>
                        <div>
                          <div className="flex justify-between text-xs text-secondary mb-2">
                            <span>Opportunity score</span>
                            <span>{(gap.opportunity_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="opportunity-bar">
                            <div
                              className="opportunity-fill"
                              style={{
                                width: `${Math.min(gap.opportunity_score * 100, 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                        <p className="text-xs text-muted leading-relaxed">
                          Lower avg views vs other clusters in {capitalize(niche)} —
                          potential room to create standout content on these topics.
                        </p>
                      </div>
                    </div>
                  </GlassCard>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </PageLayout>
  );
}
