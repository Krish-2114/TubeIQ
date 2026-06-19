import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { NICHES, NICHE_GRADIENTS, fetchInsights, formatViews, capitalize } from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import PerformanceBadge from '../components/PerformanceBadge';
import { PreviewBarChart } from '../components/Charts';

const FEATURES = [
  {
    num: '01',
    title: 'Predict Performance',
    desc: 'Score any title as Low, Medium, High, or Viral relative to your niche.',
    link: '/analyze',
  },
  {
    num: '02',
    title: 'Analyze Any Channel',
    desc: 'Personalized insights from a channel\'s own history — upload timing, top videos, and keywords.',
    link: '/channel',
  },
  {
    num: '03',
    title: 'Niche Intelligence',
    desc: 'Charts and patterns for upload timing, duration, title structure, and keywords.',
    link: '/insights',
  },
  {
    num: '04',
    title: 'Discover Content Gaps',
    desc: 'Identify underexplored topics with detailed niche gap reports.',
    link: '/gaps',
  },
  {
    num: '05',
    title: 'Title Inspiration',
    desc: 'See top-performing videos by keyword and similar hits to inspire your next title.',
    link: '/improve',
  },
];

const EXAMPLE_CHANNELS = ['@mkbhd', '@MrBeast', '@CarryMinati', '@gordonramsay'];

function NicheSkeleton() {
  return (
    <GlassCard className="p-6">
      <div className="skeleton h-1 w-full mb-5" />
      <div className="skeleton h-5 w-28 mb-3" />
      <div className="skeleton h-4 w-20 mb-5" />
      <div className="skeleton h-9 w-24 mb-2" />
      <div className="skeleton h-3 w-36" />
    </GlassCard>
  );
}

function NicheCard({ niche, data, onClick }) {
  if (!data) return <NicheSkeleton />;

  return (
    <GlassCard className="p-6 card-hover" onClick={onClick}>
      <div className="niche-card-accent" style={{ background: NICHE_GRADIENTS[niche] }} />
      <h3 className="text-lg font-bold text-primary mb-3">{capitalize(niche)}</h3>
      <div className="text-3xl font-bold text-primary mt-1">
        {formatViews(data.niche_median_views)}
      </div>
      <div className="text-xs mt-1 uppercase tracking-wider text-muted">
        median views
      </div>
      <div
        className="text-sm mt-4 pt-4 border-t text-secondary"
        style={{ borderColor: '#f4f4f5' }}
      >
        Best day: {data.best_upload_day}
      </div>
    </GlassCard>
  );
}

export default function Landing() {
  const navigate = useNavigate();
  const [insights, setInsights] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    Promise.allSettled(NICHES.map((niche) => fetchInsights(niche).then((data) => [niche, data])))
      .then((results) => {
        const next = {};
        let failures = 0;
        results.forEach((result) => {
          if (result.status === 'fulfilled') {
            const [niche, data] = result.value;
            next[niche] = data;
          } else {
            failures += 1;
          }
        });
        setInsights(next);
        if (failures === NICHES.length) {
          setError('Could not load niche data. Make sure the backend is running on port 8000.');
        } else if (failures > 0) {
          setError('Some niches failed to load. Showing available data.');
        }
        setLoading(false);
      });
  }, []);

  return (
    <PageLayout mesh>
      <section className="max-w-7xl mx-auto px-6 pt-16 pb-24 lg:pt-24 min-h-[85vh] flex flex-col lg:flex-row items-center gap-16">
        <div className="flex-1 max-w-xl">
          <p className="label-badge mb-5" style={{ color: '#ff0000' }}>
            YOUTUBE INTELLIGENCE PLATFORM
          </p>
          <h1 className="section-heading mb-5" style={{ fontSize: 'clamp(42px, 6vw, 68px)' }}>
            Know What Works
            <br />
            <span className="gradient-text">Before You Upload.</span>
          </h1>
          <p className="text-lg leading-relaxed mb-10 text-secondary">
            Stop guessing. Analyze any YouTube channel, predict title performance,
            and find content gaps across 8 creator niches.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/analyze" className="btn-primary px-7 py-3.5 text-sm">
              Analyze a Title
            </Link>
            <Link to="/insights" className="btn-secondary px-7 py-3.5 text-sm">
              Niche Insights
            </Link>
            <Link to="/channel" className="btn-secondary px-7 py-3.5 text-sm">
              Analyze Channel
            </Link>
            <Link to="/ab-test" className="btn-secondary px-7 py-3.5 text-sm">
              A/B Test
            </Link>
          </div>
        </div>

        <div className="flex-1 flex justify-center w-full">
          <GlassCard className="preview-glow preview-tilt p-7 w-full max-w-md" glow>
            <div className="flex items-center justify-between mb-6">
              <span className="label-badge text-muted">LIVE PREVIEW</span>
              <span
                className="text-xs px-2 py-1 rounded font-medium"
                style={{ background: '#fff5f5', color: '#ff0000' }}
              >
                Gaming
              </span>
            </div>

            <div className="mb-6 p-4 preview-panel">
              <p className="label-badge mb-3 text-secondary">Prediction</p>
              <div className="flex items-center gap-5">
                <PerformanceBadge label="High" size="lg" />
                <div>
                  <div className="text-2xl font-bold text-primary">78.4%</div>
                  <div className="text-xs text-muted">probability on &quot;High&quot;</div>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <p className="label-badge mb-2 text-secondary">Probabilities</p>
              <PreviewBarChart />
            </div>

            <div>
              <p className="label-badge mb-3 text-secondary">Similar Titles</p>
              {[
                { title: 'Horror Game Marathon Part 3', meta: 'ExampleChannel · 10M views' },
                { title: 'GTA Gameplay Walkthrough', meta: 'SampleGamer · 8.2M views' },
                { title: 'IRL Stream Highlights', meta: 'DemoCreator · 5.1M views' },
              ].map((row) => (
                <div
                  key={row.title}
                  className="py-3 border-b last:border-0"
                  style={{ borderColor: '#f4f4f5' }}
                >
                  <div className="text-sm text-primary truncate">{row.title}</div>
                  <div className="text-xs mt-1 text-muted">{row.meta}</div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {FEATURES.map((f) => (
            <Link key={f.num} to={f.link}>
              <GlassCard className="p-6 card-hover h-full">
                <div className="feature-icon mb-4">{f.num}</div>
                <h3 className="font-bold text-primary text-lg mb-2">{f.title}</h3>
                <p className="text-sm leading-relaxed text-secondary">{f.desc}</p>
              </GlassCard>
            </Link>
          ))}
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-20">
        <GlassCard className="p-8 lg:p-10" glow>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div>
              <p className="label-badge mb-3" style={{ color: '#ff0000' }}>CHANNEL ANALYZER</p>
              <h2 className="section-heading text-2xl md:text-3xl mb-4">
                Insights From Any Channel&apos;s History
              </h2>
              <p className="text-secondary leading-relaxed mb-6">
                Enter any YouTube handle and get median views, best upload day,
                optimal duration, top keywords, and your best-performing videos —
                all based on that channel&apos;s own uploads.
              </p>
              <div className="flex flex-wrap gap-2 mb-6">
                {EXAMPLE_CHANNELS.map((handle) => (
                  <Link
                    key={handle}
                    to={`/channel?q=${encodeURIComponent(handle)}`}
                    className="keyword-pill text-xs hover:border-red-300"
                  >
                    {handle}
                  </Link>
                ))}
              </div>
              <Link to="/channel" className="btn-primary px-7 py-3 text-sm inline-block">
                Analyze a Channel
              </Link>
            </div>
            <div className="preview-panel p-5 space-y-4">
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-full shrink-0"
                  style={{ background: 'linear-gradient(135deg, #ff0000, #990000)' }}
                />
                <div>
                  <div className="font-semibold text-primary">Marques Brownlee</div>
                  <div className="text-xs text-muted">21M subs · 200 videos analyzed</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Median views', value: '4.3M' },
                  { label: 'Best day', value: 'Wednesday' },
                  { label: 'Engagement', value: '3.27%' },
                  { label: 'Viral hits', value: '7' },
                ].map((s) => (
                  <div key={s.label} className="p-3 rounded-lg bg-white border" style={{ borderColor: '#e4e4e7' }}>
                    <div className="text-xs text-muted">{s.label}</div>
                    <div className="font-bold text-primary mt-1">{s.value}</div>
                  </div>
                ))}
              </div>
              <div>
                <div className="text-xs text-muted mb-2">Top keywords in best videos</div>
                <div className="flex flex-wrap gap-1.5">
                  {['iphone', 'review', 'samsung', 'apple'].map((w) => (
                    <span key={w} className="keyword-pill text-xs">{w}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-28">
        <SectionHeader
          label="EXPLORE"
          title="Choose Your Niche"
          subtitle="Select a niche to explore detailed charts and performance patterns."
          align="center"
        />
        {error && <p className="error-msg text-center max-w-md mx-auto">{error}</p>}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {loading
            ? NICHES.map((n) => <NicheSkeleton key={n} />)
            : NICHES.map((niche) => (
                <NicheCard
                  key={niche}
                  niche={niche}
                  data={insights[niche]}
                  onClick={() => navigate(`/insights?niche=${niche}`)}
                />
              ))}
        </div>
      </section>
    </PageLayout>
  );
}
