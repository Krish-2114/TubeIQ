import { useState } from 'react';
import {
  NICHES,
  compareTitles,
  capitalize,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import CrownIcon from '../components/CrownIcon';
import LoadingSpinner from '../components/LoadingSpinner';

const defaultPanel = {
  title: '',
  niche: 'gaming',
  duration_seconds: 600,
  day_of_week: 2,
  hour_of_day: 14,
};

export default function ABTest() {
  const [panelA, setPanelA] = useState({ ...defaultPanel });
  const [panelB, setPanelB] = useState({ ...defaultPanel });
  const [result, setResult] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState('');

  const handleCompare = async () => {
    if (!panelA.title.trim() || !panelB.title.trim()) return;
    if (panelA.niche !== panelB.niche) {
      setError('Both titles must use the same niche for a fair comparison.');
      return;
    }
    setComparing(true);
    setError('');
    setResult(null);
    try {
      const comparison = await compareTitles({
        niche: panelA.niche,
        title_a: panelA.title.trim(),
        title_b: panelB.title.trim(),
        duration_seconds_a: Number(panelA.duration_seconds) || 600,
        duration_seconds_b: Number(panelB.duration_seconds) || 600,
        day_of_week_a: Number(panelA.day_of_week),
        day_of_week_b: Number(panelB.day_of_week),
        hour_of_day_a: Number(panelA.hour_of_day),
        hour_of_day_b: Number(panelB.hour_of_day),
      });
      setResult(comparison);
    } catch (e) {
      setError(e.message);
    } finally {
      setComparing(false);
    }
  };

  const scorePct = (score) =>
    Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : '—';

  const winner = result?.winner === 'title_a' ? 'A' : result?.winner === 'title_b' ? 'B' : null;

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="A/B TEST"
          title="Title A/B Tester"
          subtitle="Compare two title ideas in the same niche with the Random Forest blend model (titles + planned duration/upload time)."
        />

        <div className="mb-6 max-w-xs">
          <label className="field-label">Niche</label>
          <select
            className="select-field"
            value={panelA.niche}
            onChange={(e) => {
              const niche = e.target.value;
              setPanelA({ ...panelA, niche });
              setPanelB({ ...panelB, niche });
            }}
          >
            {NICHES.map((n) => (
              <option key={n} value={n}>{capitalize(n)}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col lg:flex-row gap-5 mb-8">
          {[
            { key: 'A', panel: panelA, setPanel: setPanelA, accent: '#ff0000' },
            { key: 'B', panel: panelB, setPanel: setPanelB, accent: '#a855f7' },
          ].map(({ key, panel, setPanel, accent }) => (
            <GlassCard key={key} className="p-6 flex-1" accent={accent}>
              <h3 className="section-heading text-lg mb-5">Title {key}</h3>
              <label className="field-label">Title</label>
              <input
                className="input-field mb-4"
                placeholder="Enter title..."
                value={panel.title}
                onChange={(e) => setPanel({ ...panel, title: e.target.value })}
              />
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="field-label">Duration (sec)</label>
                  <input
                    className="input-field"
                    type="number"
                    min={30}
                    value={panel.duration_seconds}
                    onChange={(e) =>
                      setPanel({
                        ...panel,
                        duration_seconds: e.target.value,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="field-label">Day (0–6)</label>
                  <input
                    className="input-field"
                    type="number"
                    min={0}
                    max={6}
                    value={panel.day_of_week}
                    onChange={(e) =>
                      setPanel({ ...panel, day_of_week: e.target.value })
                    }
                  />
                </div>
                <div>
                  <label className="field-label">Hour (0–23)</label>
                  <input
                    className="input-field"
                    type="number"
                    min={0}
                    max={23}
                    value={panel.hour_of_day}
                    onChange={(e) =>
                      setPanel({ ...panel, hour_of_day: e.target.value })
                    }
                  />
                </div>
              </div>
            </GlassCard>
          ))}
        </div>

        <div className="flex justify-center mb-8">
          <button
            className="btn-primary px-10 py-3.5 flex items-center gap-2 text-sm"
            onClick={handleCompare}
            disabled={comparing || !panelA.title.trim() || !panelB.title.trim()}
          >
            {comparing && <LoadingSpinner />}
            Compare Titles
          </button>
        </div>

        {error && <p className="error-msg text-center max-w-md mx-auto">{error}</p>}

        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 fade-in">
            {[
              {
                label: 'A',
                title: panelA.title,
                score: result.title_a_score,
                color: '#ff0000',
                glow: 'winner-glow-red',
              },
              {
                label: 'B',
                title: panelB.title,
                score: result.title_b_score,
                color: '#a855f7',
                glow: 'winner-glow-purple',
              },
            ].map(({ label, title, score, color, glow }) => (
              <GlassCard
                key={label}
                className={`p-6 relative ${winner === label ? glow : ''}`}
              >
                {winner === label && (
                  <div className="absolute top-5 right-5">
                    <CrownIcon color={color} />
                  </div>
                )}
                <p className="field-label mb-2">Title {label} Result</p>
                <p className="text-sm mb-4 line-clamp-3">{title}</p>
                <p className="text-3xl font-semibold" style={{ color: '#18181b' }}>
                  {scorePct(score)}
                </p>
                <p className="text-xs mt-2" style={{ color: '#a1a1aa' }}>
                  Mean similarity to High/Viral reference titles
                </p>
              </GlassCard>
            ))}
          </div>
        )}
      </div>
    </PageLayout>
  );
}
