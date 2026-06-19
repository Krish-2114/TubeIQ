import { useState } from 'react';
import {
  NICHES,
  LABEL_RANK,
  predictTitle,
  capitalize,
  probToPercent,
} from '../utils/api';
import PageLayout from '../components/PageLayout';
import GlassCard from '../components/GlassCard';
import SectionHeader from '../components/SectionHeader';
import TitleInputPanel from '../components/TitleInputPanel';
import PredictionCertainty from '../components/PredictionCertainty';
import CrownIcon from '../components/CrownIcon';
import LoadingSpinner from '../components/LoadingSpinner';
import { ProbabilityChart } from '../components/Charts';

const defaultPanel = {
  title: '',
  niche: 'gaming',
  duration: 10,
  uploadDay: 5,
  uploadHour: 18,
};

export default function ABTest() {
  const [panelA, setPanelA] = useState({ ...defaultPanel });
  const [panelB, setPanelB] = useState({ ...defaultPanel, niche: 'tech' });
  const [resultA, setResultA] = useState(null);
  const [resultB, setResultB] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState('');

  const handleCompare = async () => {
    if (!panelA.title.trim() || !panelB.title.trim()) return;
    setComparing(true);
    setError('');
    try {
      const mkBody = (p) => ({
        title: p.title.trim(),
        niche: p.niche,
        duration_seconds: p.duration * 60,
        upload_day: p.uploadDay,
        upload_hour: p.uploadHour,
      });
      const [a, b] = await Promise.all([
        predictTitle(mkBody(panelA)),
        predictTitle(mkBody(panelB)),
      ]);
      setResultA(a);
      setResultB(b);
    } catch (e) {
      setError(e.message);
    } finally {
      setComparing(false);
    }
  };

  const getWinner = () => {
    if (!resultA || !resultB) return null;
    const rankA = LABEL_RANK[resultA.label] || 0;
    const rankB = LABEL_RANK[resultB.label] || 0;
    if (rankA > rankB) return 'A';
    if (rankB > rankA) return 'B';
    return probToPercent(resultA.confidence) >= probToPercent(resultB.confidence)
      ? 'A'
      : 'B';
  };

  const winner = getWinner();

  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <SectionHeader
          label="A/B TEST"
          title="Title A/B Tester"
          subtitle="Compare two title ideas side by side. The winner is the higher performance tier, or the higher class probability if tied."
        />

        <div className="flex flex-col lg:flex-row gap-5 mb-8">
          <TitleInputPanel
            label="Title A"
            borderColor="#ff0000"
            state={panelA}
            setState={setPanelA}
          />
          <TitleInputPanel
            label="Title B"
            borderColor="#a855f7"
            state={panelB}
            setState={setPanelB}
          />
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

        {(resultA || resultB) && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 fade-in">
            {[
              { label: 'A', result: resultA, color: '#ff0000', glow: 'winner-glow-red' },
              { label: 'B', result: resultB, color: '#a855f7', glow: 'winner-glow-purple' },
            ].map(({ label, result, color, glow }) =>
              result ? (
                <GlassCard
                  key={label}
                  className={`p-6 relative ${winner === label ? glow : ''}`}
                >
                  {winner === label && (
                    <div className="absolute top-5 right-5">
                      <CrownIcon color={color} />
                    </div>
                  )}
                  <p className="field-label mb-4">Title {label} Result</p>
                  <PredictionCertainty
                    label={result.label}
                    confidence={result.confidence}
                    size="md"
                  />
                  <div className="mt-6 pt-5 border-t" style={{ borderColor: '#f4f4f5' }}>
                    <p className="field-label mb-3">All class probabilities</p>
                    <ProbabilityChart probabilities={result.probabilities} height={170} />
                  </div>
                </GlassCard>
              ) : null
            )}
          </div>
        )}
      </div>
    </PageLayout>
  );
}
