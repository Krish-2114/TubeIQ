import GlassCard from './GlassCard';
import { NICHES, DAYS, HOURS, capitalize } from '../utils/api';

export default function TitleInputPanel({ label, borderColor, state, setState }) {
  return (
    <GlassCard className="p-6 flex-1" accent={borderColor}>
      <h3 className="section-heading text-lg mb-5">{label}</h3>
      <div className="space-y-4">
        <div>
          <label className="field-label">Title</label>
          <input
            className="input-field"
            placeholder="Enter title..."
            value={state.title}
            onChange={(e) => setState({ ...state, title: e.target.value })}
          />
        </div>
        <div>
          <label className="field-label">Niche</label>
          <select
            className="select-field"
            value={state.niche}
            onChange={(e) => setState({ ...state, niche: e.target.value })}
          >
            {NICHES.map((n) => (
              <option key={n} value={n}>{capitalize(n)}</option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="field-label">Min</label>
            <input
              type="number"
              className="input-field"
              min={1}
              value={state.duration}
              onChange={(e) => setState({ ...state, duration: Number(e.target.value) })}
            />
          </div>
          <div>
            <label className="field-label">Day</label>
            <select
              className="select-field"
              value={state.uploadDay}
              onChange={(e) => setState({ ...state, uploadDay: Number(e.target.value) })}
            >
              {DAYS.map((d, i) => (
                <option key={d} value={i}>{d.slice(0, 3)}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="field-label">Hour</label>
            <select
              className="select-field"
              value={state.uploadHour}
              onChange={(e) => setState({ ...state, uploadHour: Number(e.target.value) })}
            >
              {HOURS.map((h) => (
                <option key={h} value={h}>
                  {h <= 12 ? `${h}A` : `${h - 12}P`}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
