import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Cell } from 'recharts'
import './DriftTimelineView.css'

const SEVERITY_COLOR = {
  none: 'var(--signal-ok)',
  low: 'var(--signal-ok)',
  moderate: 'var(--signal-trace)',
  high: 'var(--signal-critical)',
  critical: 'var(--signal-critical)',
}

export default function DriftTimelineView({ result }) {
  if (!result) {
    return (
      <div className="empty-state mono">
        <p>No drift evidence yet.</p>
        <p className="empty-sub">Run the pipeline to see per-node statistical drift results.</p>
      </div>
    )
  }

  const { trace } = result
  const pred = trace.prediction_drift

  return (
    <div className="drift-view">
      <section className="drift-card">
        <header className="drift-card-header">
          <h3>Prediction output drift — {trace.model_urn.split(',')[1]}</h3>
          <span
            className="severity-badge mono"
            style={{ color: SEVERITY_COLOR[pred.severity], borderColor: SEVERITY_COLOR[pred.severity] }}
          >
            {pred.severity}
          </span>
        </header>
        <div className="drift-stats mono">
          <Stat label="method" value={pred.method} />
          <Stat label="KS statistic" value={pred.statistic.toFixed(4)} />
          <Stat label="p-value" value={pred.p_value < 0.0001 ? pred.p_value.toExponential(2) : pred.p_value.toFixed(4)} />
        </div>
      </section>

      <section className="drift-card">
        <header className="drift-card-header">
          <h3>Upstream candidates examined</h3>
          <span className="drift-card-sub mono">{trace.candidates_examined.length} node(s) tested</span>
        </header>

        <ResponsiveContainer width="100%" height={Math.max(220, trace.candidates_examined.length * 70)}>
          <BarChart
            layout="vertical"
            data={trace.candidates_examined.map((c) => ({
              name: c.node_name,
              'KS statistic': c.drift_result.statistic,
              'Intervention Δ': c.intervention_delta,
              genuine: c.is_genuine_cause,
            }))}
            margin={{ left: 24, right: 24 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#232b3a" horizontal={false} />
            <XAxis type="number" stroke="#7c8698" fontSize={11} domain={[0, 1]} />
            <YAxis type="category" dataKey="name" stroke="#7c8698" fontSize={11} width={170} />
            <Tooltip
              contentStyle={{ background: '#171d29', border: '1px solid #232b3a', fontSize: 12 }}
              labelStyle={{ color: '#e8ecf4' }}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: '#7c8698' }} />
            <Bar dataKey="KS statistic" fill="#7c8698" radius={[0, 3, 3, 0]} />
            <Bar dataKey="Intervention Δ" radius={[0, 3, 3, 0]}>
              {trace.candidates_examined.map((c, i) => (
                <Cell key={i} fill={c.is_genuine_cause ? '#ff5d5d' : '#ffb454'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <p className="drift-note mono">
          Intervention Δ = how much of the downstream (model output) drift disappears when this
          node's contribution is counterfactually held at its baseline behavior. High raw drift
          with low Δ means the node co-drifted but isn't the cause — see confounded_with in the report.
        </p>
      </section>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  )
}
