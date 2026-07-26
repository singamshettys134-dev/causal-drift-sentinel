import React from 'react'
import './TopBar.css'

const STAGE_LABELS = {
  idle: 'Idle',
  ingesting: 'Ingesting lineage',
  detecting: 'Detecting drift',
  tracing: 'Isolating root cause',
  reasoning: 'Generating diagnosis',
  writing_back: 'Writing back',
  done: 'Complete',
}

export default function TopBar({ stage, onReplay }) {
  const running = stage !== 'idle' && stage !== 'done'

  return (
    <header className="top-bar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">◆</span>
        <div>
          <h1>Causal Drift Sentinel</h1>
          <p className="brand-sub mono">watching fraud_model_v3 · production</p>
        </div>
      </div>

      <div className="pipeline-status mono" aria-live="polite">
        <span className={`status-dot ${running ? 'status-dot-live' : ''} ${stage === 'done' ? 'status-dot-done' : ''}`} />
        {STAGE_LABELS[stage]}
      </div>

      <div className="replay-controls">
        <button
          className="btn btn-ghost"
          disabled={running}
          onClick={() => onReplay(false)}
          title="Run the pipeline against a healthy pipeline — no drift injected"
        >
          Run control (no drift)
        </button>
        <button
          className="btn btn-primary"
          disabled={running}
          onClick={() => onReplay(true)}
        >
          {running ? 'Investigating…' : '▶ Replay a failure'}
        </button>
      </div>
    </header>
  )
}
