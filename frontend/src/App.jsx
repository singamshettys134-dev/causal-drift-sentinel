import React, { Suspense, lazy, useCallback, useState } from 'react'
import { investigate } from './api/client'
import TopBar from './components/TopBar.jsx'
import StatusFooter from './components/StatusFooter.jsx'
import './App.css'

// LineageGraphView (reactflow) and DriftTimelineView (recharts) pull in the
// two heaviest dependencies in the bundle. Lazy-loading them means the
// initial page load only ships TopBar/StatusFooter/RootCauseReportView —
// the graph/chart libraries load on first visit to those tabs instead of
// blocking first paint.
const LineageGraphView = lazy(() => import('./views/LineageGraphView.jsx'))
const DriftTimelineView = lazy(() => import('./views/DriftTimelineView.jsx'))
const RootCauseReportView = lazy(() => import('./views/RootCauseReportView.jsx'))

const MODEL_URN = 'urn:li:mlModel:(demo,fraud_model_v3,PROD)'

const STAGES = ['idle', 'ingesting', 'detecting', 'tracing', 'reasoning', 'writing_back', 'done']

export default function App() {
  const [result, setResult] = useState(null)
  const [stage, setStage] = useState('idle')
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('graph')

  const runInvestigation = useCallback(async (injectDrift) => {
    setError(null)
    setResult(null)
    setStage('ingesting')
    try {
      // Stage choreography purely for the demo's legibility — the backend
      // runs the real pipeline in one call; we narrate its known stages
      // while the request is in flight per spec Section 4's "Replay a
      // failure" mode.
      const stageTimer = setTimeout(() => setStage('detecting'), 400)
      const stageTimer2 = setTimeout(() => setStage('tracing'), 900)
      const stageTimer3 = setTimeout(() => setStage('reasoning'), 1500)

      const data = await investigate({ modelUrn: MODEL_URN, injectDrift })

      clearTimeout(stageTimer); clearTimeout(stageTimer2); clearTimeout(stageTimer3)
      setStage('writing_back')
      await new Promise((r) => setTimeout(r, 350))
      setResult(data)
      setStage('done')
      setActiveTab(data.report ? 'report' : 'graph')
    } catch (e) {
      setError(e.message || String(e))
      setStage('idle')
    }
  }, [])

  return (
    <div className="app-shell">
      <TopBar stage={stage} onReplay={runInvestigation} />

      {error && (
        <div className="error-banner mono">
          <strong>Pipeline failed.</strong> {error}
        </div>
      )}

      <div className="tab-bar mono">
        {[
          { id: 'graph', label: '01 · Lineage Graph' },
          { id: 'timeline', label: '02 · Drift Timeline' },
          { id: 'report', label: '03 · Root-Cause Report', disabled: !result?.report },
        ].map((t) => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'tab-active' : ''}`}
            disabled={t.disabled}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <main className="main-panel">
        <Suspense fallback={<div className="empty-state mono">Loading view…</div>}>
          {activeTab === 'graph' && <LineageGraphView result={result} stage={stage} />}
          {activeTab === 'timeline' && <DriftTimelineView result={result} />}
          {activeTab === 'report' && <RootCauseReportView result={result} />}
        </Suspense>
      </main>

      <StatusFooter result={result} stage={stage} />
    </div>
  )
}
