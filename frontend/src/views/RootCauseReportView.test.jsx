import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import RootCauseReportView from './RootCauseReportView.jsx'

describe('RootCauseReportView', () => {
  it('shows the empty state when no report has been generated', () => {
    render(<RootCauseReportView result={null} />)
    expect(screen.getByText(/no report generated/i)).toBeTruthy()
  })

  it('shows the empty state when a result exists but has no report (healthy/control run)', () => {
    render(<RootCauseReportView result={{ report: null }} />)
    expect(screen.getByText(/no report generated/i)).toBeTruthy()
  })

  it('renders the diagnosis, root causes, and evidence table when a report is present', () => {
    const result = {
      report: {
        confidence: 'high',
        summary: 'raw_user_profiles drifted, causing predictions to shift.',
        detailed_explanation: 'account_age_days shifted sharply younger.',
        root_causes: ['raw_user_profiles'],
        suggested_fixes: [
          { action: 'add validation gate', target_urn: 'urn:li:dataset:(demo,raw_user_profiles,PROD)', rationale: 'prevent silent cohort shifts' },
        ],
        raw_trace: {
          candidates_examined: [
            {
              node_urn: 'urn:li:dataset:(demo,raw_user_profiles,PROD)',
              node_name: 'raw_user_profiles',
              hops_from_model: 2,
              drift_result: { method: 'ks_test', statistic: 0.42 },
              intervention_delta: 0.31,
              is_genuine_cause: true,
            },
          ],
        },
      },
    }
    render(<RootCauseReportView result={result} />)
    expect(screen.getAllByText('raw_user_profiles').length).toBeGreaterThan(0)
    expect(screen.getByText(/high confidence/i)).toBeTruthy()
    expect(screen.getByText('add validation gate')).toBeTruthy()
  })
})
