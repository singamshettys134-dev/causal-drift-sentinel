import { describe, expect, it } from 'vitest'
import { nodeStatus } from './LineageGraphView.jsx'

const URN_ROOT_CAUSE = 'urn:li:dataset:(demo,raw_user_profiles,PROD)'
const URN_CONFOUNDED = 'urn:li:mlFeatureTable:(demo,feature_user_risk_score)'
const URN_ON_PATH = 'urn:li:mlModel:(demo,fraud_model_v3,PROD)'
const URN_IDLE = 'urn:li:dataset:(demo,raw_transactions,PROD)'

const fakeResult = {
  trace: {
    isolated_root_causes: [{ node_urn: URN_ROOT_CAUSE }],
    graph_path: [URN_ROOT_CAUSE, URN_ON_PATH],
    candidates_examined: [
      { node_urn: URN_ROOT_CAUSE, is_genuine_cause: true },
      { node_urn: URN_CONFOUNDED, is_genuine_cause: false },
    ],
  },
}

describe('nodeStatus', () => {
  it('returns idle when no result is present yet', () => {
    expect(nodeStatus(URN_ROOT_CAUSE, null)).toBe('idle')
  })

  it('marks the isolated root cause node as root-cause', () => {
    expect(nodeStatus(URN_ROOT_CAUSE, fakeResult)).toBe('root-cause')
  })

  it('marks a candidate that drifted but was ruled out as confounded', () => {
    expect(nodeStatus(URN_CONFOUNDED, fakeResult)).toBe('confounded')
  })

  it('marks a node on the causal graph path (but not the root cause itself) as on-path', () => {
    expect(nodeStatus(URN_ON_PATH, fakeResult)).toBe('on-path')
  })

  it('marks an unrelated node as idle even with a result present', () => {
    expect(nodeStatus(URN_IDLE, fakeResult)).toBe('idle')
  })
})
