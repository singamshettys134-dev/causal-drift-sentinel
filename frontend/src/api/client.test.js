import { afterEach, describe, expect, it, vi } from 'vitest'
import { getLineage, investigate } from './client.js'

describe('api client', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('investigate() posts to /api/investigate with the given params', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ graph: {}, trace: {}, report: null, writeback: null }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await investigate({ modelUrn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)', injectDrift: true })

    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toContain('/api/investigate')
    expect(url).toContain('inject_drift=true')
    expect(options.method).toBe('POST')
  })

  it('investigate() throws with response detail when the request fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => 'Pipeline execution failed.',
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(investigate({ modelUrn: 'x', injectDrift: false })).rejects.toThrow(/500/)
  })

  it('getLineage() URL-encodes the model URN', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)

    await getLineage('urn:li:mlModel:(demo,fraud_model_v3,PROD)')

    const [url] = fetchMock.mock.calls[0]
    expect(url).toContain('/api/lineage/')
    expect(url).not.toContain('(demo,fraud_model_v3,PROD)') // should be encoded, not raw
  })
})
