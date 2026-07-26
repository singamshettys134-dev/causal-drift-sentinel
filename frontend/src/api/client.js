const BASE = '/api'

export async function investigate({ modelUrn, injectDrift }) {
  const params = new URLSearchParams({
    model_urn: modelUrn,
    inject_drift: String(injectDrift),
  })
  const res = await fetch(`${BASE}/investigate?${params.toString()}`, { method: 'POST' })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`investigate failed: ${res.status} ${detail}`)
  }
  return res.json()
}

export async function getLineage(modelUrn) {
  const res = await fetch(`${BASE}/lineage/${encodeURIComponent(modelUrn)}`)
  if (!res.ok) throw new Error(`getLineage failed: ${res.status}`)
  return res.json()
}
