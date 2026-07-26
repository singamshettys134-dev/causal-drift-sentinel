// In local dev, Vite's proxy (vite.config.js) forwards /api -> localhost:8000.
// In production, frontend and backend are usually deployed on separate hosts,
// so set VITE_API_BASE_URL to the deployed backend's URL at build time.
const BASE = `${import.meta.env.VITE_API_BASE_URL || ''}/api`

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
