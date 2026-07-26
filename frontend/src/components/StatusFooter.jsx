import React from 'react'
import './StatusFooter.css'

export default function StatusFooter({ result, stage }) {
  const wb = result?.writeback

  return (
    <footer className="status-footer mono">
      <div className="footer-item">
        <span className="footer-label">DataHub incident</span>
        <span className={wb?.datahub_incident_urn ? 'footer-value footer-value-ok' : 'footer-value'}>
          {wb?.datahub_incident_urn ?? '—'}
        </span>
      </div>
      <div className="footer-item">
        <span className="footer-label">GitHub issue</span>
        <span className={wb?.github_issue_url ? 'footer-value footer-value-ok' : 'footer-value'}>
          {wb?.github_issue_url ? (
            <a href={wb.github_issue_url} target="_blank" rel="noreferrer">{wb.github_issue_url}</a>
          ) : '— (set GITHUB_TOKEN / GITHUB_REPO to enable)'}
        </span>
      </div>
      <div className="footer-item footer-item-right">
        <span className="footer-label">write-back status</span>
        <span className="footer-value">{wb?.status ?? (stage === 'done' ? 'no incident — pipeline healthy' : 'pending')}</span>
      </div>
    </footer>
  )
}
