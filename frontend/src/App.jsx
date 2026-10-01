import { useMemo, useState } from 'react'

const API_URL = 'http://127.0.0.1:8000/api/v1/risk/analyze'

const defaultForm = {
  amount: 900,
  recipient_new: 0,
  device_changed: 0,
  location_changed: 0,
  transactions_last_1h: 1,
  transactions_last_24h: 3,
  avg_amount_30d: 1000,
  usual_transaction_hour: 14,
  hour: 14,
  account_age_days: 700,
}

const highForm = {
  amount: 21301.05,
  recipient_new: 1,
  device_changed: 1,
  location_changed: 1,
  transactions_last_1h: 4,
  transactions_last_24h: 10,
  avg_amount_30d: 1349.64,
  usual_transaction_hour: 15,
  hour: 4,
  account_age_days: 400,
}

function formatScore(value) {
  return Number(value ?? 0).toFixed(2)
}

function App() {
  const [form, setForm] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const riskClass = useMemo(() => {
    const level = result?.risk_level || ''
    if (level.includes('HIGH')) return 'risk-high'
    if (level.includes('REVIEW')) return 'risk-review'
    return 'risk-low'
  }, [result])

  function update(field, value) {
    setForm((current) => ({
      ...current,
      [field]: value,
    }))
  }

  function loadPreset(preset) {
    setResult(null)
    setError('')
    setForm(preset)
  }

  async function analyzeTransaction(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...form,
          amount: Number(form.amount),
          recipient_new: Number(form.recipient_new),
          device_changed: Number(form.device_changed),
          location_changed: Number(form.location_changed),
          transactions_last_1h: Number(form.transactions_last_1h),
          transactions_last_24h: Number(form.transactions_last_24h),
          avg_amount_30d: Number(form.avg_amount_30d),
          usual_transaction_hour: Number(form.usual_transaction_hour),
          hour: Number(form.hour),
          account_age_days: Number(form.account_age_days),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'API request failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        `${err.message} Make sure FastAPI is running on http://127.0.0.1:8000.`,
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">AI FINANCIAL SAFETY & INTELLIGENCE</div>
          <h1>Upay Sentinel</h1>
          <p>Transaction risk analysis, behavioral anomaly detection and explainable alerts.</p>
        </div>
        <div className="status-pill">
          <span className="status-dot" />
          AI ENGINE ONLINE
        </div>
      </header>

      <main className="content-grid">
        <section className="panel input-panel">
          <div className="panel-heading">
            <div>
              <h2>Transaction Analyzer</h2>
              <p>Enter transaction context and send it to the Sentinel API.</p>
            </div>
          </div>

          <div className="preset-row">
            <button type="button" className="ghost-btn" onClick={() => loadPreset(defaultForm)}>
              Load Low-Risk Example
            </button>
            <button type="button" className="ghost-btn" onClick={() => loadPreset(highForm)}>
              Load High-Risk Example
            </button>
          </div>

          <form onSubmit={analyzeTransaction}>
            <div className="form-grid">
              <Field label="Transaction Amount" prefix="৳">
                <input type="number" step="0.01" value={form.amount} onChange={(e) => update('amount', e.target.value)} required />
              </Field>

              <Field label="30-Day Average Amount" prefix="৳">
                <input type="number" step="0.01" value={form.avg_amount_30d} onChange={(e) => update('avg_amount_30d', e.target.value)} required />
              </Field>

              <Field label="Transactions — Last 1h">
                <input type="number" min="0" value={form.transactions_last_1h} onChange={(e) => update('transactions_last_1h', e.target.value)} required />
              </Field>

              <Field label="Transactions — Last 24h">
                <input type="number" min="0" value={form.transactions_last_24h} onChange={(e) => update('transactions_last_24h', e.target.value)} required />
              </Field>

              <Field label="Usual Transaction Hour">
                <input type="number" min="0" max="23" value={form.usual_transaction_hour} onChange={(e) => update('usual_transaction_hour', e.target.value)} required />
              </Field>

              <Field label="Current Transaction Hour">
                <input type="number" min="0" max="23" value={form.hour} onChange={(e) => update('hour', e.target.value)} required />
              </Field>

              <Field label="Account Age (days)">
                <input type="number" min="0" value={form.account_age_days} onChange={(e) => update('account_age_days', e.target.value)} required />
              </Field>

              <Field label="Recipient Status">
                <select value={form.recipient_new} onChange={(e) => update('recipient_new', e.target.value)}>
                  <option value={0}>Known recipient</option>
                  <option value={1}>New recipient</option>
                </select>
              </Field>

              <Field label="Device">
                <select value={form.device_changed} onChange={(e) => update('device_changed', e.target.value)}>
                  <option value={0}>Usual device</option>
                  <option value={1}>Device changed</option>
                </select>
              </Field>

              <Field label="Location">
                <select value={form.location_changed} onChange={(e) => update('location_changed', e.target.value)}>
                  <option value={0}>Usual location</option>
                  <option value={1}>Location changed</option>
                </select>
              </Field>
            </div>

            <button className="analyze-btn" disabled={loading}>
              {loading ? 'Analyzing…' : 'Analyze Transaction'}
            </button>
          </form>

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="panel result-panel">
          {!result ? (
            <EmptyState />
          ) : (
            <>
              <div className="result-header">
                <div>
                  <div className="eyebrow">SENTINEL DECISION</div>
                  <h2>Risk Assessment</h2>
                </div>
                <div className={`risk-badge ${riskClass}`}>{result.risk_level}</div>
              </div>

              <div className="score-main">
                <div className="score-circle">
                  <div className="score-value">{formatScore(result.final_risk_score)}</div>
                  <div className="score-label">FINAL RISK / 100</div>
                </div>
                <div className="score-copy">
                  <p className="score-title">Combined Sentinel score</p>
                  <p>{result.recommended_action}</p>
                </div>
              </div>

              <div className="mini-grid">
                <Metric label="Transaction Risk" value={result.transaction_risk} />
                <Metric label="Behavior Anomaly" value={result.anomaly_score} />
                <Metric label="Model Threshold" value={result.model_threshold} suffix="%" />
              </div>

              <div className="section-block">
                <div className="section-title">Why was this flagged?</div>
                <div className="reason-list">
                  {result.reasons.map((reason, index) => (
                    <div className="reason-item" key={`${reason.feature}-${index}`}>
                      <div className={`reason-indicator ${reason.shap_value >= 0 ? 'positive' : 'negative'}`} />
                      <div className="reason-copy">
                        <strong>{index + 1}. {prettyFeature(reason.feature)}</strong>
                        <span>{reason.direction}</span>
                      </div>
                      <div className="reason-value">{reason.shap_value >= 0 ? '+' : ''}{Number(reason.shap_value).toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </section>
      </main>

      <footer>
        Prototype • Synthetic data only • Human review required for high-impact decisions
      </footer>
    </div>
  )
}

function Field({ label, prefix, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="input-wrap">
        {prefix && <b>{prefix}</b>}
        {children}
      </div>
    </label>
  )
}

function Metric({ label, value, suffix = '' }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{formatScore(value)}{suffix || ' / 100'}</strong>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="shield-icon">S</div>
      <h2>Ready for analysis</h2>
      <p>Submit a transaction to see its risk score, anomaly score, explanation and recommended action.</p>
    </div>
  )
}

function prettyFeature(feature) {
  return feature
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export default App
