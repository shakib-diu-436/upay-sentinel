import { useEffect, useMemo, useState } from 'react'

const API_URL =
  'http://127.0.0.1:8000/api/v1/investigation/analyze'

const CASES_URL =
  'http://127.0.0.1:8000/api/v1/cases?limit=10'

// ============================================================
// LOW-RISK EXAMPLE
// ============================================================

const defaultForm = {
  transaction_id: 'TX001758',
  customer_id: 'C01234',
  recipient_id: 'R04567',
  timestamp: '2026-07-01T14:20:00',

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

// ============================================================
// HIGH-RISK EXAMPLE
// ============================================================

const highForm = {
  transaction_id: 'TX005878',
  customer_id: 'C02354',
  recipient_id: 'R06576',
  timestamp: '2026-07-01T00:22:56',

  amount: 8348.93,

  recipient_new: 1,
  device_changed: 1,
  location_changed: 1,

  transactions_last_1h: 3,
  transactions_last_24h: 9,

  avg_amount_30d: 947.56,

  usual_transaction_hour: 8,
  hour: 0,

  account_age_days: 2145,
}

// ============================================================
// HELPERS
// ============================================================

function formatScore(value) {
  return Number(value ?? 0).toFixed(2)
}

function formatAmount(value) {
  return `৳${Number(value ?? 0).toLocaleString('en-BD', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return '—'
  }

  const date = new Date(timestamp)

  if (Number.isNaN(date.getTime())) {
    return timestamp
  }

  return date.toLocaleString('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

function prettyFeature(feature) {
  return String(feature || '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function getRiskClass(level = '') {
  if (level.includes('HIGH')) {
    return 'risk-high'
  }

  if (level.includes('REVIEW')) {
    return 'risk-review'
  }

  return 'risk-low'
}

function getRiskDotClass(level = '') {
  if (level.includes('HIGH')) {
    return 'dot-high'
  }

  if (level.includes('REVIEW')) {
    return 'dot-review'
  }

  return 'dot-low'
}

function getMiniRiskClass(level = '') {
  if (level.includes('HIGH')) {
    return 'risk-mini-high'
  }

  if (level.includes('REVIEW')) {
    return 'risk-mini-review'
  }

  return 'risk-mini-low'
}

// ============================================================
// APP
// ============================================================

function App() {
  const [form, setForm] = useState({
    ...defaultForm,
  })

  const [result, setResult] = useState(null)

  const [cases, setCases] = useState([])

  const [loading, setLoading] = useState(false)

  const [error, setError] = useState('')

  const [casesLoading, setCasesLoading] = useState(false)

  // ==========================================================
  // RISK CSS CLASS
  // ==========================================================

  const riskClass = useMemo(() => {
    return getRiskClass(result?.risk_level)
  }, [result])

  // ==========================================================
  // LOAD RECENT CASES FROM DATABASE
  // ==========================================================

  async function loadRecentCases() {
    setCasesLoading(true)

    try {
      const response = await fetch(CASES_URL)

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Could not load recent cases.',
        )
      }

      // Backend may return:
      // { cases: [...] }
      // or directly [...]
      const rawCases = Array.isArray(data)
        ? data
        : data.cases || data.results || []

      const normalizedCases = rawCases.map((item) => ({
        ...item,

        // Support multiple possible backend field names.
        amount:
          item.amount ??
          item.transaction_amount ??
          item.raw_transaction?.amount ??
          0,

        customer_id:
          item.customer_id ??
          item.raw_transaction?.customer_id ??
          '—',

        recipient_id:
          item.recipient_id ??
          item.raw_transaction?.recipient_id ??
          '—',

        timestamp:
          item.timestamp ??
          item.transaction_timestamp ??
          item.raw_transaction?.timestamp ??
          null,

        analyzedAt:
          item.created_at ??
          item.updated_at ??
          item.analyzedAt ??
          null,
      }))

      setCases(normalizedCases.slice(0, 10))
    } catch (err) {
      console.error(
        'Failed to load recent cases:',
        err,
      )
    } finally {
      setCasesLoading(false)
    }
  }

  // ==========================================================
  // LOAD CASES WHEN APP STARTS / REFRESHES
  // ==========================================================

  useEffect(() => {
    loadRecentCases()
  }, [])

  // ==========================================================
  // UPDATE FORM
  // ==========================================================

  function update(field, value) {
    setForm((current) => ({
      ...current,
      [field]: value,
    }))
  }

  // ==========================================================
  // LOAD PRESET
  // ==========================================================

  function loadPreset(preset) {
    setResult(null)

    setError('')

    setForm({
      ...preset,
    })
  }

  // ==========================================================
  // OPEN EXISTING CASE
  // ==========================================================

  async function openCase(item) {
    setError('')

    // If backend does not provide a case_id,
    // use the list item directly.
    if (!item.case_id) {
      setResult(item)
      return
    }

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/v1/cases/${encodeURIComponent(
          item.case_id,
        )}`,
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Could not load case.',
        )
      }

      // Support:
      // { case: {...} }
      // or directly {...}
      const fullCase = data.case || data

      setResult({
        ...item,
        ...fullCase,

        amount:
          fullCase.amount ??
          item.amount ??
          fullCase.transaction_amount ??
          0,
      })
    } catch (err) {
      // Fall back to the list record.
      console.error(
        'Could not open full case:',
        err,
      )

      setResult(item)
    }
  }

  // ==========================================================
  // ANALYZE TRANSACTION
  // ==========================================================

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
          transaction_id:
            form.transaction_id,

          customer_id:
            form.customer_id,

          recipient_id:
            form.recipient_id,

          timestamp:
            form.timestamp,

          amount:
            Number(form.amount),

          recipient_new:
            Number(form.recipient_new),

          device_changed:
            Number(form.device_changed),

          location_changed:
            Number(form.location_changed),

          transactions_last_1h:
            Number(
              form.transactions_last_1h,
            ),

          transactions_last_24h:
            Number(
              form.transactions_last_24h,
            ),

          avg_amount_30d:
            Number(
              form.avg_amount_30d,
            ),

          usual_transaction_hour:
            Number(
              form.usual_transaction_hour,
            ),

          hour:
            Number(form.hour),

          account_age_days:
            Number(
              form.account_age_days,
            ),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
            'API request failed.',
        )
      }

      // ------------------------------------------------------
      // Keep amount because the investigation API response
      // may not currently return the transaction amount.
      // ------------------------------------------------------

      const newCase = {
        ...data,

        amount:
          Number(form.amount),

        customer_id:
          data.customer_id ??
          form.customer_id,

        recipient_id:
          data.recipient_id ??
          form.recipient_id,

        timestamp:
          data.timestamp ??
          form.timestamp,

        analyzedAt:
          new Date().toISOString(),
      }

      setResult(newCase)

      // ------------------------------------------------------
      // Immediately update visible case list.
      // ------------------------------------------------------

      setCases((current) => [
        newCase,

        ...current.filter(
          (item) =>
            item.transaction_id !==
            data.transaction_id,
        ),
      ].slice(0, 10))

      // ------------------------------------------------------
      // IMPORTANT:
      // Reload from SQLite database so the frontend state
      // exactly matches the persistent backend state.
      // ------------------------------------------------------

      await loadRecentCases()
    } catch (err) {
      setError(
        `${err.message} Make sure FastAPI is running on http://127.0.0.1:8000.`,
      )
    } finally {
      setLoading(false)
    }
  }

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="app-shell">

      {/* ====================================================
          HEADER
      ==================================================== */}

      <header className="topbar">

        <div>

          <div className="eyebrow">
            AI FINANCIAL SAFETY & INTELLIGENCE
          </div>

          <h1>
            Upay Sentinel
          </h1>

          <p>
            Transaction risk analysis, behavioral anomaly
            detection and evidence-grounded investigation.
          </p>

        </div>

        <div className="status-pill">

          <span className="status-dot" />

          AI ENGINE ONLINE

        </div>

      </header>

      {/* ====================================================
          MAIN CONTENT
      ==================================================== */}

      <main className="content-grid">

        {/* ==================================================
            INPUT PANEL
        ================================================== */}

        <section className="panel input-panel">

          <div className="panel-heading">

            <div>

              <div className="eyebrow">
                TRANSACTION INPUT
              </div>

              <h2>
                Transaction Analyzer
              </h2>

              <p>
                Enter transaction context and send it to
                the Sentinel investigation API.
              </p>

            </div>

          </div>

          {/* PRESETS */}

          <div className="preset-row">

            <button
              type="button"
              className="ghost-btn"
              onClick={() =>
                loadPreset(defaultForm)
              }
            >
              Load Low-Risk Example
            </button>

            <button
              type="button"
              className="ghost-btn"
              onClick={() =>
                loadPreset(highForm)
              }
            >
              Load High-Risk Example
            </button>

          </div>

          {/* FORM */}

          <form onSubmit={analyzeTransaction}>

            <div className="form-grid">

              {/* Transaction ID */}

              <Field label="Transaction ID">

                <input
                  type="text"
                  value={
                    form.transaction_id
                  }
                  onChange={(e) =>
                    update(
                      'transaction_id',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Customer ID */}

              <Field label="Customer ID">

                <input
                  type="text"
                  value={
                    form.customer_id
                  }
                  onChange={(e) =>
                    update(
                      'customer_id',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Recipient ID */}

              <Field label="Recipient ID">

                <input
                  type="text"
                  value={
                    form.recipient_id
                  }
                  onChange={(e) =>
                    update(
                      'recipient_id',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Timestamp */}

              <Field
                label="Transaction Timestamp"
              >

                <input
                  type="datetime-local"
                  value={
                    form.timestamp
                      ? form.timestamp.slice(
                          0,
                          16,
                        )
                      : ''
                  }
                  onChange={(e) =>
                    update(
                      'timestamp',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Transaction Amount */}

              <Field
                label="Transaction Amount"
                prefix="৳"
              >

                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={
                    form.amount
                  }
                  onChange={(e) =>
                    update(
                      'amount',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Average Amount */}

              <Field
                label="30-Day Average Amount"
                prefix="৳"
              >

                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={
                    form.avg_amount_30d
                  }
                  onChange={(e) =>
                    update(
                      'avg_amount_30d',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Last 1h */}

              <Field
                label="Transactions — Last 1h"
              >

                <input
                  type="number"
                  min="0"
                  value={
                    form.transactions_last_1h
                  }
                  onChange={(e) =>
                    update(
                      'transactions_last_1h',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Last 24h */}

              <Field
                label="Transactions — Last 24h"
              >

                <input
                  type="number"
                  min="0"
                  value={
                    form.transactions_last_24h
                  }
                  onChange={(e) =>
                    update(
                      'transactions_last_24h',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Usual Hour */}

              <Field
                label="Usual Transaction Hour"
              >

                <input
                  type="number"
                  min="0"
                  max="23"
                  value={
                    form.usual_transaction_hour
                  }
                  onChange={(e) =>
                    update(
                      'usual_transaction_hour',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Current Hour */}

              <Field
                label="Current Transaction Hour"
              >

                <input
                  type="number"
                  min="0"
                  max="23"
                  value={
                    form.hour
                  }
                  onChange={(e) =>
                    update(
                      'hour',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Account Age */}

              <Field
                label="Account Age (days)"
              >

                <input
                  type="number"
                  min="0"
                  value={
                    form.account_age_days
                  }
                  onChange={(e) =>
                    update(
                      'account_age_days',
                      e.target.value,
                    )
                  }
                  required
                />

              </Field>

              {/* Recipient */}

              <Field label="Recipient Status">

                <select
                  value={
                    form.recipient_new
                  }
                  onChange={(e) =>
                    update(
                      'recipient_new',
                      e.target.value,
                    )
                  }
                >

                  <option value={0}>
                    Known recipient
                  </option>

                  <option value={1}>
                    New recipient
                  </option>

                </select>

              </Field>

              {/* Device */}

              <Field label="Device">

                <select
                  value={
                    form.device_changed
                  }
                  onChange={(e) =>
                    update(
                      'device_changed',
                      e.target.value,
                    )
                  }
                >

                  <option value={0}>
                    Usual device
                  </option>

                  <option value={1}>
                    Device changed
                  </option>

                </select>

              </Field>

              {/* Location */}

              <Field label="Location">

                <select
                  value={
                    form.location_changed
                  }
                  onChange={(e) =>
                    update(
                      'location_changed',
                      e.target.value,
                    )
                  }
                >

                  <option value={0}>
                    Usual location
                  </option>

                  <option value={1}>
                    Location changed
                  </option>

                </select>

              </Field>

            </div>

            <button
              className="analyze-btn"
              disabled={loading}
            >
              {loading
                ? 'Investigating…'
                : 'Analyze & Investigate Transaction'}
            </button>

          </form>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

        </section>

        {/* ==================================================
            RESULT PANEL
        ================================================== */}

        <section className="panel result-panel">

          {!result ? (

            <EmptyState />

          ) : (

            <>

              {/* RESULT HEADER */}

              <div className="result-header">

                <div>

                  <div className="eyebrow">
                    SENTINEL DECISION
                  </div>

                  <h2>
                    Risk Assessment
                  </h2>

                </div>

                <div
                  className={`risk-badge ${riskClass}`}
                >
                  {result.risk_level}
                </div>

              </div>

              {/* SCORE */}

              <div className="score-main">

                <div
                  className={`score-circle ${riskClass}`}
                >

                  <div className="score-value">
                    {formatScore(
                      result.final_risk_score,
                    )}
                  </div>

                  <div className="score-label">
                    FINAL RISK / 100
                  </div>

                </div>

                <div className="score-copy">

                  <p className="score-title">
                    {result.case_status}
                  </p>

                  <p>
                    {result.recommended_human_action}
                  </p>

                </div>

              </div>

              {/* METRICS */}

              <div className="mini-grid">

                <Metric
                  label="Transaction Risk"
                  value={
                    result.transaction_risk
                  }
                />

                <Metric
                  label="Behavior Anomaly"
                  value={
                    result.anomaly_score
                  }
                />

                <Metric
                  label="Case Status"
                  value={
                    result.case_status
                  }
                  text
                />

              </div>

              {/* =================================================
                  TRANSACTION INFORMATION
              ================================================= */}

              <div className="section-block">

                <div className="section-title">
                  Transaction Information
                </div>

                <div className="transaction-grid">

                  <InfoCard
                    label="Transaction ID"
                    value={
                      result.transaction_id
                    }
                  />

                  <InfoCard
                    label="Customer ID"
                    value={
                      result.customer_id
                    }
                  />

                  <InfoCard
                    label="Recipient ID"
                    value={
                      result.recipient_id
                    }
                  />

                  <InfoCard
                    label="Amount"
                    value={
                      formatAmount(
                        result.amount,
                      )
                    }
                  />

                  <InfoCard
                    label="Timestamp"
                    value={
                      formatTimestamp(
                        result.timestamp,
                      )
                    }
                  />

                </div>

              </div>

              {/* =================================================
                  EVIDENCE
              ================================================= */}

              <div className="section-block">

                <div className="section-title">
                  Why Is This Risky?
                </div>

                <div className="evidence-list">

                  {result.evidence?.map(
                    (item, index) => (

                      <div
                        className="evidence-item"
                        key={`${item.type}-${index}`}
                      >

                        <div
                          className={`evidence-dot evidence-${String(
                            item.severity,
                          ).toLowerCase()}`}
                        />

                        <div className="evidence-copy">

                          <div className="evidence-top">

                            <strong>
                              {item.title}
                            </strong>

                            <span
                              className={`severity-badge severity-${String(
                                item.severity,
                              ).toLowerCase()}`}
                            >
                              {item.severity}
                            </span>

                          </div>

                          <p>
                            {item.detail}
                          </p>

                        </div>

                      </div>

                    ),
                  )}

                </div>

              </div>

              {/* =================================================
                  MODEL EXPLANATION
              ================================================= */}

              <div className="section-block">

                <div className="section-title">
                  Model Explanation
                </div>

                <div className="reason-list">

                  {result.model_reasons?.map(
                    (reason, index) => (

                      <div
                        className="reason-item"
                        key={`${reason.feature}-${index}`}
                      >

                        <div
                          className={`reason-indicator ${
                            reason.shap_value >= 0
                              ? 'positive'
                              : 'negative'
                          }`}
                        />

                        <div className="reason-copy">

                          <strong>
                            {index + 1}.{' '}
                            {reason.label ||
                              prettyFeature(
                                reason.feature,
                              )}
                          </strong>

                          <span>
                            {reason.direction}
                          </span>

                        </div>

                        <div className="reason-value">

                          {reason.shap_value >= 0
                            ? '+'
                            : ''}

                          {Number(
                            reason.shap_value,
                          ).toFixed(3)}

                        </div>

                      </div>

                    ),
                  )}

                </div>

              </div>

              {/* =================================================
                  AI INVESTIGATION SUMMARY
              ================================================= */}

              <div className="section-block">

                <div className="section-title">
                  AI Investigation Summary
                </div>

                <div className="summary-box">
                  {result.summary}
                </div>

              </div>

              {/* =================================================
                  HUMAN ACTION
              ================================================= */}

              <div className="section-block">

                <div className="section-title">
                  Recommended Human Action
                </div>

                <div className="action-box">

                  <div className="action-icon">
                    !
                  </div>

                  <div>
                    {
                      result.recommended_human_action
                    }
                  </div>

                </div>

              </div>

              {/* =================================================
                  RESPONSIBLE AI NOTICE
              ================================================= */}

              <div className="notice-box">

                <strong>
                  Human-in-the-loop:
                </strong>{' '}

                {result.notice}

              </div>

            </>

          )}

        </section>

      </main>

      {/* ======================================================
          INVESTIGATION CASE HISTORY
      ====================================================== */}

      {cases.length > 0 && (

        <section className="panel cases-panel">

          <div className="cases-header">

            <div>

              <div className="eyebrow">
                INVESTIGATION QUEUE
              </div>

              <h2>
                Recent Cases
              </h2>

              <p>
                Recently analyzed transactions and their
                Sentinel risk assessments.
              </p>

            </div>

            <div className="case-count">

              {casesLoading
                ? 'LOADING'
                : (
                  <>
                    {cases.length}{' '}
                    CASE
                    {cases.length !== 1
                      ? 'S'
                      : ''}
                  </>
                )}

            </div>

          </div>

          <div className="case-list">

            {cases.map(
              (item, index) => (

                <button
                  type="button"
                  className={`case-card ${
                    result?.transaction_id ===
                    item.transaction_id
                      ? 'case-active'
                      : ''
                  }`}
                  key={`${item.transaction_id}-${index}`}
                  onClick={() =>
                    openCase(item)
                  }
                >

                  {/* STATUS DOT */}

                  <div className="case-status-dot">

                    <span
                      className={getRiskDotClass(
                        item.risk_level,
                      )}
                    />

                  </div>

                  {/* CASE MAIN */}

                  <div className="case-main">

                    <div className="case-top">

                      <strong>
                        {item.transaction_id}
                      </strong>

                      <span
                        className={`risk-mini ${getMiniRiskClass(
                          item.risk_level,
                        )}`}
                      >
                        {item.risk_level}
                      </span>

                    </div>

                    <div className="case-meta">

                      <span>
                        Customer:{' '}
                        {item.customer_id}
                      </span>

                      <span>
                        Recipient:{' '}
                        {item.recipient_id}
                      </span>

                      <span>
                        {formatAmount(
                          item.amount,
                        )}
                      </span>

                    </div>

                  </div>

                  {/* SCORE */}

                  <div className="case-score">

                    <strong>
                      {formatScore(
                        item.final_risk_score,
                      )}
                    </strong>

                    <span>
                      / 100
                    </span>

                  </div>

                  {/* ARROW */}

                  <div className="case-arrow">
                    →
                  </div>

                </button>

              ),
            )}

          </div>

        </section>

      )}

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>
        Prototype • Synthetic data only • Human review required
        for high-impact decisions
      </footer>

    </div>
  )
}

// ============================================================
// FIELD
// ============================================================

function Field({
  label,
  prefix,
  children,
}) {
  return (
    <label className="field">

      <span>
        {label}
      </span>

      <div className="input-wrap">

        {prefix && (
          <b>
            {prefix}
          </b>
        )}

        {children}

      </div>

    </label>
  )
}

// ============================================================
// METRIC
// ============================================================

function Metric({
  label,
  value,
  text = false,
}) {
  return (
    <div className="metric-card">

      <span>
        {label}
      </span>

      <strong>
        {text
          ? value
          : `${formatScore(value)} / 100`}
      </strong>

    </div>
  )
}

// ============================================================
// INFO CARD
// ============================================================

function InfoCard({
  label,
  value,
}) {
  return (
    <div className="info-card">

      <span>
        {label}
      </span>

      <strong>
        {value || '—'}
      </strong>

    </div>
  )
}

// ============================================================
// EMPTY STATE
// ============================================================

function EmptyState() {
  return (
    <div className="empty-state">

      <div className="shield-icon">
        S
      </div>

      <h2>
        Ready for investigation
      </h2>

      <p>
        Submit a transaction to see its risk score,
        behavioral anomaly, evidence, model explanation,
        investigation summary and recommended human action.
      </p>

    </div>
  )
}

export default App