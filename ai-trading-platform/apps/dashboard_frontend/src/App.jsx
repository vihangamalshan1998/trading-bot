import React, { useState, useEffect, useRef, useMemo } from 'react'
import { LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer } from 'recharts'
import Swal from 'sweetalert2'
import './App.css'

// FeatureEngine output indices (core/schemas/state_layout.py: state slot = 20 + index)
const F = { RET_24H: 6, DEPTH_IMB_01: 25, DEPTH_IMB_1: 26, FUNDING: 80, LS_RATIO: 84, TIME_TO_FUNDING: 86, XEX_SPREAD: 87, XEX_MEAN: 88, XEX_OI_1H: 89 }
const ROUND_TRIP_FEE = 0.0007 // maker entry 0.02% + taker exit 0.05%
const HORIZON_W = [0.1, 0.6, 0.3] // exit_policy.HORIZON_WEIGHTS: blended 1-4h expected move
const TRAINER = 'ai-trainer'

// ---------------------------------------------------------------- helpers
const fmtPct = (v, d = 2) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`
const fmtPP = (v, d = 2) => `${v >= 0 ? '+' : ''}${Number(v || 0).toFixed(d)}%` // value already in %
const tone = (v, eps = 0) => (v > eps ? 'pos' : v < -eps ? 'neg' : '')
const money = (v) => `${v < 0 ? '-' : ''}$${Math.abs(v || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const now = () => Date.now() / 1000
const ago = (ts) => {
  if (!ts) return '—'
  const s = Math.max(0, now() - ts)
  if (s < 90) return `${Math.round(s)}s ago`
  if (s < 5400) return `${Math.round(s / 60)} min ago`
  if (s < 172800) return `${(s / 3600).toFixed(1)} h ago`
  return `${(s / 86400).toFixed(1)} days ago`
}
const duration = (since) => {
  if (!since) return '—'
  const s = Math.max(0, now() - since)
  if (s < 3600) return `${Math.round(s / 60)}m`
  if (s < 172800) return `${(s / 3600).toFixed(1)}h`
  return `${(s / 86400).toFixed(1)}d`
}
const expectedMove = (s) => HORIZON_W[0] * (s.ai_critic_return || 0) + HORIZON_W[1] * (s.ai_critic_1h || 0) + HORIZON_W[2] * (s.ai_critic_4h || 0)

const usePoll = (url, ms, enabled = true) => {
  const [data, setData] = useState(null)
  const [error, setError] = useState(false)
  useEffect(() => {
    if (!enabled) return
    let alive = true
    const load = async () => {
      try {
        const res = await fetch(url)
        const json = await res.json()
        if (alive) { setData(json); setError(false) }
      } catch {
        if (alive) setError(true)
      }
    }
    load()
    const t = setInterval(load, ms)
    return () => { alive = false; clearInterval(t) }
  }, [url, ms, enabled])
  return [data, error]
}

const confirmAction = async (title, text, icon, confirmButtonText, confirmButtonColor) => {
  const r = await Swal.fire({ title, text, icon, showCancelButton: true, confirmButtonText, confirmButtonColor, cancelButtonColor: '#2f3a4d' })
  return r.isConfirmed
}
const postAction = async (url) => {
  try {
    const res = await fetch(url, { method: 'POST' })
    const json = await res.json()
    Swal.fire(json.status === 'success' ? 'Done' : 'Something went wrong', json.message, json.status === 'success' ? 'success' : 'error')
  } catch {
    Swal.fire('Something went wrong', 'Could not reach the dashboard server.', 'error')
  }
}

// ---------------------------------------------------------------- icons
const Icon = ({ name }) => {
  const p = {
    overview: 'M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z',
    coins: 'M12 2a10 10 0 100 20 10 10 0 000-20zm1 15.9V19h-2v-1.1c-1.7-.3-3-1.4-3.1-3h1.9c.1.9.8 1.4 2.2 1.4 1.5 0 1.9-.7 1.9-1.2 0-.6-.3-1.2-2-1.6-1.9-.5-3.3-1.2-3.3-2.9 0-1.4 1.1-2.4 2.4-2.7V7h2v1c1.4.3 2.3 1.3 2.4 2.7H14.5c-.1-.8-.6-1.3-1.7-1.3-1.1 0-1.7.5-1.7 1.1 0 .5.4.9 2 1.3 1.6.4 3.3 1.1 3.3 3 0 1.4-1 2.5-3.4 3.1z',
    brain: 'M12 2a4 4 0 00-4 4v.3A4 4 0 005 10a4 4 0 001 2.6A4 4 0 008 19.7V20a2 2 0 004 0V2zm0 0a4 4 0 014 4v.3A4 4 0 0119 10a4 4 0 01-1 2.6 4 4 0 01-2 7.1V20a2 2 0 01-4 0',
    trades: 'M4 6h16v2H4V6zm0 5h10v2H4v-2zm0 5h16v2H4v-2zm14-6l3 2-3 2v-4z',
    services: 'M4 4h16v6H4V4zm0 10h16v6H4v-6zm3-8v2h2V6H7zm0 10v2h2v-2H7z',
    power: 'M13 3h-2v10h2V3zm4.8 2.2l-1.4 1.4A6.96 6.96 0 0119 12a7 7 0 11-11.4-5.4L6.2 5.2A9 9 0 1021 12a8.96 8.96 0 00-3.2-6.8z',
    check: 'M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z',
    alert: 'M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z',
    cpu: 'M9 9h6v6H9V9zm-2 12v-2H5a2 2 0 01-2-2v-2H1v-2h2V9H1V7h2V5a2 2 0 012-2h2V1h2v2h2V1h2v2h2V1h2v2h2a2 2 0 012 2v2h2v2h-2v4h2v2h-2v2a2 2 0 01-2 2h-2v2h-2v-2h-2v2h-2v-2H9v2H7zm-2-4h14V5H5v12z',
  }[name]
  return <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d={p} /></svg>
}

// ---------------------------------------------------------------- small components
const Info = ({ text }) => <span className="info">?<span className="tip">{text}</span></span>
const KV = ({ label, value, cls = '', help }) => (
  <div className="kv">
    <div className="kv-label">{label}{help && <Info text={help} />}</div>
    <div className={`kv-value ${cls}`}>{value}</div>
  </div>
)
const Card = ({ title, help, desc, right, children, style }) => (
  <section className="card" style={style}>
    {(title || right) && (
      <div className="card-head">
        {title && <h3 className="card-title">{title}</h3>}
        {help && <Info text={help} />}
        {right && <div className="right">{right}</div>}
      </div>
    )}
    {desc && <p className="card-desc">{desc}</p>}
    {children}
  </section>
)
const Empty = ({ title, children }) => <div className="empty"><strong>{title}</strong>{children}</div>
const Dot = ({ color }) => <span className={`dot ${color}`} />
const statusColor = (s) => (s === 'online' ? 'green' : s === 'silent' || s === 'launching' || s === 'file' ? 'amber' : s === 'not-reporting' ? 'grey' : 'red')
const Dir = ({ side }) => {
  const s = (side || '').toUpperCase()
  const cls = s.includes('LONG') ? 'long' : s.includes('SHORT') ? 'short' : 'flat'
  return <span className={`dir ${cls}`}>{cls === 'long' ? '▲ LONG' : cls === 'short' ? '▼ SHORT' : '• FLAT'}</span>
}
const Meter = ({ value, max = 1, color = '' }) => (
  <div className="meter">
    <div className="meter-track"><div className={`meter-fill ${color}`} style={{ width: `${Math.min(100, (Math.abs(value) / max) * 100)}%` }} /></div>
    <span className="meter-val">{(value * 100).toFixed(0)}%</span>
  </div>
)
const SplitMeter = ({ value, max }) => {
  const w = Math.min(50, (Math.abs(value) / max) * 50)
  return (
    <div className="split-meter" title={`${value.toFixed(2)}%`}>
      <div className="mid" />
      <div className="fill" style={{ left: value >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: value >= 0 ? 'var(--green)' : 'var(--red)' }} />
    </div>
  )
}
const CoinTag = ({ sym }) => {
  const base = sym.replace('USDT', '').replace('1000', '')
  return <span className="coin"><span className="coin-icon">{base.slice(0, 3)}</span>{sym.replace('USDT', '')}</span>
}

// ---------------------------------------------------------------- positions
const positionMath = (pos) => {
  const entry = pos.entryPrice || pos.entry_price || 0
  const notional = Math.abs(pos.quantity) * entry
  const margin = pos.margin ? pos.margin : notional / (pos.leverage || 1)
  const fees = notional * ROUND_TRIP_FEE
  const gross = pos.pnl || 0
  const net = gross - fees
  return { entry, notional, margin, fees, gross, net, roi: margin > 0 ? (net / margin) * 100 : 0,
    mark: pos.markPrice || pos.mark_price || entry, liq: pos.liquidationPrice || pos.liquidation_price || 0 }
}
const openPositions = (positions) => (positions || []).filter((p) => p.quantity && p.quantity !== 0)

const fmtDur = (sec) => {
  if (!sec || sec < 0) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  const h = Math.floor(sec / 3600)
  return h < 48 ? `${h}h ${Math.round((sec % 3600) / 60)}m` : `${Math.round(h / 24)}d`
}
const PositionCards = ({ positions }) => {
  const open = openPositions(positions)
  if (!open.length) return <Empty title="No open positions">The bot will open a trade when the AI finds one worth the fees.</Empty>
  return (
    <div className="pos-cards">
      {open.map((pos, i) => {
        const m = positionMath(pos)
        const move = m.entry ? ((m.mark - m.entry) / m.entry) * (pos.side === 'SHORT' ? -1 : 1) : 0
        const liqDist = m.liq && m.mark ? Math.abs(m.liq - m.mark) / m.mark : 0
        return (
          <div key={i} className="pos-card">
            <div className="pos-card-head">
              <CoinTag sym={pos.symbol} />
              <span className="small muted">{pos.leverage}x · {pos.entry_time ? `open ${fmtDur(now() - pos.entry_time)}` : (pos.marginType || 'cross')}</span>
              <Dir side={pos.side} />
            </div>
            <div className="kv-grid">
              <KV label="Profit after fees" value={`${money(m.net)} (${m.roi >= 0 ? '+' : ''}${m.roi.toFixed(1)}%)`} cls={tone(m.net)} help="Unrealized profit minus estimated entry + exit fees, and as % of the margin used." />
              <KV label="Price move" value={fmtPct(move, 2)} cls={tone(move)} help="How far the price has moved in the trade's favour (+) or against it (-) since entry." />
              <KV label="Entry price" value={m.entry ? m.entry.toPrecision(6) : '—'} />
              <KV label="Current price" value={m.mark ? m.mark.toPrecision(6) : '—'} />
              <KV label="Margin used" value={money(m.margin)} />
              <KV label="Position size" value={money(m.notional)} help={`${pos.quantity} ${pos.symbol.replace('USDT', '')} at ${pos.leverage}x leverage.`} />
              <KV label="Gross P&L" value={money(m.gross)} cls={tone(m.gross)} />
              <KV label="Est. fees" value={money(m.fees)} help="Estimated entry + exit fees for this position." />
              <KV label="Liquidation" value={m.liq ? `${m.liq.toPrecision(6)} (${(liqDist * 100).toFixed(1)}% away)` : 'Far away (cross)'} cls={m.liq && liqDist < 0.03 ? 'neg' : 'warn'}
                help="Price at which the exchange would force-close. On cross margin Binance reports none while the account balance easily covers the position." />
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------- overview page
const StatusTile = ({ icon, color, label, value, note }) => (
  <Card>
    <div className="tile">
      <div className={`tile-icon ${color}`}><Icon name={icon} /></div>
      <div>
        <div className="tile-label">{label}</div>
        <div className="tile-value">{value}</div>
        <div className="tile-note">{note}</div>
      </div>
    </div>
  </Card>
)

const Overview = ({ ctx }) => {
  const { stats, brain, services, state, markets, blocked, killActive, go, confLabel } = ctx
  const svc = services?.services || []
  const pm2 = svc.filter((s) => !s.remote)
  const down = pm2.filter((s) => s.status !== 'online' && s.status !== 'file')
  const trainer = svc.find((s) => s.remote)
  const champ = brain?.champion
  const g = stats.global_context || {}
  const m = stats.macro || {}
  const fg = m.fear_greed_index ?? 0.5
  const toNext = g.econ_time_to_next ?? 1
  const imp = g.econ_next_importance || 0

  const ideas = markets
    .map(([sym, s]) => ({ sym, s, side: s.ai_predicted_side, conf: s.ai_confidence_calibrated ?? s.ai_confidence ?? 0, move: expectedMove(s) }))
    .filter((x) => x.side === 'LONG' || x.side === 'SHORT')
    .sort((a, b) => b.conf * Math.abs(b.move) - a.conf * Math.abs(a.move))
    .slice(0, 6)

  return (
    <>
      <div className="grid cols-4">
        <StatusTile icon="power" color={killActive ? 'red' : 'green'} label="Trading"
          value={killActive ? 'Paused' : 'Active'}
          note={killActive ? 'Kill switch is on. No new trades.' : `${openPositions(state.positions).length} open position(s)`} />
        <StatusTile icon="brain" color={champ?.is_v3 ? 'green' : 'amber'} label="Live brain"
          value={champ ? (champ.is_v3 ? 'Proven on unseen data' : 'Old brain (unproven)') : '—'}
          note={champ ? `Version ${champ.version}` : 'Loading…'} />
        <StatusTile icon="services" color={down.length ? 'red' : 'green'} label="Services"
          value={pm2.length ? `${pm2.length - down.length} of ${pm2.length} running` : '—'}
          note={down.length ? `Down: ${down.map((s) => s.name).join(', ')}` : 'All healthy'} />
        <StatusTile icon="cpu" color={trainer?.status === 'online' ? 'green' : 'red'} label="Trainer (laptop)"
          value={trainer?.status === 'online' ? 'Training' : 'Not reporting'}
          note={trainer?.status === 'online' ? `Seen ${trainer.last_seen_s}s ago` : 'Start MASTER_TRAINER on the laptop'} />
      </div>

      {!champ?.is_v3 && champ && !killActive && (
        <div className="banner amber">
          <Icon name="alert" />
          <span className="grow">The live brain has not passed the honest after-fees test yet. Recommended: turn the kill switch on until a tournament winner arrives.</span>
          <button className="btn sm" onClick={() => go('brain')}>See brain</button>
        </div>
      )}

      <div className="grid side">
        <Card title="Open positions" right={<button className="btn sm ghost" onClick={() => go('trades')}>All trades →</button>}>
          <PositionCards positions={state.positions} />
        </Card>
        <Card title="Market mood" help="Market-wide signals the AI reads every minute.">
          <div className="kv-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <KV label="Fear & Greed" value={`${Math.round(fg * 100)} / 100`} cls={fg > 0.6 ? 'pos' : fg < 0.4 ? 'neg' : ''} />
            <KV label="News sentiment" value={(stats.latest_sentiment || 0).toFixed(2)} cls={tone(stats.latest_sentiment || 0, 0.02)} help="Average of every headline's score (-1 very bearish … +1 very bullish)." />
            <KV label="Options fear (DVOL)" value={`${((g.dvol_btc || 0) * 100).toFixed(1)}`} cls={(g.dvol_btc_change_1h || 0) > 0.05 ? 'neg' : ''} help="Deribit BTC implied volatility. Higher = bigger moves expected. A sudden jump is a warning." />
            <KV label="US buyers (Coinbase)" value={fmtPct(g.coinbase_premium_btc || 0, 3)} cls={tone(g.coinbase_premium_btc || 0, 0.0005)} help="Coinbase BTC price vs Binance. Positive = US demand pushing price." />
            <KV label="Next US release" value={toNext >= 1 ? 'None in 24h' : `In ${(toNext * 24).toFixed(1)}h`} cls={toNext < 2 / 24 ? 'neg' : ''} help="Fed / CPI / jobs reports often move crypto 1–3% in minutes." />
            <KV label="Importance" value={imp >= 1 ? 'Fed decision' : imp >= 0.8 ? 'CPI / jobs' : imp >= 0.5 ? 'High' : imp > 0 ? 'Medium' : '—'} cls={imp >= 0.8 ? 'neg' : ''} />
          </div>
        </Card>
      </div>

      <div className="grid cols-2">
        <Card title="What the AI wants right now" help="Coins where the AI currently leans long or short, ranked by win chance × expected move. It only trades if all safety checks pass. With the old brain the confidence is unproven." right={<button className="btn sm ghost" onClick={() => go('coins')}>All coins →</button>}>
          {ideas.length === 0 ? <Empty title="No strong ideas">The AI is neutral on every coin right now.</Empty> : (
            <div className="table-wrap">
              <table className="tbl">
                <thead><tr><th>Coin</th><th>Direction</th><th>{confLabel}</th><th>Expected move (1–4h)</th></tr></thead>
                <tbody>
                  {ideas.map((x) => (
                    <tr key={x.sym}>
                      <td><CoinTag sym={x.sym} /> {blocked.has(x.sym) && <span className="badge red">blocked</span>}</td>
                      <td><Dir side={x.side} /></td>
                      <td><Meter value={x.conf} color={x.conf >= 0.55 ? 'green' : 'amber'} /></td>
                      <td className={tone(x.move)}>{fmtPP(x.move)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <Card title="Latest headlines" right={<span className="small muted">{stats.news_count || 0} analysed</span>}>
          <div className="list">
            {(stats.latest_headlines || []).slice(0, 7).map((h, i) => {
              const title = typeof h === 'string' ? h : h.title
              const url = typeof h === 'string' ? null : h.url
              return <div key={i} className="list-row">{url ? <a className="headline" href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</div>
            })}
            {!(stats.latest_headlines || []).length && <Empty title="No headlines yet" />}
          </div>
        </Card>
      </div>

      <Card title="World markets (1 day change)" help="Traditional markets that often move crypto. Updated every 5 minutes.">
        <div className="kv-grid">
          <KV label="S&P 500" value={fmtPct(m.sp500_momentum || 0)} cls={tone(m.sp500_momentum || 0)} />
          <KV label="Nasdaq 100" value={fmtPct(m.ndx_momentum || 0)} cls={tone(m.ndx_momentum || 0)} />
          <KV label="VIX (stock fear)" value={fmtPct(m.vix_momentum || 0)} cls={tone(-(m.vix_momentum || 0))} />
          <KV label="US Dollar (DXY)" value={fmtPct(m.dxy_momentum || 0)} cls={tone(-(m.dxy_momentum || 0))} />
          <KV label="Gold" value={fmtPct(m.gold_momentum || 0)} />
          <KV label="US 10Y yield" value={fmtPct(m.treasury_yield_momentum || 0)} />
          <KV label="ETH options fear" value={`${((g.dvol_eth || 0) * 100).toFixed(1)}`} />
          <KV label="US buyers (ETH)" value={fmtPct(g.coinbase_premium_eth || 0, 3)} cls={tone(g.coinbase_premium_eth || 0, 0.0005)} />
        </div>
      </Card>
    </>
  )
}

// ---------------------------------------------------------------- coins page
const Coins = ({ ctx }) => {
  const { markets, blocked, isV3, confLabel, killActive } = ctx
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState({ key: 'conf', dir: -1 })
  const [open, setOpen] = useState(null)

  const rows = markets.map(([sym, s]) => {
    const f = s.features || []
    const fv = (i) => (f.length > i ? f[i] : 0)
    return {
      sym, s,
      side: s.ai_predicted_side || 'WAITING',
      conf: s.ai_confidence_calibrated ?? s.ai_confidence ?? 0,
      raw: s.ai_confidence || 0,
      move: expectedMove(s),
      size: s.ai_target_size || 0,
      price: s.mid_price || 0,
      chg: fv(F.RET_24H),
      spread: s.spread && s.mid_price ? (s.spread / s.mid_price) * 10000 : 0,
      funding: fv(F.FUNDING),
      xexFunding: fv(F.XEX_MEAN),
      book: fv(F.DEPTH_IMB_01),
      book1: fv(F.DEPTH_IMB_1),
      oi1h: fv(F.XEX_OI_1H),
      fundIn: fv(F.TIME_TO_FUNDING),
      blocked: blocked.has(sym),
      stale: !killActive && !!s.ai_updated && now() - s.ai_updated > 120,
    }
  })
  const filtered = rows.filter((r) => filter === 'all' || (filter === 'long' && r.side === 'LONG') || (filter === 'short' && r.side === 'SHORT') || (filter === 'blocked' && r.blocked))
  const sorted = [...filtered].sort((a, b) => {
    const va = a[sort.key], vb = b[sort.key]
    if (typeof va === 'string') return sort.dir * va.localeCompare(vb)
    return sort.dir * ((sort.key === 'move' ? Math.abs(va) - Math.abs(vb) : va - vb))
  })
  const head = (key, label, help) => (
    <th className="sortable" onClick={() => setSort({ key, dir: sort.key === key ? -sort.dir : -1 })}>
      {label}{sort.key === key ? (sort.dir < 0 ? ' ↓' : ' ↑') : ''}{help && <> <Info text={help} /></>}
    </th>
  )
  const counts = { all: rows.length, long: rows.filter((r) => r.side === 'LONG').length, short: rows.filter((r) => r.side === 'SHORT').length, blocked: rows.filter((r) => r.blocked).length }

  return (
    <Card title="All coins" desc="Click a coin for details. Click a column title to sort."
      right={(
        <div className="segmented">
          {['all', 'long', 'short', 'blocked'].map((k) => (
            <button key={k} className={filter === k ? 'active' : ''} onClick={() => setFilter(k)}>{k[0].toUpperCase() + k.slice(1)} ({counts[k]})</button>
          ))}
        </div>
      )}>
      {rows.length === 0 ? <Empty title="Waiting for market data">Is the market-collector service running?</Empty> : (
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                {head('sym', 'Coin')}
                {head('price', 'Price')}
                {head('chg', '24h')}
                {head('side', 'AI view')}
                {head('conf', confLabel, isV3 ? 'The AI confidence mapped to the win rate it really achieved on unseen data.' : "The old brain's own confidence. It was never checked against real results, so treat it as unproven.")}
                {head('move', 'Expected move', 'AI forecast of the price move over the next 1–4 hours (blend of its 5m / 1h / 4h forecasts).')}
                {head('size', 'Bet size', 'How much of the per-trade margin budget the AI wants to use (25-100%, bigger when it is confident and expects a big move). The bot never uses less than 70% of the budget, and the budget itself shrinks after losses on this coin.')}
                {head('funding', 'Funding', 'Paid every 8h by longs to shorts (positive) or shorts to longs (negative). Extreme = crowded.')}
                {head('book', 'Order book', 'Buy vs sell orders within 0.1% of price: + more buyers, - more sellers.')}
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <React.Fragment key={r.sym}>
                  <tr className={`clickable ${open === r.sym ? 'selected' : ''}`} onClick={() => setOpen(open === r.sym ? null : r.sym)}>
                    <td><CoinTag sym={r.sym} /></td>
                    <td className="strong">{r.price ? r.price.toPrecision(6) : '—'}</td>
                    <td className={tone(r.chg)}>{fmtPct(r.chg, 1)}</td>
                    <td><Dir side={r.side} /></td>
                    <td><Meter value={r.conf} color={r.conf >= 0.55 ? 'green' : r.conf >= 0.45 ? 'amber' : 'red'} /></td>
                    <td className={tone(r.move)}>{fmtPP(r.move)}</td>
                    <td>{r.side === 'LONG' || r.side === 'SHORT' ? <Meter value={r.size} /> : <span className="faint">—</span>}</td>
                    <td className={Math.abs(r.funding) > 0.0003 ? 'warn' : ''}>{fmtPct(r.funding, 4)}</td>
                    <td><SplitMeter value={r.book} max={1} /></td>
                    <td>
                      {r.blocked && <span className="badge red">Blocked</span>}{' '}
                      {r.stale && <span className="badge amber" title="The AI has not updated this coin for over 2 minutes. Check the trading-bot service.">Stale</span>}
                      {!r.blocked && !r.stale && (killActive ? <span className="badge grey" title="Kill switch is on: the AI view is not being refreshed.">Paused</span> : <span className="badge grey">OK</span>)}
                    </td>
                  </tr>
                  {open === r.sym && (
                    <tr className="detail-row">
                      <td colSpan="10">
                        <div className="detail-grid">
                          <KV label="Bet size wanted" value={`${(r.size * 100).toFixed(0)}%`} help="How much of the per-trade margin budget the AI wants to use (25-100%, bigger when it is confident and expects a big move). The bot never uses less than 70% of the budget, and the budget itself shrinks after losses on this coin." />
                          <KV label="Bet size used" value={`${(Math.max(0.7, r.size) * 100).toFixed(0)}% of budget`} help="What the bot actually sizes with: never below 70%." />
                          <KV label="Forecast 5 min" value={fmtPP(r.s.ai_critic_return || 0)} cls={tone(r.s.ai_critic_return || 0)} />
                          <KV label="Forecast 1 hour" value={fmtPP(r.s.ai_critic_1h || 0)} cls={tone(r.s.ai_critic_1h || 0)} />
                          <KV label="Forecast 4 hours" value={fmtPP(r.s.ai_critic_4h || 0)} cls={tone(r.s.ai_critic_4h || 0)} />
                          <KV label="Raw confidence" value={`${(r.raw * 100).toFixed(0)}%`} help="The brain's own confidence before calibration." />
                          <KV label="Spread" value={`${r.spread.toFixed(1)} bps`} cls={r.spread > 6 ? 'neg' : ''} />
                          <KV label="Funding (3 exchanges)" value={fmtPct(r.xexFunding, 4)} help="Average of Binance, Bybit and OKX funding." />
                          <KV label="Order book within 1%" value={r.book1.toFixed(2)} cls={tone(r.book1, 0.1)} />
                          <KV label="Open interest 1h" value={fmtPct(r.oi1h, 1)} help="Change in total open interest across Binance, Bybit and OKX over the last hour." />
                          <KV label="Next funding in" value={`${(r.fundIn * 8).toFixed(1)}h`} />
                          <KV label="AI updated" value={ago(r.s.ai_updated)} cls={r.stale ? 'warn' : ''} />
                        </div>
                        {r.blocked && <p className="small neg" style={{ marginBottom: 0 }}>This coin is blocked: the live brain lost money on it in its unseen-data test, so the bot will not open new trades here.</p>}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------- brain page
const reasonLabel = {
  SUPERIOR_TO_CHAMPION: 'Winner', NO_EDGE_AFTER_COSTS: 'No edge after fees', INFERIOR_TO_CHAMPION: 'Worse than champion',
  CRITIC_DEGRADED: 'Forecast worse', ACTION_MONOCULTURE: 'One-sided',
}
const tooltipStyle = { backgroundColor: '#0d1220', border: '1px solid #2f3a4d', borderRadius: 8, fontSize: 12 }

const Brain = ({ ctx }) => {
  const { brain, stats } = ctx
  if (!brain) return <div className="loading">Loading brain status…</div>
  const champ = brain.champion || {}
  const policy = champ.trade_policy || {}
  const trainer = brain.trainer
  const trainerAge = trainer ? now() - trainer.timestamp : null
  const rows = (brain.tournaments || []).map((t, i) => ({
    n: i + 1, date: t.date,
    newBrain: t.challenger_sim_pnl, champion: t.champion_sim_pnl,
    skill: Array.isArray(t.challenger_critic_skill) ? t.challenger_critic_skill[1] : null,
    winner: t.approved ? t.challenger_sim_pnl : null,
  }))
  const reasons = Object.entries(brain.tournament_reasons || {}).map(([k, v]) => ({ name: reasonLabel[k] || k, value: v, win: k === 'SUPERIOR_TO_CHAMPION' }))
  const coinStats = Object.entries(policy.coin_stats || {}).sort((a, b) => b[1].mean_pnl - a[1].mean_pnl)
  const winners = (brain.tournaments || []).filter((t) => t.approved).length
  const best = rows.reduce((b, r) => (r.newBrain != null && (b == null || r.newBrain > b) ? r.newBrain : b), null)

  return (
    <>
      <div className="grid cols-4">
        <StatusTile icon="brain" color={champ.is_v3 ? 'green' : 'amber'} label="Live brain" value={champ.version || '—'} note={champ.is_v3 ? 'Passed the honest after-fees test' : 'Old brain: never passed the new test'} />
        <StatusTile icon="cpu" color={trainerAge != null && trainerAge < 180 ? 'green' : 'red'} label="Trainer" value={trainerAge != null && trainerAge < 180 ? `Step ${trainer.step}` : 'Not reporting'} note={trainer ? `Last heartbeat ${ago(trainer.timestamp)} · ${trainer.host || ''}` : 'Start MASTER_TRAINER on the laptop'} />
        <StatusTile icon="check" color={winners ? 'green' : 'blue'} label="Tournaments" value={`${rows.length} run · ${winners} won`} note="One tournament every 100 training steps" />
        <StatusTile icon="trades" color={best != null && best > 0 ? 'green' : 'amber'} label="Best new brain so far" value={best != null ? `${best.toFixed(4)}% / decision` : '—'} note="Above 0 = profit after fees on unseen data" />
      </div>

      <Card title="Is the AI finding an edge?" help="Every 100 training steps the new brain and the live brain are tested on the newest data neither was trained on, after fees. 0 = the result of not trading."
        desc="The edge appears when the blue line rises above 0 and stays above the grey (live brain). Green dots are winners that were pushed to the VPS.">
        {rows.length === 0 ? <Empty title="No tournaments with the new test yet">The first one runs 100 training steps after MASTER_TRAINER starts.</Empty> : (
          <div className="chart">
            <ResponsiveContainer>
              <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222b3a" vertical={false} />
                <XAxis dataKey="n" stroke="#5b677b" fontSize={12} />
                <YAxis stroke="#5b677b" fontSize={12} width={64} tickFormatter={(v) => `${Number(v).toFixed(2)}%`} />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(n) => `Tournament #${n} · ${rows[n - 1]?.date || ''}`} formatter={(v) => (v == null ? '—' : `${Number(v).toFixed(4)}%`)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0} stroke="#8592a7" />
                <Line type="monotone" dataKey="newBrain" name="New brain" stroke="#5b8cff" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="champion" name="Live brain" stroke="#8592a7" strokeDasharray="5 5" dot={false} />
                <Line type="monotone" dataKey="winner" name="Winner" stroke="#22c55e" strokeWidth={0} dot={{ r: 5, fill: '#22c55e' }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      <div className="grid cols-2">
        <Card title="Forecast skill (1 hour)" help="How well the brain predicts 1-hour price moves compared with guessing 'no move'. Above 0 = real skill.">
          {rows.some((r) => r.skill != null) ? (
            <div className="chart short">
              <ResponsiveContainer>
                <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222b3a" vertical={false} />
                  <XAxis dataKey="n" stroke="#5b677b" fontSize={12} />
                  <YAxis stroke="#5b677b" fontSize={12} width={48} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <ReferenceLine y={0} stroke="#8592a7" />
                  <Line type="monotone" dataKey="skill" name="Skill" stroke="#a78bfa" strokeWidth={2} dot={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty title="Recorded from the next tournament on" />}
        </Card>
        <Card title="Tournament results" help="Why new brains were accepted or rejected. Expect mostly rejections at first; that is the safety check working.">
          {reasons.length ? (
            <div className="chart short">
              <ResponsiveContainer>
                <BarChart data={reasons} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222b3a" vertical={false} />
                  <XAxis dataKey="name" stroke="#5b677b" fontSize={11} />
                  <YAxis stroke="#5b677b" fontSize={12} width={36} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>{reasons.map((r, i) => <Cell key={i} fill={r.win ? '#22c55e' : r.name.startsWith('No edge') ? '#f5a524' : '#f05252'} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty title="No tournaments yet" />}
          <p className="small faint" style={{ marginBottom: 0 }}>{brain.legacy_tournaments} older tournaments are hidden (they used the old, broken scoring).</p>
        </Card>
      </div>

      <div className="grid side">
        <Card title="Live brain: results per coin" help="How the live brain did on each coin in its unseen-data test (trades above its confidence threshold). Coins that lost money over 20+ trades are blocked.">
          {coinStats.length === 0 ? <Empty title="Available once a brain wins the new tournament" /> : (
            <div className="table-wrap">
              <table className="tbl">
                <thead><tr><th>Coin</th><th>Test trades</th><th>Win rate</th><th>Avg per trade (after fees)</th><th>Status</th></tr></thead>
                <tbody>
                  {coinStats.map(([sym, s]) => {
                    const isBlocked = (policy.blocked_symbols || []).includes(sym)
                    return (
                      <tr key={sym}>
                        <td><CoinTag sym={sym} /></td>
                        <td>{s.n}</td>
                        <td>{(s.win_rate * 100).toFixed(0)}%</td>
                        <td className={tone(s.mean_pnl)}>{fmtPP(s.mean_pnl, 3)}</td>
                        <td>{isBlocked ? <span className="badge red">Blocked</span> : s.n < 20 ? <span className="badge amber">Too few trades</span> : <span className="badge green">Allowed</span>}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <Card title="Live brain settings">
          <div className="kv-grid" style={{ gridTemplateColumns: '1fr' }}>
            <KV label="Approved by" value={champ.is_v3 ? 'Honest holdout test' : 'Old test (unreliable)'} cls={champ.is_v3 ? 'pos' : 'warn'} />
            <KV label="Profit per decision (test)" value={champ.is_v3 && champ.holdout_pnl != null ? `${Number(champ.holdout_pnl).toFixed(4)}%` : 'Not measured (old test)'} cls={champ.is_v3 ? tone(champ.holdout_pnl || 0) : 'muted'} />
            <KV label="Minimum confidence to trade" value={policy.min_confidence ?? '—'} help="Learned from the test: the confidence above which trades made money after fees." />
            <KV label="Blocked coins" value={(policy.blocked_symbols || []).length ? policy.blocked_symbols.map((s) => s.replace('USDT', '')).join(', ') : 'None'} cls={(policy.blocked_symbols || []).length ? 'neg' : ''} />
            <KV label="Model file updated" value={ago(champ.model_updated_at)} />
            <KV label="Experiences on VPS" value={(stats.total_experiences || 0).toLocaleString()} />
          </div>
        </Card>
      </div>
    </>
  )
}

// ---------------------------------------------------------------- trades page
const isClose = (t) => !!t.side && !t.side.includes('OPEN') && !t.side.includes('DUST') && typeof t.realized_pnl === 'number'
const tradeDir = (t) => ((t.side || '').includes('LONG') ? 'LONG' : (t.side || '').includes('SHORT') ? 'SHORT' : '')
const priceMove = (t) => {
  if (!isClose(t) || !t.entry_price || !t.price) return null
  return ((t.price - t.entry_price) / t.entry_price) * (tradeDir(t) === 'SHORT' ? -1 : 1)
}
const groupStats = (rows, keyFn) => {
  const g = {}
  rows.forEach((t) => {
    const k = keyFn(t) || 'Not recorded'
    g[k] = g[k] || { key: k, n: 0, wins: 0, net: 0 }
    g[k].n += 1
    g[k].net += t.realized_pnl
    if (t.realized_pnl > 0) g[k].wins += 1
  })
  return Object.values(g).sort((x, y) => y.net - x.net)
}
const GroupTable = ({ title, rows }) => (
  <table className="tbl">
    <thead><tr><th>{title}</th><th>Trades</th><th>Win rate</th><th>Net</th><th>Avg</th></tr></thead>
    <tbody>
      {rows.map((r) => (
        <tr key={r.key}>
          <td>{r.key.endsWith('USDT') ? <CoinTag sym={r.key} /> : r.key}</td>
          <td>{r.n}</td>
          <td>{((r.wins / r.n) * 100).toFixed(0)}%</td>
          <td className={tone(r.net)}>{money(r.net)}</td>
          <td className={tone(r.net)}>{money(r.net / r.n)}</td>
        </tr>
      ))}
    </tbody>
  </table>
)

const Trades = ({ ctx }) => {
  const { state } = ctx
  const [historyData] = usePoll('/api/history', 5000)
  const history = historyData?.history || []
  const [open, setOpen] = useState(null)
  const [symbol, setSymbol] = useState('all')
  const [kind, setKind] = useState('all')
  const closes = history.filter(isClose)
  const net = closes.reduce((a, t) => a + t.realized_pnl, 0)
  const gross = closes.reduce((a, t) => a + (t.gross_pnl || 0), 0)
  const fees = closes.reduce((a, t) => a + (t.fees_paid || 0), 0)
  const wins = closes.filter((t) => t.realized_pnl > 0)
  const losses = closes.filter((t) => t.realized_pnl <= 0)
  const sumWin = wins.reduce((a, t) => a + t.realized_pnl, 0)
  const sumLoss = losses.reduce((a, t) => a + t.realized_pnl, 0)
  const avgWin = wins.length ? sumWin / wins.length : 0
  const avgLoss = losses.length ? sumLoss / losses.length : 0
  const best = closes.reduce((a, t) => (!a || t.realized_pnl > a.realized_pnl ? t : a), null)
  const worst = closes.reduce((a, t) => (!a || t.realized_pnl < a.realized_pnl ? t : a), null)
  const held = closes.filter((t) => t.hold_seconds > 0)
  const avgHold = held.length ? held.reduce((a, t) => a + t.hold_seconds, 0) / held.length : 0
  const winRate = closes.length ? wins.length / closes.length : 0
  // Win rate needed to break even with the current average win / loss
  const breakEven = avgWin - avgLoss > 0 && avgLoss < 0 ? -avgLoss / (avgWin - avgLoss) : 0
  const symbols = [...new Set(history.map((t) => t.symbol))].sort()
  const shown = history.filter((t) => (symbol === 'all' || t.symbol === symbol)
    && (kind === 'all' || (kind === 'closes' && isClose(t)) || (kind === 'opens' && (t.side || '').startsWith('OPEN'))
      || (kind === 'wins' && isClose(t) && t.realized_pnl > 0) || (kind === 'losses' && isClose(t) && t.realized_pnl <= 0)))

  return (
    <>
      <Card title={`Open positions (${openPositions(state.positions).length})`}>
        <PositionCards positions={state.positions} />
      </Card>
      <Card title="Recent results" desc="Based on the last 100 trade records kept by the bot (a partial close counts as its own record).">
        <div className="kv-grid">
          <KV label="Closed records" value={`${closes.length} (${wins.length}W / ${losses.length}L)`} />
          <KV label="Win rate" value={closes.length ? `${(winRate * 100).toFixed(0)}%` : '—'} cls={closes.length && breakEven ? (winRate >= breakEven ? 'pos' : 'neg') : ''}
            help="Share of closed records that made money after fees." />
          <KV label="Break-even win rate" value={breakEven ? `${(breakEven * 100).toFixed(0)}%` : '—'}
            help="With the current average win and average loss, the bot needs at least this win rate just to not lose money. A real win rate below it means the bot is losing." />
          <KV label="Net result" value={money(net)} cls={tone(net)} help="Total profit after fees." />
          <KV label="Gross (before fees)" value={money(gross)} cls={tone(gross)} />
          <KV label="Fees paid" value={money(fees)} cls={fees ? 'neg' : ''} help="Exchange fees for these closed records (entry + exit)." />
          <KV label="Average win" value={money(avgWin)} cls="pos" />
          <KV label="Average loss" value={money(avgLoss)} cls="neg" />
          <KV label="Win / loss size" value={avgLoss ? (avgWin / Math.abs(avgLoss)).toFixed(2) : '—'} help="Average win divided by average loss. Above 1 means wins are bigger than losses." />
          <KV label="Profit factor" value={sumLoss ? (sumWin / Math.abs(sumLoss)).toFixed(2) : '—'} cls={sumLoss ? tone(sumWin / Math.abs(sumLoss) - 1) : ''}
            help="All winnings divided by all losses. Above 1 = profitable overall; 1.5+ is good." />
          <KV label="Best trade" value={best ? `${money(best.realized_pnl)} ${best.symbol.replace('USDT', '')}` : '—'} cls="pos" />
          <KV label="Worst trade" value={worst ? `${money(worst.realized_pnl)} ${worst.symbol.replace('USDT', '')}` : '—'} cls="neg" />
          <KV label="Average hold" value={held.length ? fmtDur(avgHold) : '—'} help="Only trades recorded after this update carry their hold time." />
        </div>
      </Card>
      {closes.length > 0 && (
        <div className="grid cols-3">
          <Card title="By coin"><div className="table-wrap"><GroupTable title="Coin" rows={groupStats(closes, (t) => t.symbol)} /></div></Card>
          <Card title="Long vs short"><div className="table-wrap"><GroupTable title="Side" rows={groupStats(closes, tradeDir)} /></div></Card>
          <Card title="By exit reason" help="Why each trade was closed. Only trades recorded after this update carry a reason.">
            <div className="table-wrap"><GroupTable title="Reason" rows={groupStats(closes, (t) => t.exit_reason)} /></div>
          </Card>
        </div>
      )}
      <Card title="Trade history" desc="Click a row for full details." right={(
        <div className="row-gap">
          <div className="segmented">
            {[['all', 'All'], ['closes', 'Closes'], ['opens', 'Opens'], ['wins', 'Wins'], ['losses', 'Losses']].map(([k, l]) => (
              <button key={k} className={kind === k ? 'active' : ''} onClick={() => { setKind(k); setOpen(null) }}>{l}</button>
            ))}
          </div>
          <select className="select" value={symbol} onChange={(e) => { setSymbol(e.target.value); setOpen(null) }}>
            <option value="all">All coins</option>
            {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      )}>
        {shown.length === 0 ? <Empty title="No trades match" /> : (
          <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Time</th><th>Coin</th><th>Action</th><th>Size</th><th>Entry</th><th>Exit</th><th>Move</th><th>Held</th><th>Fees</th><th>Net (ROI)</th><th>Reason</th></tr></thead>
              <tbody>
                {shown.map((t) => {
                  const side = t.side || ''
                  const closed = isClose(t)
                  const isOpen = side.includes('OPEN')
                  const px = typeof t.price === 'number' ? t.price : 0
                  const notional = t.notional || (t.quantity || 0) * (t.entry_price || px)
                  const mv = priceMove(t)
                  const margin = t.leverage ? notional / t.leverage : null
                  const buying = side.includes('OPEN_LONG') || side.includes('CLOSE_SHORT')
                  const slip = t.mid_price && px ? ((px - t.mid_price) / t.mid_price) * (buying ? 1 : -1) : null
                  const key = `${t.order_id}-${t.timestamp}`
                  return (
                    <React.Fragment key={key}>
                      <tr className={`clickable ${open === key ? 'selected' : ''}`} onClick={() => setOpen(open === key ? null : key)}>
                        <td className="muted">{new Date(t.timestamp * 1000).toLocaleString()}</td>
                        <td><CoinTag sym={t.symbol || ''} /></td>
                        <td><span className={`badge ${isOpen ? 'blue' : side.includes('PARTIAL') ? 'amber' : closed ? (t.realized_pnl > 0 ? 'green' : 'red') : 'grey'}`}>{side.replaceAll('_', ' ')}</span></td>
                        <td>{t.quantity}<div className="faint small">{notional ? money(notional) : ''}</div></td>
                        <td>{t.entry_price ? t.entry_price.toPrecision(6) : isOpen ? px.toPrecision(6) : '—'}</td>
                        <td>{closed ? px.toPrecision(6) : '—'}</td>
                        <td className={tone(mv || 0)}>{mv === null ? '—' : fmtPct(mv, 2)}</td>
                        <td>{t.hold_seconds ? fmtDur(t.hold_seconds) : '—'}</td>
                        <td className="muted">{closed && t.fees_paid ? money(t.fees_paid) : '—'}</td>
                        <td className={tone(t.realized_pnl || 0)}>{isOpen ? 'open' : closed ? `${money(t.realized_pnl)} (${(t.roi_pct || 0) >= 0 ? '+' : ''}${(t.roi_pct || 0).toFixed(1)}%)` : '—'}</td>
                        <td className="small">{t.exit_reason || (isOpen && t.confidence_calibrated != null ? `win chance ${(t.confidence_calibrated * 100).toFixed(0)}%` : '—')}</td>
                      </tr>
                      {open === key && (
                        <tr className="detail-row">
                          <td colSpan="11">
                            <div className="detail-grid">
                              <KV label="Order ID" value={t.order_id || '—'} />
                              <KV label="Order type" value={t.order_type || '—'} help="MAKER LIMIT pays the low fee (0.02%) but may not fill; MARKET fills at once and pays the taker fee (0.05%)." />
                              <KV label="Leverage" value={t.leverage ? `${t.leverage}x` : '—'} />
                              <KV label="Position size" value={notional ? money(notional) : '—'} />
                              <KV label="Margin" value={margin ? money(margin) : '—'} />
                              {closed && <KV label="Gross P&L" value={money(t.gross_pnl || 0)} cls={tone(t.gross_pnl || 0)} />}
                              {closed && <KV label="Fees" value={money(t.fees_paid || 0)} cls="neg" />}
                              {closed && <KV label="Net P&L" value={money(t.realized_pnl)} cls={tone(t.realized_pnl)} />}
                              {closed && <KV label="Return on margin" value={fmtPP(t.roi_pct || 0, 1)} cls={tone(t.roi_pct || 0)} />}
                              {closed && <KV label="Exit reason" value={t.exit_reason || t.reason || '—'} />}
                              {closed && <KV label="Held for" value={fmtDur(t.hold_seconds)} />}
                              {isOpen && <KV label="AI confidence (raw)" value={`${((t.confidence || 0) * 100).toFixed(0)}%`} />}
                              {isOpen && <KV label="Honest win chance" value={t.confidence_calibrated != null ? `${(t.confidence_calibrated * 100).toFixed(0)}%` : '—'} help="Raw confidence mapped to the win rate the brain really achieved in its unseen-data test. Equals raw confidence for an untested brain." />}
                              {isOpen && <KV label="Forecast move" value={t.predicted_move_pct != null ? fmtPP(t.predicted_move_pct) : '—'} cls={tone(t.predicted_move_pct || 0)} help="The AI's expected price move for this trade (1-4h)." />}
                              {isOpen && <KV label="Estimated costs" value={t.est_cost_pct != null ? `${t.est_cost_pct.toFixed(3)}%` : '—'} help="Fees + spread + slippage the bot expected, as % of position size. The forecast must beat this by a margin." />}
                              <KV label="Slippage vs mid" value={slip === null ? '—' : fmtPct(slip, 3)} cls={slip > 0.001 ? 'neg' : ''} help="How much worse (+) or better (-) the fill was than the mid price when the order was sent." />
                              <KV label="Market trend" value={t.regime || '—'} help="The higher-timeframe trend label at the time of the order." />
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}

// ---------------------------------------------------------------- services & logs page
const parseLine = (line) => {
  try {
    const p = JSON.parse(line)
    return { time: p.timestamp ? new Date(p.timestamp).toLocaleTimeString() : '', level: p.level || 'INFO', msg: p.message ?? line, raw: line }
  } catch {
    const level = /\b(error|traceback|exception|critical)\b/i.test(line) ? 'ERROR' : /\bwarn(ing)?\b/i.test(line) ? 'WARNING' : 'INFO'
    return { time: '', level, msg: line, raw: line }
  }
}
const highlight = (text, q) => {
  if (!q) return text
  const i = text.toLowerCase().indexOf(q)
  if (i < 0) return text
  return <>{text.slice(0, i)}<mark>{text.slice(i, i + q.length)}</mark>{text.slice(i + q.length)}</>
}

const EMPTY_LOG = { logs: [], source: null, stale: false }
const LogViewer = ({ service }) => {
  const [stream, setStream] = useState('out')
  const [lines, setLines] = useState(500)
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState('all')
  const [paused, setPaused] = useState(false)
  const [follow, setFollow] = useState(true)
  const [data, setData] = useState(EMPTY_LOG)
  const body = useRef(null)
  const isTrainer = service === TRAINER

  useEffect(() => {
    if (paused) return
    let alive = true
    let hadGood = false
    // A failed or empty read (pm2 busy, log file being rotated, trainer between heartbeats) keeps the
    // last good lines on screen instead of blanking the viewer until the next good read.
    const keepLast = () => setData((d) => ({ ...d, stale: true }))
    const load = async () => {
      try {
        const res = await fetch(`/api/logs/${encodeURIComponent(service)}?stream=${isTrainer ? 'out' : stream}&lines=${lines}`)
        if (!res.ok) throw new Error(res.status)
        const json = await res.json()
        if (!alive) return
        const logs = json.logs || []
        if (hadGood && (json.ok === false || logs.length === 0)) return keepLast()
        if (json.ok !== false && logs.length) hadGood = true
        setData({ logs, source: json.source || null, stale: false })
      } catch { if (alive && hadGood) keepLast() }
    }
    load()
    const t = setInterval(load, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [service, stream, lines, paused, isTrainer])
  useEffect(() => { if (follow && body.current) body.current.scrollTop = body.current.scrollHeight }, [data, follow])

  const parsed = useMemo(() => data.logs.map(parseLine), [data])
  const q = query.trim().toLowerCase()
  const shown = parsed.filter((l) => (level === 'all' || (level === 'errors' ? (l.level === 'ERROR' || l.level === 'CRITICAL') : (l.level === 'WARNING' || l.level === 'ERROR' || l.level === 'CRITICAL'))) && (!q || l.raw.toLowerCase().includes(q)))
  const nErr = parsed.filter((l) => l.level === 'ERROR' || l.level === 'CRITICAL').length
  const nWarn = parsed.filter((l) => l.level === 'WARNING').length

  return (
    <div className="terminal">
      <div className="terminal-bar">
        <span className="title">{service}</span>
        <span className="stats">{shown.length} lines · <span className={nErr ? 'neg' : ''}>{nErr} errors</span> · <span className={nWarn ? 'warn' : ''}>{nWarn} warnings</span></span>
        <div className="right">
          {!isTrainer && (
            <div className="segmented">
              <button className={stream === 'out' ? 'active' : ''} onClick={() => { setStream('out'); setData(EMPTY_LOG) }}>Output</button>
              <button className={stream === 'err' ? 'active' : ''} onClick={() => { setStream('err'); setData(EMPTY_LOG) }}>Error log</button>
            </div>
          )}
          <div className="segmented">
            {[['all', 'All'], ['warn', 'Warnings+'], ['errors', 'Errors']].map(([k, l]) => <button key={k} className={level === k ? 'active' : ''} onClick={() => setLevel(k)}>{l}</button>)}
          </div>
          <input className="input" style={{ width: 190 }} placeholder="Search…" value={query} onChange={(e) => setQuery(e.target.value)} />
          <select className="select" value={lines} onChange={(e) => { setLines(Number(e.target.value)); setData(EMPTY_LOG) }}>
            {[200, 500, 1000, 3000].map((n) => <option key={n} value={n}>Last {n}</option>)}
          </select>
          <button className="btn sm" onClick={() => setPaused(!paused)}>{paused ? '▶ Resume' : '⏸ Pause'}</button>
          <button className="btn sm" onClick={() => { setFollow(true); if (body.current) body.current.scrollTop = body.current.scrollHeight }}>{follow ? 'Following' : '↓ Follow'}</button>
          {!isTrainer && <button className="btn sm" onClick={() => window.open(`/api/logs/download/${encodeURIComponent(service)}?stream=${stream}`, '_blank')}>Download</button>}
        </div>
      </div>
      <div className="terminal-body" ref={body} onScroll={() => { const el = body.current; if (el) setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 40) }}>
        {shown.length === 0 && <div className="log-line faint">{data.logs.length ? 'No lines match.' : 'Loading…'}</div>}
        {shown.map((l, i) => (
          <div key={i} className={`log-line ${l.level}`}>
            {l.time && <span className="t">{l.time}</span>}{l.level !== 'INFO' && <b>{l.level} </b>}{highlight(String(l.msg), q)}
          </div>
        ))}
      </div>
      <div className="terminal-foot">{data.source ? `source: ${data.source}` : ' '}{paused ? ' · paused' : ' · refreshing every 3s'}{data.stale && !paused ? ' · last read failed, showing previous lines' : ''}</div>
    </div>
  )
}

const Services = ({ ctx }) => {
  const { services } = ctx
  const list = services?.services || []
  const [selected, setSelected] = useState(null)
  const current = selected || (list.find((s) => s.name === 'trading-bot') || list[0])?.name

  const restart = async (name) => {
    if (await confirmAction(`Restart ${name}?`, 'The service stops for a few seconds and starts again.', 'warning', 'Restart', '#f5a524')) {
      postAction(`/api/system/pm2/restart/${encodeURIComponent(name)}`)
    }
  }
  const describe = {
    'market-collector': 'Prices, order book, funding, depth, Coinbase/DVOL/calendar',
    'trading-bot': 'Runs the AI and places trades',
    'whale-tracker': 'Large trades and order-book walls',
    'liquidation-hunter': 'Liquidations on Binance, Bybit, OKX',
    'news-macro': 'News sentiment and world markets',
    'statarb-collector': 'Price gaps between exchanges',
    [TRAINER]: 'Trains new brains (runs on the laptop)',
  }

  return (
    <div className="svc-layout">
      <Card title="Services" help="Every process managed by pm2 on the VPS, plus the trainer on the laptop.">
        {!services?.pm2 && <p className="small warn">pm2 not found here: showing local log files.</p>}
        <div className="svc-list">
          {list.map((s) => (
            <div key={s.name} className={`svc ${current === s.name ? 'active' : ''}`} onClick={() => setSelected(s.name)}>
              <Dot color={statusColor(s.status)} />
              <div style={{ minWidth: 0 }}>
                <div className="svc-name">{s.name}</div>
                <div className="svc-meta">
                  {s.remote
                    ? (s.status === 'online' ? `seen ${s.last_seen_s}s ago` : s.status.replace('-', ' '))
                    : `${s.status} · up ${duration(s.uptime_since)}${s.cpu != null ? ` · ${s.cpu}% CPU` : ''}${s.memory_mb ? ` · ${s.memory_mb} MB` : ''}`}
                </div>
                {describe[s.name] && <div className="svc-meta faint">{describe[s.name]}</div>}
                {s.restarts > 10 && <div className="svc-err">{s.restarts} restarts: check its error log</div>}
              </div>
              {!s.remote && services?.pm2 && !s.name.includes('dashboard') && (
                <button className="btn sm ghost" title="Restart" onClick={(e) => { e.stopPropagation(); restart(s.name) }}>↻</button>
              )}
            </div>
          ))}
        </div>
      </Card>
      {current ? <LogViewer key={current} service={current} /> : <Card><Empty title="No services found" /></Card>}
    </div>
  )
}

// ---------------------------------------------------------------- app shell
const PAGES = [
  { id: 'overview', label: 'Overview', icon: 'overview', title: 'Overview', sub: 'Everything important at a glance' },
  { id: 'coins', label: 'Coins', icon: 'coins', title: 'Coins', sub: 'What the AI sees and thinks for every coin' },
  { id: 'brain', label: 'Brain & Training', icon: 'brain', title: 'Brain & Training', sub: 'Is the AI learning an edge?' },
  { id: 'trades', label: 'Trades', icon: 'trades', title: 'Trades', sub: 'Open positions and trade history' },
  { id: 'services', label: 'Services & Logs', icon: 'services', title: 'Services & Logs', sub: 'Health and live logs of every service' },
]

function App() {
  const initial = (typeof window !== 'undefined' && window.location.hash.slice(1)) || 'overview'
  const [page, setPageState] = useState(PAGES.some((p) => p.id === initial) ? initial : 'overview')
  const setPage = (id) => { setPageState(id); window.location.hash = id }
  useEffect(() => {
    const onHash = () => { const h = window.location.hash.slice(1); if (PAGES.some((p) => p.id === h)) setPageState(h) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const [stateData, stateErr] = usePoll('/api/state', 2000)
  const [statsData] = usePoll('/api/system_stats', 5000)
  const [brain] = usePoll('/api/brain', 15000)
  const [services] = usePoll('/api/services', 5000)

  const state = stateData || { equity: 0, positions: [], market_states: {} }
  const stats = statsData || { latest_headlines: [] }
  const killActive = !!stats.kill_switch_active
  const blocked = useMemo(() => new Set(brain?.champion?.trade_policy?.blocked_symbols || []), [brain])
  const markets = useMemo(() => Object.entries(stateData?.market_states || {}).sort(([a], [b]) => a.localeCompare(b)), [stateData])
  const svc = services?.services || []
  const down = svc.filter((s) => !s.remote && s.status !== 'online' && s.status !== 'file').length
  const trainerUp = svc.find((s) => s.remote)?.status === 'online'
  const nOpen = openPositions(state.positions).length
  const meta = PAGES.find((p) => p.id === page)

  const toggleKill = async () => {
    if (killActive) {
      if (await confirmAction('Resume trading?', 'The bot will open new trades again with the current live brain.', 'warning', 'Resume trading', '#22c55e')) postAction('/api/system/pm2/resume')
    } else if (await confirmAction('Turn on the kill switch?', 'The bot stops opening new trades immediately. Data collection and training continue.', 'error', 'Stop trading', '#f05252')) {
      postAction('/api/system/pm2/stop')
    }
  }
  const restartAll = async () => {
    if (await confirmAction('Restart all services?', 'Every pm2 service except the dashboard restarts. The kill switch stays as it is.', 'warning', 'Restart all', '#f5a524')) postAction('/api/system/pm2/restart')
  }

  const isV3 = !!brain?.champion?.is_v3
  const confLabel = isV3 ? 'Honest win chance' : 'Confidence (unproven)'
  const ctx = { stats, brain, services, state, markets, blocked, killActive, go: setPage, isV3, confLabel }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">AI</div>
          <div>
            <div className="brand-name">Trading Terminal</div>
            <div className="brand-sub">Binance futures · AI</div>
          </div>
        </div>
        <nav className="nav">
          {PAGES.map((p) => (
            <button key={p.id} className={`nav-item ${page === p.id ? 'active' : ''}`} onClick={() => setPage(p.id)}>
              <Icon name={p.icon} /> {p.label}
              {p.id === 'trades' && nOpen > 0 && <span className="nav-count">{nOpen}</span>}
              {p.id === 'services' && down > 0 && <span className="nav-count alert">{down}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="row"><span>Trading</span><span className={killActive ? 'neg' : 'pos'}>{killActive ? 'Paused' : 'Active'}</span></div>
          <div className="row"><span>Trainer</span><span className={trainerUp ? 'pos' : 'neg'}>{trainerUp ? 'Running' : 'Offline'}</span></div>
          <div className="row"><span>Server</span><span className={stateErr ? 'neg' : 'pos'}>{stateErr ? 'Unreachable' : 'Connected'}</span></div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div>
            <h1 className="page-title">{meta.title}</h1>
            <div className="page-sub">{meta.sub}</div>
          </div>
          <div className="topbar-right">
            <span className="pill"><Dot color={killActive ? 'red' : 'green'} />{killActive ? 'Trading paused' : 'Trading active'}</span>
            <span className="pill"><Dot color={down ? 'red' : 'green'} />{down ? `${down} service(s) down` : 'Services OK'}</span>
            <div className="equity">
              <div className="label">Equity</div>
              <div className="value">{money(state.equity || 0)}</div>
            </div>
            <button className={`btn ${killActive ? 'green' : 'red'}`} onClick={toggleKill}>{killActive ? '▶ Resume trading' : '■ Kill switch'}</button>
            <button className="btn" onClick={restartAll} title="Restart all services">↻ Restart all</button>
          </div>
        </header>

        <main className="content">
          {stateErr && <div className="banner red"><Icon name="alert" /><span className="grow">Cannot reach the dashboard server. Data shown may be out of date.</span></div>}
          {killActive && page !== 'overview' && <div className="banner red"><Icon name="power" /><span className="grow">Kill switch is ON: the bot is not opening new trades. Data collection and training continue.</span></div>}
          {page === 'overview' && <Overview ctx={ctx} />}
          {page === 'coins' && <Coins ctx={ctx} />}
          {page === 'brain' && <Brain ctx={ctx} />}
          {page === 'trades' && <Trades ctx={ctx} />}
          {page === 'services' && <Services ctx={ctx} />}
        </main>
      </div>
    </div>
  )
}

export default App
