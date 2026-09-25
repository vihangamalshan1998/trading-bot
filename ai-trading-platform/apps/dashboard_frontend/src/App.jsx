import React, { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Swal from 'sweetalert2'
import './App.css'

// --- New Feature: Neural Brain Map ---
const NeuralBrainMap = ({ stateVector }) => {
  if (!stateVector || stateVector.length === 0) return <div className="brain-map-empty">Awaiting Neural Signals...</div>;
  
  return (
    <div className="neural-brain-map">
      <div className="brain-grid">
        {stateVector.map((val, idx) => {
          // Normalize intensity for color
          let intensity = Math.min(Math.abs(val), 1.0);
          if (idx >= 2 && idx <= 26) {
             // Market features might be scaled differently, we just cap for visual effect
             intensity = Math.min(Math.abs(val) / 5.0, 1.0);
          } else if (idx >= 30) {
             // Macro slots
             intensity = Math.min(Math.abs(val) * 2, 1.0);
          }
          
          let colorClass = "neutral";
          if (val > 0.01) colorClass = "positive";
          if (val < -0.01) colorClass = "negative";
          if (Math.abs(val) < 0.000001) colorClass = "empty";
          
          return (
            <div key={idx} className={`brain-node ${colorClass}`} style={{ opacity: colorClass !== 'empty' ? 0.3 + (intensity * 0.7) : 0.2 }} title={`Slot ${idx}: ${val}`}></div>
          )
        })}
      </div>
      <div className="brain-legend">
        <span>Portfolio (0-1)</span>
        <span>Market Data (2-26)</span>
        <span>Position (27)</span>
        <span>Alt-Data (28-40)</span>
      </div>
    </div>
  );
};

// --- New Feature: Funding Rate Speedometer ---
const FundingGauge = ({ rate }) => {
  // Normalize rate between -0.001 and +0.001 for gauge rotation
  const clampedRate = Math.max(-0.001, Math.min(0.001, rate));
  // Map -0.001 to -90deg, 0 to 0deg, +0.001 to +90deg
  const rotation = (clampedRate / 0.001) * 90;
  
  let status = "NORMAL";
  let color = "#27c93f"; // Green
  
  if (rate > 0.0001) {
     status = "OVER-LEVERAGED (LONG)";
     color = "#ff5f56"; // Red
  } else if (rate < -0.0001) {
     status = "OVER-LEVERAGED (SHORT)";
     color = "#ff5f56"; // Red
  } else if (rate === 0.0001) {
     status = "BASELINE";
     color = "#ffbd2e"; // Yellow
  }

  return (
    <div className="funding-gauge-container">
      <div className="gauge-label">Live Funding Rate</div>
      <div className="gauge">
        <div className="gauge-bg"></div>
        <div className="gauge-needle" style={{ transform: `rotate(${rotation}deg)` }}></div>
      </div>
      <div className="gauge-value" style={{ color: color }}>{(rate * 100).toFixed(4)}%</div>
      <div className="gauge-status">{status}</div>
    </div>
  );
};


const LogViewer = ({ botName, title, activeTab }) => {
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 30;
    setAutoScroll(isAtBottom);
  };

  useEffect(() => {
    if (activeTab !== 'logs') return;
    if (!autoScroll) return; 
    
    const fetchLogs = async () => {
      try {
        const res = await fetch(`/api/logs/${botName}`);
        const json = await res.json();
        setLogs(json.logs || []);
      } catch (err) {
        console.error("Failed to fetch logs:", err);
      }
    };
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [botName, activeTab, autoScroll]);

  useEffect(() => {
    if (autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [logs, autoScroll]);

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <span className="terminal-title">{title}</span>
        <button 
          onClick={() => setAutoScroll(!autoScroll)}
          style={{ 
            background: 'transparent', 
            border: '1px solid #555', 
            color: autoScroll ? '#27c93f' : '#ffbd2e', 
            borderRadius: '4px', 
            fontSize: '10px', 
            padding: '2px 8px', 
            cursor: 'pointer',
            marginLeft: '10px'
          }}
        >
          {autoScroll ? '🟢 Auto-Scroll ON' : '🟡 Auto-Scroll OFF'}
        </button>
        <button 
          onClick={() => window.open(`/api/logs/download/${botName}`, '_blank')}
          style={{ 
            background: 'transparent', 
            border: '1px solid #8b9bb4', 
            color: '#8b9bb4', 
            borderRadius: '4px', 
            fontSize: '10px', 
            padding: '2px 8px', 
            cursor: 'pointer',
            marginLeft: '10px'
          }}
        >
          📥 Download Full Log
        </button>
        <div className="terminal-dots" style={{ marginLeft: 'auto' }}>
          <span className="dot red"></span>
          <span className="dot yellow"></span>
          <span className="dot green"></span>
        </div>
      </div>
      <div className="terminal-body" ref={containerRef} onScroll={handleScroll}>
        {logs.length === 0 ? <div className="log-line">Loading logs...</div> : null}
        {logs.map((log, idx) => {
          try {
            const parsed = JSON.parse(log);
            let color = '#39ff14';
            if (parsed.level === 'ERROR' || parsed.level === 'CRITICAL') color = '#ff5f56';
            if (parsed.level === 'WARNING') color = '#ffbd2e';
            
            const time = new Date(parsed.timestamp).toLocaleTimeString();
            
            return (
              <div key={idx} className="log-line" style={{ color }}>
                <span style={{color: '#8b9bb4'}}>[{time}]</span> [{parsed.level}] {parsed.message}
              </div>
            );
          } catch (e) {
            return <div key={idx} className="log-line">{log}</div>
          }
        })}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('live') // 'live', 'training', 'history', 'logs'
  const [activeSubTab, setActiveSubTab] = useState('trading_bot') // 'market_collector', 'ai_trainer', 'trading_bot'
  const [expandedTradeIdx, setExpandedTradeIdx] = useState(null);
  
  const [data, setData] = useState({
    equity: 0.0,
    positions: [],
    market_states: {}
  });

  const [systemStats, setSystemStats] = useState({
    news_count: 0,
    latest_sentiment: 0.0,
    latest_regime: 0.0,
    model_update_count: 0,
    last_model_update_time: null,
    latest_headlines: []
  });

  const [trainingMetrics, setTrainingMetrics] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);

  useEffect(() => {
    const fetchLiveState = async () => {
      try {
        const res = await fetch("/api/state");
        const json = await res.json();
        setData(json);
      } catch (err) {
        console.error("Failed to fetch live state:", err);
      }
    };
    
    const fetchSystemStats = async () => {
      try {
        const res = await fetch("/api/system_stats");
        const json = await res.json();
        setSystemStats(json);
      } catch (err) {
        console.error("Failed to fetch system stats:", err);
      }
    };
    
    fetchLiveState();
    fetchSystemStats();
    const interval = setInterval(() => {
      fetchLiveState();
      fetchSystemStats();
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (activeTab !== 'history' && activeTab !== 'live') return;
    
    const fetchHistory = async () => {
      try {
        const res = await fetch("/api/history");
        const json = await res.json();
        setTradeHistory(json.history || []);
      } catch (err) {
        console.error("Failed to fetch history:", err);
      }
    };
    
    fetchHistory();
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== 'training') return;
    
    const fetchMetrics = async () => {
      try {
        const res = await fetch("/api/training");
        const json = await res.json();
        const formatted = json.metrics.map((m, idx) => ({
          ...m,
          name: `Step ${idx}`,
        }));
        setTrainingMetrics(formatted);
      } catch (err) {
        console.error("Failed to fetch training metrics:", err);
      }
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, [activeTab]);

  const handleSystemAction = (action) => {
    const isStop = action === 'stop';
    Swal.fire({
      title: isStop ? 'EMERGENCY STOP?' : 'Restart System?',
      text: isStop 
        ? "This will instantly kill all trading bots and background processes! Are you sure?"
        : "This will restart all PM2 processes. Trading will briefly pause.",
      icon: isStop ? 'error' : 'warning',
      showCancelButton: true,
      confirmButtonColor: isStop ? '#d33' : '#f39c12',
      cancelButtonColor: '#3085d6',
      confirmButtonText: isStop ? 'YES, KILL EVERYTHING!' : 'Yes, restart it!'
    }).then(async (result) => {
      if (result.isConfirmed) {
        try {
          const res = await fetch(`/api/system/pm2/${action}`, { method: 'POST' });
          const json = await res.json();
          if (json.status === 'success') {
            Swal.fire('Success!', json.message, 'success');
          } else {
            Swal.fire('Error!', json.message, 'error');
          }
        } catch (err) {
          Swal.fire('Error!', 'Failed to communicate with backend.', 'error');
        }
      }
    });
  };

  const toggleAccordion = (idx) => {
    if (expandedTradeIdx === idx) {
      setExpandedTradeIdx(null);
    } else {
      setExpandedTradeIdx(idx);
    }
  }

  // --- Calculate Maker Fee Analytics ---
  const historyVolume = tradeHistory.reduce((acc, trade) => {
      const px = trade.entry_price || trade.price || 0;
      return acc + (trade.quantity * px);
  }, 0);
  
  const openVolume = (data.positions || []).reduce((acc, pos) => {
      if (!pos.quantity || pos.quantity === 0) return acc;
      const px = pos.entryPrice || pos.entry_price || 0;
      return acc + (Math.abs(pos.quantity) * px);
  }, 0);

  const totalVolume = historyVolume + openVolume;
  // Taker fee: 0.05%, Maker fee: 0.02%. Savings = 0.03%
  const savedFees = totalVolume * 0.0003;

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <div className="header-title">
          <div className="pulse-indicator"></div>
          <h1>AI Trading Terminal <span className="version-badge">v5.0</span></h1>
        </div>
        
        <div className="nav-tabs">
          <button className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>
            Live Market & Macro
          </button>
          <button className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`} onClick={() => setActiveTab('training')}>
            Brain & Training
          </button>
          <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
            Positions & History
          </button>
          <button className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
            Live Logs
          </button>
        </div>

        <div className="system-controls" style={{ display: 'flex', gap: '10px', marginLeft: 'auto', marginRight: '20px' }}>
          <button 
            onClick={() => handleSystemAction('stop')}
            style={{ backgroundColor: 'rgba(255, 71, 87, 0.2)', color: '#ff4757', border: '1px solid #ff4757', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', transition: 'all 0.3s' }}
          >
            🛑 KILL SWITCH
          </button>
          <button 
            onClick={() => handleSystemAction('restart')}
            style={{ backgroundColor: 'rgba(255, 165, 2, 0.2)', color: '#ffa502', border: '1px solid #ffa502', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', transition: 'all 0.3s' }}
          >
            🔄 Restart Bots
          </button>
        </div>
        
        <div className="equity-display">
          <span>Total Equity</span>
          <h2 className={data.equity > 10000 ? 'profit' : data.equity < 10000 ? 'loss' : ''}>
            ${data.equity.toFixed(2)}
          </h2>
        </div>
      </header>

      <main className="dashboard-grid">
        {activeTab === 'live' && (
          <>
            <div className="top-dashboard-row" style={{ gridColumn: '1 / -1' }}>
              <section className="glass-panel stat-panel flex-1">
                <h3>Macro Sentiment & News</h3>
                <div className="stat-grid">
                  <div className="stat-card">
                    <span className="stat-label">News Analyzed</span>
                    <span className="stat-value highlight">{systemStats.news_count}</span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Gemini Sentiment</span>
                    <span className={`stat-value ${systemStats.latest_sentiment > 0 ? 'profit' : systemStats.latest_sentiment < 0 ? 'loss' : ''}`}>
                      {systemStats.latest_sentiment.toFixed(2)}
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Market Regime</span>
                    <span className={`stat-value ${systemStats.latest_regime > 0 ? 'profit' : systemStats.latest_regime < 0 ? 'loss' : ''}`}>
                      {systemStats.latest_regime > 0 ? 'Bullish' : systemStats.latest_regime < 0 ? 'Bearish' : 'Neutral'}
                    </span>
                  </div>
                </div>

                {systemStats.latest_headlines && systemStats.latest_headlines.length > 0 && (
                  <div className="news-headlines">
                    <h4>Latest Headlines</h4>
                    <ul>
                      {systemStats.latest_headlines.map((headline, idx) => {
                        const title = typeof headline === 'string' ? headline : headline.title;
                        const url = typeof headline === 'string' ? null : headline.url;
                        return (
                          <li key={idx}>
                            📰 {url ? <a href={url} target="_blank" rel="noopener noreferrer" style={{color: '#00f2fe', textDecoration: 'none'}} onMouseOver={(e) => e.target.style.textDecoration='underline'} onMouseOut={(e) => e.target.style.textDecoration='none'}>{title}</a> : title}
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )}
              </section>

              <div className="flex-1" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <section className="glass-panel stat-panel" style={{ flex: '0 0 auto' }}>
                   <h3 style={{ marginBottom: '1rem' }}>Maker Fee Analytics</h3>
                   <div className="fee-saved-container" style={{ padding: '1rem' }}>
                      <div className="fee-saved-amount profit" style={{ fontSize: '2rem' }}>${savedFees.toFixed(2)}</div>
                      <div className="fee-saved-label" style={{ fontSize: '0.9rem' }}>Capital Saved via AI Limit Orders</div>
                      <div className="fee-saved-subtitle" style={{ fontSize: '0.7rem' }}>Bypassed Taker Fees (0.05% → 0.02%)</div>
                   </div>
                </section>
                
                <section className="glass-panel stat-panel" style={{ flex: '1 1 auto', overflowY: 'auto', maxHeight: '500px' }}>
                  <h3>Live Open Positions</h3>
                  {(!data.positions || data.positions.length === 0) ? (
                    <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                      <p>No active positions.</p>
                      <small>AI is scanning for optimal entry points...</small>
                    </div>
                  ) : (
                    <div className="position-list">
                      {data.positions.map((pos, idx) => {
                        if (pos.quantity === 0) return null;
                        const isLong = pos.side.toUpperCase().includes("LONG");
                        const pnlClass = pos.pnl >= 0 ? 'profit' : 'loss';
                        return (
                          <div key={idx} className={`position-card ${isLong ? 'long' : 'short'}`}>
                            <div className="pos-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                              <span className="symbol" style={{ fontWeight: 'bold' }}>{pos.symbol} <span style={{ fontSize: '0.8rem', color: '#888' }}>{pos.leverage}x</span></span>
                              <span className="side">{pos.side}</span>
                            </div>
                            <div className="pos-details" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.85rem' }}>
                              <div style={{ display: 'flex', flexDirection: 'column' }}>
                                <span style={{ color: '#888' }}>Size / Margin</span>
                                <span>{Math.abs(pos.quantity).toFixed(4)} / ${pos.margin ? pos.margin.toFixed(2) : ((Math.abs(pos.quantity) * (pos.entryPrice || pos.entry_price || 0)) / (pos.leverage || 1)).toFixed(2)}</span>
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                <span style={{ color: '#888' }}>PnL</span>
                                <span className={pnlClass} style={{ fontWeight: 'bold' }}>
                                  {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)} USDT
                                </span>
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column' }}>
                                <span style={{ color: '#888' }}>Entry Price</span>
                                <span>${pos.entryPrice ? pos.entryPrice.toFixed(4) : pos.entry_price ? pos.entry_price.toFixed(4) : '0.00'}</span>
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                <span style={{ color: '#888' }}>Liq Price</span>
                                <span style={{ color: '#ffbd2e' }}>${pos.liquidationPrice ? pos.liquidationPrice.toFixed(4) : pos.liquidation_price ? pos.liquidation_price.toFixed(4) : '0.00'}</span>
                              </div>
                            </div>
                            <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)', fontSize: '0.75rem', color: '#a0a0a0', textAlign: 'center' }}>
                               Executed via AI Limit Order
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </section>
              </div>
            </div>

            <section className="glass-panel market-panel full-width">
              <h3>Live Market States & AI Brain Maps</h3>
              <div className="market-list">
                {Object.keys(data.market_states).length === 0 ? (
                  <p className="empty-text">Awaiting market data from Binance...</p>
                ) : (
                  Object.entries(data.market_states).map(([sym, state], idx) => {
                    const latestFunding = state.funding_rate || 0;
                    
                    return (
                    <div key={idx} className="market-card">
                      <div className="market-card-header">
                        <span className="symbol">{sym}</span>
                        <span className={`trend ${state.trend ? state.trend.toLowerCase() : ''}`}>{state.trend || 'N/A'}</span>
                      </div>
                      
                      <div className="market-card-metrics-row">
                        <div className="market-metric">
                          <span className="label">Price</span>
                          <span className="value">${state.mid_price ? state.mid_price.toFixed(2) : '0.00'}</span>
                        </div>
                        <div className="market-metric">
                          <span className="label">Spread</span>
                          <span className="value">
                            {state.spread && state.mid_price 
                              ? ((state.spread / state.mid_price) * 10000).toFixed(1) 
                              : '0.0'} bps
                          </span>
                        </div>
                        <div className="market-metric">
                          <FundingGauge rate={latestFunding} />
                        </div>
                      </div>
                      
                      <div className="ai-brain-section">
                        <div className="ai-brain-header">
                          <span>AI Decision</span>
                          <span className={`ai-decision ${state.ai_predicted_side === 'LONG' ? 'profit' : state.ai_predicted_side === 'SHORT' ? 'loss' : 'neutral'}`}>
                            {state.ai_predicted_side || 'WAITING'}
                          </span>
                        </div>
                        <div className="ai-brain-metric">
                          <span>Confidence:</span>
                          <div className="progress-bar-container">
                            <div className="progress-bar" style={{ width: `${(state.ai_confidence || 0) * 100}%` }}></div>
                          </div>
                          <span>{((state.ai_confidence || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="ai-brain-metric">
                          <span>Target Size:</span>
                          <div className="progress-bar-container">
                            <div className="progress-bar alloc" style={{ width: `${(state.ai_target_size || 0) * 100}%` }}></div>
                          </div>
                          <span>{((state.ai_target_size || 0) * 100).toFixed(1)}%</span>
                        </div>
                        
                      </div>
                    </div>
                  )})
                )}
              </div>
            </section>
          </>
        )}
        
        {activeTab === 'history' && (
          <section className="glass-panel full-width">
            <h3>Active AI Positions</h3>
            <div style={{marginBottom: '2rem'}}>
              {(!data.positions || data.positions.length === 0) ? (
                <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                  <p>No active positions.</p>
                  <small>AI is scanning for optimal entry points...</small>
                </div>
              ) : (
                <div className="position-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.25rem' }}>
                  {data.positions.map((pos, idx) => {
                    if (pos.quantity === 0) return null;
                    const isLong = pos.side.toUpperCase().includes("LONG");
                    const pnlClass = pos.pnl >= 0 ? 'profit' : 'loss';
                    return (
                      <div key={idx} className={`position-card ${isLong ? 'long' : 'short'}`}>
                        <div className="pos-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <span className="symbol" style={{ fontWeight: 'bold' }}>{pos.symbol} <span style={{ fontSize: '0.8rem', color: '#888' }}>{pos.leverage}x</span></span>
                          <span className="side">{pos.side}</span>
                        </div>
                        <div className="pos-details" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.85rem' }}>
                          <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ color: '#888' }}>Size / Margin</span>
                            <span>{Math.abs(pos.quantity).toFixed(4)} / ${pos.margin ? pos.margin.toFixed(2) : ((Math.abs(pos.quantity) * (pos.entryPrice || pos.entry_price || 0)) / (pos.leverage || 1)).toFixed(2)}</span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                            <span style={{ color: '#888' }}>PnL</span>
                            <span className={pnlClass} style={{ fontWeight: 'bold' }}>
                              {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)} USDT
                            </span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ color: '#888' }}>Entry Price</span>
                            <span>${pos.entryPrice ? pos.entryPrice.toFixed(4) : pos.entry_price ? pos.entry_price.toFixed(4) : '0.00'}</span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                            <span style={{ color: '#888' }}>Liq Price</span>
                            <span style={{ color: '#ffbd2e' }}>${pos.liquidationPrice ? pos.liquidationPrice.toFixed(4) : pos.liquidation_price ? pos.liquidation_price.toFixed(4) : '0.00'}</span>
                          </div>
                        </div>
                        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)', fontSize: '0.75rem', color: '#a0a0a0', textAlign: 'center' }}>
                           Executed via AI Limit Order
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <h3>AI Trade History (Last 100)</h3>
            <div className="table-responsive accordion-table">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>TIME</th>
                    <th>SYMBOL</th>
                    <th>ACTION</th>
                    <th>SIZE</th>
                    <th>ENTRY PRICE</th>
                    <th>CLOSE PRICE</th>
                    <th>PNL (ROI)</th>
                    <th>DETAILS</th>
                  </tr>
                </thead>
                <tbody>
                  {tradeHistory.length === 0 ? (
                    <tr>
                      <td colSpan="8" className="empty-text">No trade history found.</td>
                    </tr>
                  ) : (
                    tradeHistory.map((trade, idx) => {
                      const isExpanded = expandedTradeIdx === idx;
                      return (
                        <React.Fragment key={idx}>
                          <tr className={`accordion-row ${isExpanded ? 'expanded' : ''}`} onClick={() => toggleAccordion(idx)}>
                            <td>{new Date(trade.timestamp * 1000).toLocaleString()}</td>
                            <td className="symbol">{trade.symbol}</td>
                            <td className={trade.side.includes("LONG") ? "long" : "short"}>{trade.side}</td>
                            <td>{trade.quantity} {trade.symbol.replace('USDT', '')}</td>
                            <td>
                              {trade.entry_price 
                                ? `$${trade.entry_price.toFixed(4)}` 
                                : (trade.side.includes("OPEN") ? `$${trade.price.toFixed(4)}` : '-')}
                            </td>
                            <td>{trade.side.includes("CLOSE") ? `$${trade.price.toFixed(4)}` : '-'}</td>
                            <td>
                              {trade.realized_pnl !== undefined ? (
                                <span className={trade.realized_pnl > 0 ? "profit" : "loss"}>
                                  ${trade.realized_pnl.toFixed(2)} ({trade.roi_pct > 0 ? '+' : ''}{trade.roi_pct.toFixed(2)}%)
                                </span>
                              ) : (
                                '-'
                              )}
                            </td>
                            <td>
                               <button className="expand-btn">{isExpanded ? '▲' : '▼'}</button>
                            </td>
                          </tr>
                          {isExpanded && (
                            <tr className="accordion-details">
                              <td colSpan="8">
                                <div className="details-container">
                                  <div className="detail-col">
                                    <strong>Trade Metadata</strong>
                                    <p>Order ID: <code>{trade.order_id || 'N/A'}</code></p>
                                    <p>AI Confidence: <span className="highlight">{(trade.confidence * 100).toFixed(1)}%</span></p>
                                  </div>
                                  <div className="detail-col brain-col">
                                    <strong>AI Status</strong>
                                    <p>State captured at {new Date(trade.timestamp * 1000).toLocaleTimeString()}</p>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {activeTab === 'training' && (
          <>
            <section className="glass-panel stat-panel full-width">
              <h3>AI Brain Status</h3>
              <div className="stat-grid">
                <div className="stat-card">
                  <span className="stat-label">Total Brain Saves</span>
                  <span className="stat-value highlight">{systemStats.model_update_count}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Last Saved</span>
                  <span className="stat-value">
                    {systemStats.last_model_update_time 
                      ? new Date(systemStats.last_model_update_time * 1000).toLocaleTimeString() 
                      : 'Never'}
                  </span>
                </div>
              </div>
            </section>

            <section className="glass-panel full-width">
              <h3>PPO Training Progress (Live Loss Metrics)</h3>
              
              {trainingMetrics.length === 0 ? (
                <div className="empty-state">
                  <div className="spinner"></div>
                  <p>Waiting for training data...</p>
                  <small>Make sure `ppo.py` is running and Redis is active.</small>
                </div>
              ) : (
                <div className="chart-container" style={{ width: '100%', height: 400 }}>
                  <ResponsiveContainer>
                    <LineChart
                      data={trainingMetrics}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                      <XAxis dataKey="name" stroke="#a0a0a0" tick={{fill: '#a0a0a0'}} />
                      <YAxis 
                        stroke="#a0a0a0" 
                        domain={['auto', 'auto']} 
                        tick={{fill: '#a0a0a0'}} 
                        width={80}
                        tickFormatter={(value) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(value)}
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(20,20,25,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', backdropFilter: 'blur(10px)' }} 
                        itemStyle={{ color: '#fff' }}
                      />
                      <Legend wrapperStyle={{ paddingTop: '20px' }} />
                      <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Total Loss" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                      <Line type="monotone" dataKey="actor_loss" stroke="#00f2fe" name="Actor Loss" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="critic_loss" stroke="#4facfe" name="Critic Loss" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </section>
          </>
        )}

        {activeTab === 'logs' && (
          <section className="glass-panel full-width">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
              <h3>Live System Logs</h3>
              <div className="nav-tabs" style={{ marginBottom: 0 }}>
                <button 
                  className={`tab-btn ${activeSubTab === 'market_collector' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('market_collector')}
                >
                  Market Collector
                </button>
                <button 
                  className={`tab-btn ${activeSubTab === 'ai_trainer' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('ai_trainer')}
                >
                  AI Trainer
                </button>
                <button 
                  className={`tab-btn ${activeSubTab === 'trading_bot' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('trading_bot')}
                >
                  Trading Bot
                </button>
                <button 
                  className={`tab-btn ${activeSubTab === 'whale_tracker' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('whale_tracker')}
                >
                  Whale Tracker
                </button>
                <button 
                  className={`tab-btn ${activeSubTab === 'statarb_collector' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('statarb_collector')}
                >
                  StatArb Collector
                </button>
              </div>
            </div>
            
            <div className="logs-grid" style={{ display: 'block' }}>
              {activeSubTab === 'market_collector' && <LogViewer activeTab={activeTab} botName="market_collector" title="Market Collector" />}
              {activeSubTab === 'ai_trainer' && <LogViewer activeTab={activeTab} botName="ai_trainer" title="AI Trainer" />}
              {activeSubTab === 'trading_bot' && <LogViewer activeTab={activeTab} botName="trading_bot" title="Trading Bot (Execution)" />}
              {activeSubTab === 'whale_tracker' && <LogViewer activeTab={activeTab} botName="whale_tracker" title="Whale Tracker" />}
              {activeSubTab === 'statarb_collector' && <LogViewer activeTab={activeTab} botName="statarb_collector" title="StatArb Collector" />}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
