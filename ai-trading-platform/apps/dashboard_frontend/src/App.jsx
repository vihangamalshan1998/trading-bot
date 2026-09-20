import React, { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

// LogViewer Component (Must be outside App to prevent re-mounting on every interval)
const LogViewer = ({ botName, title, activeTab }) => {
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    
    // If user scrolls up (more than 30px from bottom), turn OFF auto-scroll.
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 30;
    setAutoScroll(isAtBottom);
  };

  useEffect(() => {
    if (activeTab !== 'logs') return;
    if (!autoScroll) return; // Pause polling entirely so logs stop jumping
    
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
  
  // Live Trading State
  const [data, setData] = useState({
    equity: 0.0,
    positions: [],
    market_states: {}
  });

  // System Stats (Macro & Model)
  const [systemStats, setSystemStats] = useState({
    news_count: 0,
    latest_sentiment: 0.0,
    latest_regime: 0.0,
    model_update_count: 0,
    last_model_update_time: null
  });

  // Training State
  const [trainingMetrics, setTrainingMetrics] = useState([]);

  // Fetch Live Trading Updates & System Stats
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

  const [tradeHistory, setTradeHistory] = useState([]);

  useEffect(() => {
    if (activeTab !== 'history') return;
    
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

  // Fetch Training Metrics from API
  useEffect(() => {
    if (activeTab !== 'training') return;
    
    const fetchMetrics = async () => {
      try {
        const res = await fetch("/api/training");
        const json = await res.json();
        
        // Format timestamp for display
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
  // Removed internal LogViewer definition

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <div className="header-title">
          <div className="pulse-indicator"></div>
          <h1>AI Trading Platform</h1>
        </div>
        
        <div className="nav-tabs">
          <button 
            className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            Live Market & Macro
          </button>
          <button 
            className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            Brain & Training
          </button>
          <button 
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            Positions & History
          </button>
          <button 
            className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Live Logs
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
            <section className="glass-panel stat-panel">
              <h3>Macro Sentiment & News</h3>
              <div className="stat-grid">
                <div className="stat-card">
                  <span className="stat-label">News Articles Analyzed</span>
                  <span className="stat-value highlight">{systemStats.news_count}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Gemini Sentiment Score</span>
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
                <div className="news-headlines" style={{ marginTop: '20px', padding: '15px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
                  <h4 style={{ color: '#8b9bb4', marginBottom: '10px', fontSize: '12px', textTransform: 'uppercase' }}>Latest Headlines Analyzed</h4>
                  <ul style={{ listStyleType: 'none', padding: 0, margin: 0 }}>
                    {systemStats.latest_headlines.map((headline, idx) => (
                      <li key={idx} style={{ 
                        padding: '8px 0', 
                        borderBottom: idx < systemStats.latest_headlines.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                        fontSize: '13px',
                        color: '#c9d1d9',
                        lineHeight: '1.4'
                      }}>
                        📰 {headline}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>

            <section className="glass-panel position-panel">
              <h3>Active AI Positions</h3>
              <div className="position-list">
                {data.positions.length === 0 ? (
                  <p className="empty-text">No active positions.</p>
                ) : (
                  data.positions.map((pos, idx) => (
                    <div key={idx} className={`position-card ${pos.side.toLowerCase()}`}>
                      <div className="pos-header">
                        <span className="symbol">{pos.symbol}</span>
                        <span className="side">{pos.side}</span>
                      </div>
                      <div className="pos-details">
                        <span>Qty: {pos.quantity}</span>
                        <span className={`pnl ${pos.pnl >= 0 ? 'profit' : 'loss'}`}>
                          PnL: ${pos.pnl.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="glass-panel market-panel full-width">
              <h3>Live Market States (Micro-Structure)</h3>
              <div className="market-list">
                {Object.keys(data.market_states).length === 0 ? (
                  <p className="empty-text">Awaiting market data from Binance...</p>
                ) : (
                  Object.entries(data.market_states).map(([sym, state], idx) => (
                    <div key={idx} className="market-card">
                      <div className="market-card-header">
                        <span className="symbol">{sym}</span>
                        <span className={`trend ${state.trend ? state.trend.toLowerCase() : ''}`}>{state.trend || 'N/A'}</span>
                      </div>
                      <div className="market-card-body">
                        <div className="market-metric">
                          <span className="label">Price</span>
                          <span className="value">${state.mid_price ? state.mid_price.toFixed(2) : '0.00'}</span>
                        </div>
                        <div className="market-metric">
                          <span className="label">Spread (BPS)</span>
                          <span className="value">{state.spread_bps ? state.spread_bps.toFixed(1) : '0.0'}</span>
                        </div>
                        <div className="market-metric">
                          <span className="label">Imbalance</span>
                          <span className="value">{state.imbalance ? state.imbalance.toFixed(2) : '0.00'}</span>
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
                          <span>Target Alloc:</span>
                          <div className="progress-bar-container">
                            <div className="progress-bar alloc" style={{ width: `${(state.ai_target_size || 0) * 100}%` }}></div>
                          </div>
                          <span>{((state.ai_target_size || 0) * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </>
        )}
        
        {activeTab === 'history' && (
          <section className="glass-panel full-width">
            <h3>Detailed Open Positions</h3>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Entry Price</th>
                    <th>Mark Price</th>
                    <th>Liq Price</th>
                    <th>Leverage</th>
                    <th>PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {data.positions.length === 0 ? (
                    <tr>
                      <td colSpan="8" className="empty-text">No active positions.</td>
                    </tr>
                  ) : (
                    data.positions.map((pos, idx) => (
                      <tr key={idx}>
                        <td className="symbol">{pos.symbol}</td>
                        <td className={pos.side.toLowerCase()}>{pos.side}</td>
                        <td>{pos.quantity} {pos.symbol.replace('USDT', '')}</td>
                        <td>${pos.entryPrice ? pos.entryPrice.toFixed(4) : '0.0000'}</td>
                        <td>${pos.markPrice ? pos.markPrice.toFixed(4) : '0.0000'}</td>
                        <td>${pos.liquidationPrice ? pos.liquidationPrice.toFixed(4) : '0.0000'}</td>
                        <td>{pos.leverage}x</td>
                        <td className={`pnl ${pos.pnl >= 0 ? 'profit' : 'loss'}`}>
                          ${pos.pnl.toFixed(2)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <h3 style={{ marginTop: '2rem' }}>AI Trade History (Last 100)</h3>
            <div className="table-responsive">
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
                    <th>CONFIDENCE</th>
                    <th>ORDER ID</th>
                  </tr>
                </thead>
                <tbody>
                  {tradeHistory.length === 0 ? (
                    <tr>
                      <td colSpan="9" className="empty-text">No trade history found.</td>
                    </tr>
                  ) : (
                    tradeHistory.map((trade, idx) => (
                      <tr key={idx}>
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
                           <div className="progress-bar-container" style={{ width: '80px', display: 'inline-block', marginRight: '10px' }}>
                             <div className="progress-bar" style={{ width: `${(trade.confidence || 0) * 100}%` }}></div>
                           </div>
                           {(trade.confidence * 100).toFixed(1)}%
                        </td>
                        <td>{trade.order_id || 'N/A'}</td>
                      </tr>
                    ))
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
              </div>
            </div>
            
            <div className="logs-grid" style={{ display: 'block' }}>
              {activeSubTab === 'market_collector' && <LogViewer activeTab={activeTab} botName="market_collector" title="Market Collector" />}
              {activeSubTab === 'ai_trainer' && <LogViewer activeTab={activeTab} botName="ai_trainer" title="AI Trainer" />}
              {activeSubTab === 'trading_bot' && <LogViewer activeTab={activeTab} botName="trading_bot" title="Trading Bot (Execution)" />}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App

