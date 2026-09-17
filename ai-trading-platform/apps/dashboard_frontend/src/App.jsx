import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('live') // 'live' or 'training'
  
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
        </div>
        
        <div className="equity-display">
          <span>Total Equity</span>
          <h2 className={data.equity > 10000 ? 'profit' : data.equity < 10000 ? 'loss' : ''}>
            ${data.equity.toFixed(2)}
          </h2>
        </div>
      </header>

      <main className="dashboard-grid">
        {activeTab === 'live' ? (
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
                    </div>
                  ))
                )}
              </div>
            </section>
          </>
        ) : (
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
      </main>
    </div>
  )
}

export default App

