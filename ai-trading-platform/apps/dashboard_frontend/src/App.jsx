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

  // Training State
  const [trainingMetrics, setTrainingMetrics] = useState([]);

  // Fetch Live Trading Updates
  useEffect(() => {
    if (activeTab !== 'live') return;
    
    const fetchLiveState = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/state");
        const json = await res.json();
        setData(json);
      } catch (err) {
        console.error("Failed to fetch live state:", err);
      }
    };
    
    fetchLiveState();
    const interval = setInterval(fetchLiveState, 1000);
    return () => clearInterval(interval);
  }, [activeTab]);

  // Fetch Training Metrics from API
  useEffect(() => {
    if (activeTab !== 'training') return;
    
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/training");
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
        <h1>AI Trading Platform Dashboard</h1>
        
        <div className="nav-tabs">
          <button 
            className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            Live Trading
          </button>
          <button 
            className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            AI Training
          </button>
        </div>
        
        {activeTab === 'live' && (
          <div className="equity-display">
            <span>Total Equity:</span>
            <h2>${data.equity.toFixed(2)}</h2>
          </div>
        )}
      </header>

      <main className="dashboard-grid">
        {activeTab === 'live' ? (
          <>
            <section className="glass-panel">
              <h3>Active Positions</h3>
              <div className="position-list">
                {data.positions.map((pos, idx) => (
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
                ))}
              </div>
            </section>

            <section className="glass-panel">
              <h3>Live Market States</h3>
              <div className="market-list">
                {Object.entries(data.market_states).map(([sym, state], idx) => (
                  <div key={idx} className="market-card">
                    <span className="symbol">{sym}</span>
                    <span className="price">${state.price.toFixed(2)}</span>
                    <span className={`trend ${state.trend.toLowerCase()}`}>{state.trend}</span>
                  </div>
                ))}
              </div>
            </section>
          </>
        ) : (
          <section className="glass-panel full-width">
            <h3>PPO Training Progress (Live Loss Metrics)</h3>
            
            {trainingMetrics.length === 0 ? (
              <div className="empty-state">
                <p>Waiting for training data...</p>
                <small>Run `python -m apps.trainer.ppo` to start the AI training engine.</small>
              </div>
            ) : (
              <div className="chart-container" style={{ width: '100%', height: 400 }}>
                <ResponsiveContainer>
                  <LineChart
                    data={trainingMetrics}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#888" />
                    <YAxis stroke="#888" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e1e1e', borderColor: '#333' }} />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Total Loss" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="actor_loss" stroke="#82ca9d" name="Actor Loss" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="critic_loss" stroke="#ffc658" name="Critic Loss" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
