import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState({
    equity: 0.0,
    positions: [],
    market_states: {}
  });

  useEffect(() => {
    // In a real app, this would be a WebSocket or Server-Sent Events to the FastAPI backend.
    // We are mocking the polling for now to demonstrate the aesthetic.
    const interval = setInterval(() => {
      setData(prev => ({
        equity: prev.equity === 0 ? 10000 : prev.equity + (Math.random() * 10 - 5),
        positions: [
          { symbol: "BTCUSDT", side: "LONG", quantity: 0.5, pnl: Math.random() * 100 },
          { symbol: "ETHUSDT", side: "SHORT", quantity: 10.0, pnl: -Math.random() * 50 }
        ],
        market_states: {
          "BTCUSDT": { price: 65000 + Math.random() * 100, trend: "UP" },
          "ETHUSDT": { price: 3500 + Math.random() * 10, trend: "DOWN" }
        }
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <h1>AI Trading Platform Dashboard</h1>
        <div className="equity-display">
          <span>Total Equity:</span>
          <h2>${data.equity.toFixed(2)}</h2>
        </div>
      </header>

      <main className="dashboard-grid">
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
      </main>
    </div>
  )
}

export default App
