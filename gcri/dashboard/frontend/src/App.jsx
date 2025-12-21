import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Terminal, Activity, Server, Database, GitMerge } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import LogStream from './components/LogStream';
import { GraphEngine } from './utils/GraphEngine';
import './index.css';

const App = () => {
  const [logs, setLogs] = useState([]);
  const [engineState, setEngineState] = useState(new GraphEngine().state);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const engine = useMemo(() => new GraphEngine(), []);
  const wsRef = useRef(null);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.hostname;
    let port = window.location.port;
    if (port === '5173') port = '8000';

    const wsUrl = `${protocol}//${host}:${port}/ws`;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => setConnectionStatus('connected');
      ws.onclose = () => {
        setConnectionStatus('disconnected');
        setTimeout(connect, 3000);
      };
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'log') {
          handleLogMessage(message.data);
        }
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const handleLogMessage = (record) => {
    setLogs((prev) => [...prev, record]);
    const newState = engine.process(record);
    setEngineState(newState);
  };

  return (
    <div className="flex flex-col h-screen bg-[#050505] text-white overflow-hidden font-mono selection:bg-[var(--neon-cyan)] selection:text-black">
      {/* HUD Overlay Grid */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-5 pointer-events-none z-0"></div>
      <div className="absolute inset-0 pointer-events-none z-0"
        style={{ backgroundImage: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))', backgroundSize: '100% 2px, 3px 100%' }}>
      </div>

      {/* Header */}
      <header className="h-14 border-b border-[var(--glass-border)] flex items-center justify-between px-6 z-20 bg-[rgba(5,5,5,0.8)] backdrop-blur-md">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-[var(--neon-cyan)]">
            <Activity size={20} />
            <h1 className="font-bold tracking-widest text-lg">GCRI <span className="text-xs opacity-50">CORTEX VISUALIZER v2.0</span></h1>
          </div>
          <div className="h-6 w-[1px] bg-[var(--glass-border)]"></div>
          <div className="text-xs text-[var(--text-secondary)]">
            ITERATION: <span className="text-white font-bold">{engineState.iteration}</span>
          </div>
          <div className="text-xs text-[var(--text-secondary)]">
            PHASE: <span className="text-[var(--neon-purple)] font-bold uppercase">{engineState.phase}</span>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <div className={`flex items-center gap-2 ${connectionStatus === 'connected' ? 'text-[var(--neon-green)]' : 'text-[var(--neon-red)]'}`}>
            <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-[var(--neon-green)] animate-pulse' : 'bg-[var(--neon-red)]'}`}></div>
            {connectionStatus === 'connected' ? 'LINK ESTABLISHED' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <main className="flex-1 flex overflow-hidden relative z-10">

        {/* Center: Dynamic Graph */}
        <div className="flex-1 relative flex flex-col">
          <div className="absolute top-4 left-4 z-20 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
              <Server size={14} />
              <span>ACTIVE AGENTS</span>
            </div>
          </div>

          <div className="flex-1 overflow-hidden relative" id="graph-container">
            <GraphVisualizer state={engineState} />
          </div>

          {/* Bottom Status Bar */}
          <div className="h-12 border-t border-[var(--glass-border)] bg-[rgba(0,0,0,0.4)] backdrop-blur px-6 flex items-center justify-between text-xs">
            <div className="flex gap-8">
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-cyan)]"></span> Strategy
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-green)]"></span> Execution
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-purple)]"></span> Refinement
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-red)]"></span> Verification
              </div>
            </div>
          </div>
        </div>

        {/* Right: Terminal Sidebar */}
        <div className="w-[400px] border-l border-[var(--glass-border)] bg-[rgba(0,0,0,0.6)] flex flex-col backdrop-blur-xl">
          <div className="p-3 border-b border-[var(--glass-border)] flex justify-between items-center bg-[rgba(255,255,255,0.02)]">
            <div className="flex items-center gap-2 text-[var(--neon-cyan)]">
              <Terminal size={16} />
              <span className="font-bold text-sm tracking-wider">SYSTEM LOGS</span>
            </div>
          </div>
          <div className="flex-1 overflow-hidden p-0 relative">
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent to-[var(--bg-color)] opacity-50 z-10"></div>
            <LogStream logs={logs} />
          </div>

          {/* Memory / State View */}
          <div className="h-1/3 border-t border-[var(--glass-border)] flex flex-col">
            <div className="p-3 border-b border-[var(--glass-border)] flex items-center gap-2 text-[var(--neon-purple)]">
              <Database size={16} />
              <span className="font-bold text-sm tracking-wider">MEMORY STATE</span>
            </div>
            <div className="flex-1 p-4 font-mono text-xs text-[var(--text-secondary)] overflow-auto">
              {engineState.memory.length > 0 ? (
                engineState.memory.map((m, i) => <div key={i} className="mb-2 border-l-2 border-[var(--neon-purple)] pl-2">{m}</div>)
              ) : (
                <div className="opacity-30 italic">No constraints yet...</div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
