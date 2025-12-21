import React, { useEffect, useState, useRef } from 'react';
import { Activity, Terminal, Share2, GitBranch } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import LogStream from './components/LogStream';
import './index.css';

const App = () => {
  const [logs, setLogs] = useState([]);
  const [activeNode, setActiveNode] = useState('start');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);

  useEffect(() => {
    // Determine Websocket URL. In dev (Vite) we might need to point to port 8000 explicity.
    // In prod/CLI launch, we might be served from the same origin.
    // We'll try to guess logic: if port is 5173 (Vite), assume backend is 8000.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.hostname;
    let port = window.location.port;
    if (port === '5173') port = '8000'; // Dev override

    const wsUrl = `${protocol}//${host}:${port}/ws`;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => setConnectionStatus('connected');
      ws.onclose = () => {
        setConnectionStatus('disconnected');
        setTimeout(connect, 3000); // Reconnect
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

    // Simple heuristic to determine active node from logs
    // In a real implementation, we'd send structured state events.
    const msg = record.message.toLowerCase();
    if (msg.includes('generating strategies')) setActiveNode('strategy');
    else if (msg.includes('sampling hypothesis')) setActiveNode('hypothesis');
    else if (msg.includes('reasoning')) setActiveNode('reasoning');
    else if (msg.includes('verifying')) setActiveNode('verification');
    else if (msg.includes('final decision')) setActiveNode('decision');
  };

  return (
    <div className="flex flex-col h-screen bg-[var(--bg-color)] text-[var(--text-primary)]">
      {/* Header */}
      <header className="h-[var(--header-height)] glass-panel m-4 mb-0 flex items-center justify-between px-6 z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[var(--glass-border)] rounded-lg">
            <img src="/gcri-logo.png" alt="" className="w-6 h-6 object-contain hidden" />
            {/* Logo Placeholder */}
            <div className="w-6 h-6 bg-[var(--neon-cyan)] rounded-full blur-[2px] opacity-80" />
          </div>
          <h1 className="font-bold text-xl tracking-tight">
            GCRI <span className="text-[var(--text-secondary)] font-light">Agent Dashboard</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 text-sm ${connectionStatus === 'connected' ? 'text-[var(--neon-green)]' : 'text-[var(--neon-red)]'}`}>
            <span className="relative flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${connectionStatus === 'connected' ? 'bg-[var(--neon-green)]' : 'bg-[var(--neon-red)]'}`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${connectionStatus === 'connected' ? 'bg-[var(--neon-green)]' : 'bg-[var(--neon-red)]'}`}></span>
            </span>
            {connectionStatus === 'connected' ? 'System Online' : 'Reconnecting...'}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex gap-4 p-4 overflow-hidden relative">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-[var(--neon-cyan)] to-[var(--neon-purple)] opacity-[0.03] pointer-events-none z-0"></div>

        {/* Left: Graph Visualization (Large) */}
        <div className="flex-1 glass-panel flex flex-col relative z-10 overflow-hidden">
          <div className="p-4 border-b border-[var(--glass-border)] flex justify-between items-center">
            <div className="flex items-center gap-2">
              <GitBranch size={18} className="text-[var(--neon-purple)]" />
              <span className="font-semibold">Reasoning Graph</span>
            </div>
          </div>
          <div className="flex-1 relative flex items-center justify-center bg-[rgba(0,0,0,0.2)]">
            <GraphVisualizer activeNode={activeNode} />
          </div>
        </div>

        {/* Right: Logs & Stats (Sidebar) */}
        <div className="w-[450px] flex flex-col gap-4 z-10">
          {/* Logs */}
          <div className="flex-1 glass-panel flex flex-col overflow-hidden">
            <div className="p-4 border-b border-[var(--glass-border)] flex justify-between items-center bg-[rgba(255,255,255,0.02)]">
              <div className="flex items-center gap-2">
                <Terminal size={18} className="text-[var(--neon-cyan)]" />
                <span className="font-semibold">System Logs</span>
              </div>
              <span className="text-xs text-[var(--text-secondary)]">{logs.length} events</span>
            </div>
            <div className="flex-1 overflow-hidden p-0">
              <LogStream logs={logs} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
