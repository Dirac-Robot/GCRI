import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Terminal, Activity, Server, Database, GitMerge, Play } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import LogStream from './components/LogStream';
import PlanningVisualizer from './components/PlanningVisualizer';
import { GraphEngine } from './utils/GraphEngine';
import './index.css';

// --- Components ---

const TaskModal = ({ onClose }) => {
  const [task, setTask] = useState('');
  const [agentMode, setAgentMode] = useState('unit'); // 'unit' or 'planner'
  const [status, setStatus] = useState('idle'); // idle, loading, success, error

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!task.trim()) return;

    setStatus('loading');
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, agent_mode: agentMode })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setStatus('success');
      setTimeout(onClose, 1000);
    } catch (err) {
      console.error(err);
      setStatus('error');
    }
  };

  return (
    <div className="absolute inset-0 z-[60] flex items-center justify-center bg-[rgba(0,0,0,0.8)] backdrop-blur-sm p-4" onClick={onClose}>
      <div className="bg-[#0a0a0a] border border-[var(--neon-green)] w-full max-w-lg rounded-xl overflow-hidden shadow-[0_0_30px_rgba(0,255,0,0.2)]" onClick={e => e.stopPropagation()}>
        <div className="p-6 border-b border-[#333] bg-[rgba(255,255,255,0.03)] flex justify-between items-center">
          <h2 className="text-lg font-bold text-[var(--neon-green)] flex items-center gap-2">
            <Play size={18} /> INITIATE NEW PROTOCOL
          </h2>
          <button onClick={onClose} className="text-white hover:text-[var(--neon-red)]">&times;</button>
        </div>
        <div className="p-6">
          {status === 'success' ? (
            <div className="text-center py-8 text-[var(--neon-green)] font-bold animate-pulse">
              COMMAND ACCEPTED. INITIALIZING {agentMode === 'unit' ? 'UNIT AGENT' : 'META PLANNER'}...
            </div>
          ) : (
            <form onSubmit={handleSubmit}>

              {/* Agent Mode Selector */}
              <div className="flex bg-[#111] p-1 rounded mb-4 border border-[#333]">
                <button
                  type="button"
                  onClick={() => setAgentMode('unit')}
                  className={`flex-1 py-2 text-xs font-bold tracking-wider rounded transition-all ${agentMode === 'unit' ? 'bg-[var(--neon-green)] text-black shadow-[0_0_10px_rgba(0,255,0,0.3)]' : 'text-gray-500 hover:text-white'}`}
                >
                  UNIT AGENT
                </button>
                <button
                  type="button"
                  onClick={() => setAgentMode('planner')}
                  className={`flex-1 py-2 text-xs font-bold tracking-wider rounded transition-all ${agentMode === 'planner' ? 'bg-[var(--neon-cyan)] text-black shadow-[0_0_10px_rgba(0,255,255,0.3)]' : 'text-gray-500 hover:text-white'}`}
                >
                  META PLANNER
                </button>
              </div>

              <textarea
                className="w-full bg-[#050505] border border-[#333] rounded p-3 text-white font-mono text-sm focus:border-[var(--neon-green)] focus:outline-none transition-colors h-32 resize-none"
                placeholder={agentMode === 'unit' ? "Describe a specific task for the Unit Agent..." : "Describe a high-level goal for the Meta Planner..."}
                value={task}
                onChange={(e) => setTask(e.target.value)}
                autoFocus
              />
              <div className="flex justify-between items-center mt-4">
                <span className="text-xs text-gray-500">{status === 'error' && <span className="text-[var(--neon-red)]">Command Failed. Check console.</span>}</span>
                <button
                  type="submit"
                  disabled={status === 'loading' || !task.trim()}
                  className={`font-bold px-6 py-2 rounded shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed ${agentMode === 'unit' ? 'bg-[var(--neon-green)] text-black hover:bg-[#4fff4f] shadow-[0_0_10px_rgba(0,255,0,0.4)]' : 'bg-[var(--neon-cyan)] text-black hover:bg-[#4fffff] shadow-[0_0_10px_rgba(0,255,255,0.4)]'}`}
                >
                  {status === 'loading' ? 'TRANSMITTING...' : 'EXECUTE'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

const DetailsModal = ({ data, files, onClose }) => {
  const [activeTab, setActiveTab] = useState('details');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');

  if (!data) return null;

  const handleFileClick = async (file) => {
    setSelectedFile(file);
    try {
      const response = await fetch(`/api/file?path=${encodeURIComponent(file.full_path)}`);
      const result = await response.json();
      setFileContent(result.content || result.error);
    } catch (e) {
      setFileContent('Error loading file.');
    }
  };

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.7)] backdrop-blur-sm p-20" onClick={onClose}>
      <div className="bg-[#0a0a0a] border border-[var(--neon-cyan)] w-full max-w-5xl h-[80vh] rounded-xl overflow-hidden shadow-[0_0_50px_rgba(0,255,255,0.1)] flex flex-col" onClick={e => e.stopPropagation()}>

        {/* Header with Tabs */}
        <div className="flex justify-between items-center px-6 pt-4 border-b border-[#333] bg-[rgba(255,255,255,0.03)]">
          <div className="flex gap-6">
            <button
              onClick={() => setActiveTab('details')}
              className={`pb-4 text-sm font-bold tracking-wider transition-colors border-b-2 ${activeTab === 'details' ? 'text-[var(--neon-cyan)] border-[var(--neon-cyan)]' : 'text-gray-500 border-transparent hover:text-gray-300'}`}
            >
              <div className="flex items-center gap-2"><Activity size={16} /> NODE DETAILS</div>
            </button>
            <button
              onClick={() => setActiveTab('workspace')}
              className={`pb-4 text-sm font-bold tracking-wider transition-colors border-b-2 ${activeTab === 'workspace' ? 'text-[var(--neon-purple)] border-[var(--neon-purple)]' : 'text-gray-500 border-transparent hover:text-gray-300'}`}
            >
              <div className="flex items-center gap-2"><Database size={16} /> WORKSPACE</div>
            </button>
          </div>
          <button onClick={onClose} className="text-white hover:text-[var(--neon-red)] transition-colors text-2xl mb-4">&times;</button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-hidden flex">
          {activeTab === 'details' && (
            <div className="p-6 overflow-auto font-mono text-sm text-gray-300 whitespace-pre-wrap w-full">
              <h2 className="text-xl font-bold text-[var(--neon-cyan)] mb-4">{data.title}</h2>
              {data.content || <span className="opacity-50 italic">No content available yet...</span>}
            </div>
          )}

          {activeTab === 'workspace' && (
            <div className="flex w-full h-full">
              {/* File List */}
              <div className="w-1/3 border-r border-[#333] overflow-auto p-4 bg-[rgba(0,0,0,0.3)]">
                {files && files.length > 0 ? (
                  files.map((root, i) => (
                    <div key={i} className="mb-4">
                      <div className="text-[var(--neon-purple)] font-bold text-xs mb-2 uppercase">{root.name}</div>
                      <div className="pl-2 space-y-1">
                        {root.children.map((f, j) => (
                          <div
                            key={j}
                            onClick={() => handleFileClick(f)}
                            className={`cursor-pointer text-xs font-mono truncate px-2 py-1 rounded ${selectedFile === f ? 'bg-[var(--neon-purple)] text-black' : 'text-gray-400 hover:bg-[#222]'}`}
                          >
                            {f.name}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-gray-500 italic text-xs">No files monitored.</div>
                )}
              </div>
              {/* File Viewer */}
              <div className="w-2/3 p-4 overflow-auto bg-[#050505]">
                {selectedFile ? (
                  <>
                    <div className="text-xs text-gray-500 mb-2">{selectedFile.path}</div>
                    <pre className="text-xs font-mono text-gray-300 whitespace-pre-wrap border border-[#222] p-4 rounded bg-[#080808]">
                      {fileContent}
                    </pre>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-600 text-sm">Select a file to view</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// --- Main App ---

const App = () => {
  const [logs, setLogs] = useState([]);
  const [engineState, setEngineState] = useState(new GraphEngine().state);
  const [plannerState, setPlannerState] = useState(null); // New Planner State
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedNode, setSelectedNode] = useState(null);
  const [workspaceFiles, setWorkspaceFiles] = useState([]);
  const [showTaskModal, setShowTaskModal] = useState(false);

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
        } else if (message.type === 'workspace_update') {
          setWorkspaceFiles(message.data);
        } else if (message.type === 'planner_state') {
          // Handle Planner State Update
          setPlannerState(message.data);
        }
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const handleLogMessage = (record) => {
    if (!record) return;
    setLogs((prev) => [...prev, record]);
    try {
      const newState = engine.process(record);
      setEngineState(newState);
    } catch (e) {
      console.error("Engine process error", e);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#050505] text-white overflow-hidden font-mono selection:bg-[var(--neon-cyan)] selection:text-black relative">
      <DetailsModal data={selectedNode} files={workspaceFiles} onClose={() => setSelectedNode(null)} />
      {showTaskModal && <TaskModal onClose={() => setShowTaskModal(false)} />}

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
            <h1 className="font-bold tracking-widest text-lg">GCRI <span className="text-xs opacity-50">CORTEX VISUALIZER v2.2</span></h1>
          </div>

          <button
            onClick={() => setShowTaskModal(true)}
            className="ml-4 flex items-center gap-2 bg-[rgba(0,255,0,0.1)] border border-[var(--neon-green)] text-[var(--neon-green)] px-3 py-1 rounded hover:bg-[var(--neon-green)] hover:text-black transition-all text-xs font-bold shadow-[0_0_10px_rgba(0,255,0,0.2)]"
          >
            <Play size={12} /> NEW TASK
          </button>

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

        {/* Left: Planning Visualizer */}
        <PlanningVisualizer plannerState={plannerState} />

        {/* Center: Dynamic Graph */}
        <div className="flex-1 relative flex flex-col">
          <div className="absolute top-4 left-4 z-20 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
              <Server size={14} />
              <span>ACTIVE AGENTS</span> (Click for details)
            </div>
          </div>

          <div className="flex-1 overflow-hidden relative" id="graph-container">
            <GraphVisualizer state={engineState} onNodeSelect={setSelectedNode} />
          </div>

          {/* Bottom Status Bar */}
          <div className="h-12 border-t border-[var(--glass-border)] bg-[rgba(0,0,0,0.4)] backdrop-blur px-6 flex items-center justify-between text-xs">
            <div className="flex gap-8">
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-cyan)] shadow-[0_0_5px_var(--neon-cyan)]"></span> Strategy
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-green)] shadow-[0_0_5px_var(--neon-green)]"></span> Execution
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-purple)] shadow-[0_0_5px_var(--neon-purple)]"></span> Refinement
              </div>
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-3 h-3 rounded-sm bg-[var(--neon-red)] shadow-[0_0_5px_var(--neon-red)]"></span> Verification
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
          <div className="flex-1 min-h-0 relative flex flex-col">
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
