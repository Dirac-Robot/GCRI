import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Terminal, Activity, Server, Database, GitMerge, Play, StopCircle } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import LogStream from './components/LogStream';
import PlanningVisualizer from './components/PlanningVisualizer';
import StructuredView from './components/StructuredView';
import { GraphEngine } from './utils/GraphEngine';
import './index.css';

// --- Components ---

const TaskModal = ({ onClose }) => {
  const [task, setTask] = useState('');
  const [agentMode, setAgentMode] = useState('unit'); // 'unit' or 'planner'
  const [commitMode, setCommitMode] = useState('manual'); // 'auto-accept', 'auto-reject', 'manual'
  const [status, setStatus] = useState('idle'); // idle, loading, success, error

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!task.trim()) return;

    setStatus('loading');
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, agent_mode: agentMode, commit_mode: commitMode })
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

              {/* Commit Mode Selector */}
              <div className="mb-4">
                <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider">On Completion</label>
                <select
                  value={commitMode}
                  onChange={(e) => setCommitMode(e.target.value)}
                  className="w-full bg-[#111] border border-[#333] rounded px-3 py-2 text-sm text-white focus:border-[var(--neon-purple)] focus:outline-none transition-colors cursor-pointer"
                >
                  <option value="manual">🔍 Request Review</option>
                  <option value="auto-accept">✅ Always Accept</option>
                  <option value="auto-reject">❌ Always Reject (Benchmark)</option>
                </select>
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

// --- Commit Modal ---
const CommitModal = ({ context, onClose }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleResponse = async (approved) => {
    setIsLoading(true);
    try {
      await fetch('/api/commit/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved })
      });
    } catch (e) {
      console.error('Failed to send commit response:', e);
    }
    setIsLoading(false);
    onClose();
  };

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.8)] backdrop-blur-sm">
      <div className="bg-[#0a0a0a] border border-[var(--neon-green)] rounded-xl p-6 max-w-lg w-full shadow-[0_0_50px_rgba(0,255,0,0.2)]">
        <h2 className="text-xl font-bold text-[var(--neon-green)] mb-4 flex items-center gap-2">
          🏆 Task Completed!
        </h2>
        <div className="text-gray-300 mb-4">
          <p className="mb-2">Winning branch: <span className="text-[var(--neon-cyan)] font-bold">#{(context?.best_branch_index || 0) + 1}</span></p>
          <p className="text-sm text-gray-500">Apply changes from winning branch to project root?</p>
        </div>
        {context?.final_output && (
          <div className="bg-[#111] border border-[#333] rounded p-3 mb-4 max-h-32 overflow-auto">
            <div className="text-xs text-gray-500 mb-1">Final Output:</div>
            <pre className="text-xs text-gray-400 whitespace-pre-wrap">{context.final_output}</pre>
          </div>
        )}
        <div className="flex justify-end gap-3">
          <button
            onClick={() => handleResponse(false)}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-bold bg-[#333] text-gray-300 rounded hover:bg-[#444] transition-colors disabled:opacity-50"
          >
            Discard
          </button>
          <button
            onClick={() => handleResponse(true)}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-bold bg-[var(--neon-green)] text-black rounded hover:bg-[#4fff4f] transition-colors disabled:opacity-50 shadow-[0_0_10px_rgba(0,255,0,0.4)]"
          >
            {isLoading ? 'Merging...' : 'Merge Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

const DetailsModal = ({ data, files, onClose }) => {
  const [activeTab, setActiveTab] = useState('details');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [workspaceFiles, setWorkspaceFiles] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);

  // Get work_dir from node content
  const workDir = data?.content?.work_dir || null;

  // Fetch workspace files when work_dir changes or tab switches
  useEffect(() => {
    if (activeTab === 'workspace' && workDir) {
      setLoadingFiles(true);
      fetch('/api/workspace/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ work_dir: workDir })
      })
        .then(res => res.json())
        .then(result => {
          setWorkspaceFiles(result.files || []);
          setLoadingFiles(false);
        })
        .catch(() => {
          setWorkspaceFiles([]);
          setLoadingFiles(false);
        });
    }
  }, [workDir, activeTab]);

  if (!data) return null;

  const nodeColor = data.color || 'var(--neon-cyan)';

  const handleFileClick = async (file) => {
    setSelectedFile(file);
    try {
      const response = await fetch('/api/workspace/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: file.full_path })
      });
      const result = await response.json();
      setFileContent(result.content || result.error || 'No content');
    } catch (e) {
      setFileContent('Error loading file.');
    }
  };

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.7)] backdrop-blur-sm p-20" onClick={onClose}>
      <div className="bg-[#0a0a0a] w-full max-w-5xl h-[80vh] rounded-xl overflow-hidden flex flex-col" style={{ border: `1px solid ${nodeColor}`, boxShadow: `0 0 50px ${nodeColor}20` }} onClick={e => e.stopPropagation()}>

        {/* Header with Tabs */}
        <div className="flex justify-between items-center px-6 pt-4 border-b border-[#333] bg-[rgba(255,255,255,0.03)]">
          <div className="flex gap-6">
            <button
              onClick={() => setActiveTab('details')}
              className="pb-4 text-sm font-bold tracking-wider transition-colors border-b-2"
              style={{ color: activeTab === 'details' ? nodeColor : '#6b7280', borderColor: activeTab === 'details' ? nodeColor : 'transparent' }}
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
              <h2 className="text-xl font-bold mb-4" style={{ color: nodeColor }}>{data.title}</h2>
              <StructuredView data={data.content} />
            </div>
          )}

          {activeTab === 'workspace' && (
            <div className="flex w-full h-full">
              {/* File List */}
              <div className="w-1/3 border-r border-[#333] overflow-auto p-4 bg-[rgba(0,0,0,0.3)]">
                {!workDir ? (
                  <div className="text-gray-500 italic text-xs">No workspace available for this node.</div>
                ) : loadingFiles ? (
                  <div className="text-gray-500 italic text-xs">Loading files...</div>
                ) : workspaceFiles.length > 0 ? (
                  <div className="space-y-1">
                    <div className="text-[var(--neon-purple)] font-bold text-xs mb-2 uppercase">Branch Workspace</div>
                    {workspaceFiles.map((f, i) => (
                      <div
                        key={i}
                        onClick={() => handleFileClick(f)}
                        className={`cursor-pointer text-xs font-mono truncate px-2 py-1 rounded ${selectedFile?.full_path === f.full_path ? 'bg-[var(--neon-purple)] text-black' : 'text-gray-400 hover:bg-[#222]'}`}
                      >
                        {f.path}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-500 italic text-xs">No files in workspace.</div>
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
  const [plannerHistory, setPlannerHistory] = useState([]); // Array of states
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedNode, setSelectedNode] = useState(null);
  const [workspaceFiles, setWorkspaceFiles] = useState([]);
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [viewingIterationIndex, setViewingIterationIndex] = useState(null); // null = live, number = viewing history
  const [isTaskRunning, setIsTaskRunning] = useState(false);
  const [commitRequest, setCommitRequest] = useState(null); // {context, pending}

  const engine = useMemo(() => new GraphEngine(), []);
  const wsRef = useRef(null);

  // Determine which state to display
  const displayState = useMemo(() => {
    if (viewingIterationIndex !== null && engineState.history && engineState.history[viewingIterationIndex]) {
      return engineState.history[viewingIterationIndex];
    }
    return engineState;
  }, [viewingIterationIndex, engineState]);

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
        } else if (message.type === 'history') {
          // Replay all historical logs to restore state
          console.log(`Restoring state from ${message.data.length} historical logs...`);
          message.data.forEach(record => {
            try {
              engine.process(record);
            } catch (e) {
              console.error("Engine process error during replay", e);
            }
          });
          setLogs(message.data);
          setEngineState({ ...engine.state });
        } else if (message.type === 'workspace_update') {
          setWorkspaceFiles(message.data);
        } else if (message.type === 'planner_state') {
          // Accumulate history
          setPlannerHistory(prev => {
            const newState = message.data;
            // Optional: Reset if we see a 'start' stage at 0? 
            if (newState.stage === 'start' && newState.plan_count === 0) {
              return [newState];
            }
            return [...prev, newState];
          });
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

      // Track running state based on phase changes
      const extra = record.record?.extra || record.extra || {};
      if (extra.ui_event === 'phase_change') {
        const phase = extra.phase;
        if (phase === 'strategy') {
          setIsTaskRunning(true);
        } else if (phase === 'idle' || phase === 'complete') {
          setIsTaskRunning(false);
        }
      }
      // Handle state reset (new task starting)
      if (extra.ui_event === 'state_reset') {
        engine.reset();
        setEngineState(engine.state);
        setLogs([]);
        setViewingIterationIndex(null);
        setSelectedNode(null);
        setWorkspaceFiles([]);
        setPlannerHistory([]);
      }
      // Handle abort event
      if (extra.ui_event === 'abort') {
        setIsTaskRunning(false);
      }
      // Handle commit request event
      if (extra.ui_event === 'commit_request') {
        setCommitRequest({ context: extra.context, pending: true });
      }
      // Handle commit timeout
      if (extra.ui_event === 'commit_timeout') {
        setCommitRequest(null);
      }
    } catch (e) {
      console.error("Engine process error", e);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#050505] text-white overflow-hidden font-mono selection:bg-[var(--neon-cyan)] selection:text-black relative">
      <DetailsModal data={selectedNode} files={workspaceFiles} onClose={() => setSelectedNode(null)} />
      {showTaskModal && <TaskModal onClose={() => setShowTaskModal(false)} />}
      {commitRequest && (
        <CommitModal
          context={commitRequest.context}
          onClose={() => setCommitRequest(null)}
        />
      )}

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

          <button
            onClick={async () => {
              try {
                const res = await fetch('/api/abort', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'aborting') {
                  setIsTaskRunning(false);
                }
              } catch (e) {
                console.error('Abort failed:', e);
              }
            }}
            disabled={!isTaskRunning}
            className="flex items-center gap-2 bg-[rgba(255,0,0,0.1)] border border-[var(--neon-red)] text-[var(--neon-red)] px-3 py-1 rounded hover:bg-[var(--neon-red)] hover:text-black transition-all text-xs font-bold shadow-[0_0_10px_rgba(255,0,0,0.2)] disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <StopCircle size={12} /> ABORT
          </button>

          <div className="h-6 w-[1px] bg-[var(--glass-border)]"></div>

          {/* Iteration Navigation */}
          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={() => setViewingIterationIndex(prev =>
                prev === null ? (engineState.history?.length > 0 ? engineState.history.length - 1 : null) : Math.max(0, prev - 1)
              )}
              disabled={viewingIterationIndex === 0 || (viewingIterationIndex === null && (!engineState.history || engineState.history.length === 0))}
              className="px-2 py-1 bg-[rgba(255,255,255,0.05)] border border-[#333] rounded hover:border-[var(--neon-cyan)] disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ◀
            </button>
            <div className="text-[var(--text-secondary)] min-w-[80px] text-center">
              {viewingIterationIndex !== null ? (
                <span className="text-[var(--neon-yellow)]">VIEWING #{viewingIterationIndex + 1}</span>
              ) : (
                <span>ITERATION: <span className="text-white font-bold">{engineState.iteration + 1}</span></span>
              )}
            </div>
            <button
              onClick={() => setViewingIterationIndex(prev => {
                if (prev === null) return null;
                if (prev >= (engineState.history?.length || 0) - 1) return null; // Go to live
                return prev + 1;
              })}
              disabled={viewingIterationIndex === null}
              className="px-2 py-1 bg-[rgba(255,255,255,0.05)] border border-[#333] rounded hover:border-[var(--neon-cyan)] disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ▶
            </button>
            {viewingIterationIndex !== null && (
              <button
                onClick={() => setViewingIterationIndex(null)}
                className="px-2 py-1 bg-[var(--neon-green)] text-black font-bold rounded text-xs hover:shadow-[0_0_10px_var(--neon-green)]"
              >
                LIVE
              </button>
            )}
          </div>

          <div className="text-xs text-[var(--text-secondary)]">
            PHASE: <span className="text-[var(--neon-purple)] font-bold uppercase">{displayState.phase}</span>
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
        <PlanningVisualizer history={plannerHistory} />

        {/* Center: Dynamic Graph */}
        <div className="flex-1 relative flex flex-col">
          <div className="absolute top-4 left-4 z-20 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
              <Server size={14} />
              <span>ACTIVE AGENTS</span> (Click for details)
            </div>
          </div>

          <div className="flex-1 overflow-hidden relative" id="graph-container">
            <GraphVisualizer state={displayState} onNodeSelect={setSelectedNode} />
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
