import React, { useState } from 'react';
import { Target, List, Brain, ChevronDown, ChevronRight, CheckCircle, Smartphone, AlertCircle } from 'lucide-react';

const PlanningVisualizer = ({ plannerState }) => {
    const [isMemoryCollapsed, setIsMemoryCollapsed] = useState(false);
    const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false);

    if (!plannerState) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-[var(--text-secondary)] bg-[rgba(0,0,0,0.2)] p-6 text-center border-r border-[var(--glass-border)]">
                <Target size={48} className="mb-4 opacity-20" />
                <div className="text-sm">Planner Inactive</div>
                <div className="text-xs opacity-50 mt-2">Start a "Meta Planner" task to activate strategic view.</div>
            </div>
        );
    }

    const { goal, plan_count, current_task, knowledge_context, memory, stage } = plannerState;
    const constraints = memory?.active_constraints || [];

    return (
        <div className="h-full flex flex-col bg-[rgba(5,5,5,0.95)] border-r border-[var(--glass-border)] w-[400px] overflow-hidden backdrop-blur-xl">

            {/* Header / Goal */}
            <div className="p-4 border-b border-[var(--glass-border)] bg-[rgba(255,255,255,0.02)]">
                <div className="flex items-center gap-2 text-[var(--neon-cyan)] mb-2">
                    <Target size={18} />
                    <h2 className="text-sm font-bold tracking-wider">STRATEGIC OBJECTIVE</h2>
                </div>
                <div className="text-sm font-medium text-white leading-relaxed line-clamp-3" title={goal}>
                    {goal}
                </div>
            </div>

            <div className="flex-1 overflow-auto p-4 space-y-6">

                {/* Current Focus */}
                <div className="relative">
                    <div className="absolute -left-2 top-0 bottom-0 w-[2px] bg-[var(--neon-green)] opacity-30 rounded-full"></div>
                    <div className="pl-4">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold text-[var(--neon-green)] flex items-center gap-2">
                                <Smartphone size={14} className="animate-pulse" /> CURRENT EXECUTION (Iter #{plan_count})
                            </span>
                            <span className="text-[10px] uppercase font-mono bg-[rgba(0,255,0,0.1)] text-[var(--neon-green)] px-2 py-0.5 rounded border border-[var(--neon-green)] opacity-70">
                                {stage || 'IDLE'}
                            </span>
                        </div>
                        {current_task ? (
                            <div className="bg-[rgba(0,255,0,0.05)] border border-[var(--neon-green)] p-3 rounded text-sm text-gray-200 shadow-[0_0_15px_rgba(0,255,0,0.1)]">
                                {current_task}
                            </div>
                        ) : (
                            <div className="text-xs text-gray-500 italic">Determining next strategic move...</div>
                        )}
                    </div>
                </div>

                {/* History / Context */}
                <div>
                    <button
                        onClick={() => setIsHistoryCollapsed(!isHistoryCollapsed)}
                        className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-white transition-colors text-xs font-bold tracking-wider mb-2 w-full"
                    >
                        {isHistoryCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                        <List size={14} />
                        EXECUTION HISTORY ({knowledge_context?.length || 0})
                    </button>

                    {!isHistoryCollapsed && (
                        <div className="space-y-3 pl-2 border-l border-[var(--glass-border)] ml-1.5">
                            {knowledge_context && knowledge_context.length > 0 ? (
                                knowledge_context.map((ctx, idx) => (
                                    <div key={idx} className="text-xs text-gray-400 group relative pl-4">
                                        <span className="absolute left-[-5px] top-1.5 w-2 h-2 rounded-full bg-[var(--glass-border)] group-hover:bg-[var(--neon-cyan)] transition-colors"></span>
                                        {ctx}
                                    </div>
                                ))
                            ) : (
                                <div className="text-xs text-gray-600 italic pl-4">No history recorded yet.</div>
                            )}
                        </div>
                    )}
                </div>

            </div>

            {/* Global Memory (Bottom) */}
            <div className="border-t border-[var(--glass-border)] bg-[rgba(0,0,0,0.3)]">
                <button
                    onClick={() => setIsMemoryCollapsed(!isMemoryCollapsed)}
                    className="w-full p-3 flex items-center justify-between text-[var(--neon-purple)] hover:bg-[rgba(255,255,255,0.03)] transition-colors"
                >
                    <div className="flex items-center gap-2 font-bold text-xs tracking-wider">
                        <Brain size={16} />
                        GLOBAL MEMORY ({constraints.length})
                    </div>
                    {isMemoryCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                </button>

                {!isMemoryCollapsed && (
                    <div className="p-3 pt-0 max-h-[200px] overflow-auto">
                        {constraints.length > 0 ? (
                            <ul className="space-y-2">
                                {constraints.map((c, i) => (
                                    <li key={i} className="text-xs text-gray-300 bg-[rgba(147,51,234,0.1)] border border-[rgba(147,51,234,0.3)] p-2 rounded flex gap-2 items-start">
                                        <AlertCircle size={12} className="text-[var(--neon-purple)] mt-0.5 shrink-0" />
                                        <span>{c}</span>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <div className="text-xs text-gray-600 italic text-center py-2">No active constraints.</div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default PlanningVisualizer;
