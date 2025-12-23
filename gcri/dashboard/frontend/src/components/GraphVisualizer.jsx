import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const Node = ({ label, subLabel, status, color, onClick, data }) => {
    const isPending = status === 'pending';
    const isActive = status === 'active';
    const isDone = status === 'done';

    // Brighter colors for "done" state too, not just active
    const effectiveOpacity = isPending ? 0.3 : 1;
    const effectiveColor = isPending ? 'rgba(255,255,255,0.2)' : color;

    return (
        <motion.div
            layout
            onClick={() => onClick && onClick(data)}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{
                opacity: effectiveOpacity,
                scale: 1,
                borderColor: isActive || isDone ? color : 'rgba(255,255,255,0.1)',
                backgroundColor: isActive ? 'rgba(10,10,10,0.9)' : 'rgba(0,0,0,0.6)',
                boxShadow: isActive ? `0 0 25px ${color}60` : (isDone ? `0 0 10px ${color}20` : 'none')
            }}
            whileHover={{ scale: 1.05, borderColor: color, cursor: 'pointer' }}
            className={`
                relative w-48 p-4 rounded-lg border backdrop-blur-md
                flex flex-col items-center justify-center gap-2
                transition-colors duration-300 z-10
            `}
        >
            {isActive && (
                <motion.div
                    layoutId="active-glow"
                    className="absolute inset-0 rounded-lg pointer-events-none"
                    style={{ border: `2px solid ${color}`, boxShadow: `0 0 20px ${color}` }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
            )}

            {/* Progress Bar Animation for Active Nodes */}
            {isActive && (
                <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none z-0">
                    <motion.div
                        className="h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-10"
                        initial={{ x: '-100%' }}
                        animate={{ x: '100%' }}
                        transition={{
                            repeat: Infinity,
                            duration: 1.5,
                            ease: "linear"
                        }}
                    />
                    <motion.div
                        className="absolute bottom-0 left-0 h-1 bg-current opacity-50"
                        style={{ color: color, backgroundColor: color }}
                        initial={{ width: '0%' }}
                        animate={{ width: '100%' }}
                        transition={{
                            repeat: Infinity,
                            duration: 2,
                            ease: "easeInOut"
                        }}
                    />
                </div>
            )}

            <div className="relative z-10 flex flex-col items-center gap-2">
                <div className="text-xs font-bold tracking-widest uppercase text-shadow-sm" style={{ color: color, textShadow: `0 0 10px ${color}` }}>
                    {label}
                </div>
                {subLabel && <div className="text-sm font-bold text-center text-white">{subLabel}</div>}
            </div>
        </motion.div>
    );
};

const Connector = ({ active, vertical = false, height = 40 }) => (
    <div className={`flex items-center justify-center ${vertical ? 'w-full' : 'h-full'}`} style={{ height: vertical ? height : 'auto' }}>
        <motion.div
            className={`${vertical ? 'w-[2px] h-full' : 'h-[2px] w-full'} bg-[#222] relative overflow-hidden`}
        >
            {active && (
                <motion.div
                    className={`absolute inset-0 bg-white`}
                    style={{ boxShadow: '0 0 10px white' }}
                    initial={{ x: vertical ? 0 : '-100%', y: vertical ? '-100%' : 0 }}
                    animate={{ x: 0, y: 0 }}
                    transition={{ duration: 0.5 }}
                />
            )}
        </motion.div>
    </div>
);

const GraphVisualizer = ({ state, onNodeSelect }) => {
    const { phase, branches, decision } = state;

    // Helper to determine node content based on status
    const getNodeContent = (status, nodeData, logs) => {
        if (status === 'pending') return { type: 'pending' };
        if (status === 'active') return nodeData || { type: 'processing', logs: logs || [] };
        // For 'done' status: return actual data or null (will show "No data available")
        return nodeData || null;
    };

    return (
        <div className="w-full h-full flex flex-col items-center justify-center p-10 overflow-auto">

            {/* 1. Strategy Layer */}
            <Node
                label="Strategy Generator"
                subLabel={`Strategies Generated: ${state.strategies.length}`}
                status={phase === 'idle' ? 'pending' : (phase === 'strategy' ? 'active' : 'done')}
                color="var(--neon-cyan)"
                onClick={onNodeSelect}
                data={{
                    type: 'strategy',
                    title: 'Strategy Generator',
                    color: 'var(--neon-cyan)',
                    content: getNodeContent(
                        phase === 'idle' ? 'pending' : (phase === 'strategy' ? 'active' : 'done'),
                        state.strategyData || { strategies: state.strategies },
                        // Strategy doesn't have separate logs array in this context yet, 
                        // but if we wanted to support it, we'd need to pass it. 
                        // For now default to empty logs if active.
                        []
                    )
                }}
            />

            {/* Connector */}
            <Connector vertical active={phase !== 'strategy'} height={40} />

            {/* 2. Branches Layer */}
            <div className="flex gap-10 w-full justify-center">
                {branches.map((branch, i) => {
                    const hypothesisStatus = branch.step === 'hypothesis' ? 'active' : (branch.step !== 'idle' ? 'done' : 'pending');
                    const reasoningStatus = branch.step === 'reasoning' ? 'active' : (branch.step === 'verification' || branch.step === 'done' ? 'done' : 'pending'); // logic check: reasoning is done if verif/done
                    // Actually, let's keep simplistic logic from before but map to the helper

                    return (
                        <div key={i} className="flex flex-col items-center gap-4 flex-1 max-w-[250px]">
                            {/* Branch Top Connector */}
                            <div className="w-[2px] h-8 bg-[#333] relative">
                                {/* Static line for now, dynamic later */}
                            </div>

                            {/* Hypothesis Node */}
                            {(() => {
                                const hypothesisData = branch.nodes?.hypothesis;
                                const isHypothesisDone = hypothesisData && hypothesisData.type !== 'processing';
                                const hypothesisStatus = branch.step === 'hypothesis' && !isHypothesisDone
                                    ? 'active'
                                    : (isHypothesisDone || branch.step !== 'idle' ? 'done' : 'pending');
                                return (
                                    <Node
                                        label={`Branch ${i + 1}`}
                                        subLabel="Hypothesis Agent"
                                        status={hypothesisStatus}
                                        color="var(--neon-green)"
                                        onClick={onNodeSelect}
                                        data={{
                                            type: 'branch',
                                            title: `Hypothesis (Branch ${i + 1})`,
                                            color: 'var(--neon-green)',
                                            content: getNodeContent(
                                                hypothesisStatus,
                                                hypothesisData,
                                                branch.logs
                                            )
                                        }}
                                    />
                                );
                            })()}

                            <Connector vertical active={branch.step === 'reasoning' || branch.step === 'verification'} height={20} />

                            {/* Reasoning Node */}
                            {(() => {
                                const reasoningData = branch.nodes?.reasoning;
                                const isReasoningDone = reasoningData && reasoningData.type !== 'processing';
                                const reasoningStatus = branch.step === 'reasoning' && !isReasoningDone
                                    ? 'active'
                                    : (isReasoningDone || branch.step === 'verification' || phase === 'decision' || phase === 'memory' ? 'done' : 'pending');
                                return (
                                    <Node
                                        label="Refiner"
                                        subLabel="Reasoning Agent"
                                        status={reasoningStatus}
                                        color="var(--neon-purple)"
                                        onClick={onNodeSelect}
                                        data={{
                                            type: 'branch',
                                            title: `Refiner (Branch ${i + 1})`,
                                            color: 'var(--neon-purple)',
                                            content: getNodeContent(
                                                reasoningStatus,
                                                reasoningData,
                                                branch.logs
                                            )
                                        }}
                                    />
                                );
                            })()}

                            <Connector vertical active={branch.step === 'verification'} height={20} />

                            {/* Verification Node */}
                            {(() => {
                                const verificationData = branch.nodes?.verification;
                                const isVerificationDone = verificationData && verificationData.type !== 'processing';
                                const verificationStatus = branch.step === 'verification' && !isVerificationDone
                                    ? 'active'
                                    : (isVerificationDone || phase === 'decision' || phase === 'memory' ? 'done' : 'pending');
                                return (
                                    <Node
                                        label="Red Team"
                                        subLabel="Verification Agent"
                                        status={verificationStatus}
                                        color="var(--neon-red)"
                                        onClick={onNodeSelect}
                                        data={{
                                            type: 'branch',
                                            title: `Verification (Branch ${i + 1})`,
                                            color: 'var(--neon-red)',
                                            content: getNodeContent(
                                                verificationStatus,
                                                verificationData,
                                                branch.logs
                                            )
                                        }}
                                    />
                                );
                            })()}
                        </div>
                    )
                })}
            </div>

            <Connector vertical active={phase === 'decision'} />

            {/* 3. Decision Layer */}
            {/* 3. Decision Layer */}
            {(() => {
                const isDecisionMade = decision?.decision === true || decision?.result === true;
                const branchIndex = decision?.best_branch_index ?? decision?.bestBranch;
                const branchDisplay = (branchIndex !== undefined && branchIndex !== null && !isNaN(branchIndex))
                    ? branchIndex + 1
                    : '?';

                let subLabel = "Evaluating...";
                let nodeColor = "var(--neon-cyan)";

                if (decision) {
                    if (isDecisionMade) {
                        subLabel = `Valid Solution Found (Branch ${branchDisplay})`;
                        nodeColor = "var(--neon-green)";
                    } else {
                        subLabel = "Solution Rejected - Iterating...";
                        nodeColor = "var(--neon-red)"; // or orange? Red implies failure which is technically true
                    }
                }

                return (
                    <Node
                        label="Decision Maker"
                        subLabel={subLabel}
                        status={phase === 'decision' ? 'active' : (decision ? 'done' : 'pending')}
                        color={nodeColor}
                        onClick={onNodeSelect}
                        data={{
                            type: 'decision',
                            title: 'Decision Maker',
                            color: nodeColor,
                            content: getNodeContent(
                                phase === 'decision' ? 'active' : (decision ? 'done' : 'pending'),
                                decision,
                                // Decision logs not explicitly tracked in state root, passing empty for now
                                []
                            )
                        }}
                    />
                );
            })()}

            {/* 4. Memory Layer (Conditional) */}
            {state.phase === 'memory' && (
                <>
                    <Connector vertical active={true} />
                    <Node
                        label="Memory Agent"
                        subLabel="Updating Constraints"
                        status="active"
                        color="var(--neon-purple)"
                    />
                </>
            )}

        </div>
    );
};

export default GraphVisualizer;
