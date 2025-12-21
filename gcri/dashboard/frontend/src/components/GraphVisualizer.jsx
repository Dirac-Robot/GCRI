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

            <div className="text-xs font-bold tracking-widest uppercase text-shadow-sm" style={{ color: color, textShadow: `0 0 10px ${color}` }}>
                {label}
            </div>
            {subLabel && <div className="text-sm font-bold text-center text-white">{subLabel}</div>}
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

    return (
        <div className="w-full h-full flex flex-col items-center justify-center p-10 overflow-auto">

            {/* 1. Strategy Layer */}
            <Node
                label="Strategy Generator"
                subLabel={`Strategies Generated: ${state.strategies.length}`}
                status={phase === 'strategy' ? 'active' : 'done'}
                color="var(--neon-cyan)"
                onClick={onNodeSelect}
                data={{ type: 'strategy', title: 'Strategy Generator', content: state.strategies.join('\n\n') }}
            />

            <Connector vertical active={branches.some(b => b.status === 'active')} />

            {/* 2. Branch Layer (Parallel) */}
            <div className="flex gap-8 w-full justify-center">
                {branches.map((branch, i) => (
                    <div key={i} className="flex flex-col items-center gap-4 flex-1 max-w-[250px]">
                        {/* Branch Top Connector */}
                        <div className="w-[2px] h-8 bg-[#333] relative">
                            {/* Static line for now, dynamic later */}
                        </div>

                        {/* Hypothesis Node */}
                        <Node
                            label={`Branch ${i + 1}`}
                            subLabel="Hypothesis Agent"
                            status={branch.step === 'hypothesis' ? 'active' : (branch.step !== 'idle' ? 'done' : 'pending')}
                            color="var(--neon-green)"
                            onClick={onNodeSelect}
                            data={{ type: 'branch', title: `Hypothesis (Branch ${i + 1})`, content: branch.logs.join('\n') }}
                        />

                        <Connector vertical active={branch.step === 'reasoning' || branch.step === 'verification'} height={20} />

                        {/* Reasoning Node */}
                        <Node
                            label="Refiner"
                            subLabel="Reasoning Agent"
                            status={branch.step === 'reasoning' ? 'active' : (branch.step === 'verification' ? 'done' : 'pending')}
                            color="var(--neon-purple)"
                            onClick={onNodeSelect}
                            data={{ type: 'branch', title: `Refiner (Branch ${i + 1})`, content: branch.logs.join('\n') }}
                        />

                        <Connector vertical active={branch.step === 'verification'} height={20} />

                        {/* Verification Node */}
                        <Node
                            label="Red Team"
                            subLabel="Verification Agent"
                            status={branch.step === 'verification' ? 'active' : 'pending'}
                            color="var(--neon-red)"
                            onClick={onNodeSelect}
                            data={{ type: 'branch', title: `Verification (Branch ${i + 1})`, content: branch.logs.join('\n') }}
                        />
                    </div>
                ))}
            </div>

            <Connector vertical active={phase === 'decision'} />

            {/* 3. Decision Layer */}
            <Node
                label="Decision Maker"
                subLabel={decision ? `Valid Solution Found (Branch ${decision.bestBranch + 1})` : "Evaluating..."}
                status={phase === 'decision' ? 'active' : (decision ? 'done' : 'pending')}
                color={decision?.result ? 'var(--neon-green)' : 'var(--neon-cyan)'}
                onClick={onNodeSelect}
                data={{ type: 'decision', title: 'Decision Maker', content: decision ? JSON.stringify(decision, null, 2) : 'Pending...' }}
            />

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
