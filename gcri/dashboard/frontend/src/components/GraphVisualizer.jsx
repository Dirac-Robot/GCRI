import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const Node = ({ label, subLabel, status, color, icon }) => {
    const isPending = status === 'pending';
    const isActive = status === 'active';

    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{
                opacity: isPending ? 0.3 : 1,
                scale: 1,
                borderColor: isActive ? color : 'rgba(255,255,255,0.1)',
                boxShadow: isActive ? `0 0 20px ${color}40` : 'none'
            }}
            className={`
                relative w-48 p-4 rounded-lg border bg-[rgba(10,10,10,0.8)] backdrop-blur-md
                flex flex-col items-center justify-center gap-2
                transition-colors duration-500
            `}
            style={{
                borderWidth: '1px',
                borderColor: isActive ? color : 'rgba(255,255,255,0.1)'
            }}
        >
            {isActive && (
                <motion.div
                    layoutId="active-glow"
                    className="absolute inset-0 rounded-lg pointer-events-none"
                    style={{ border: `2px solid ${color}`, boxShadow: `0 0 15px ${color}` }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
            )}

            <div className="text-xs font-bold tracking-widest opacity-50 uppercase" style={{ color: color }}>
                {label}
            </div>
            {subLabel && <div className="text-sm font-medium text-center text-white">{subLabel}</div>}
        </motion.div>
    );
};

const Connector = ({ active, vertical = false, height = 40 }) => (
    <div className={`flex items-center justify-center ${vertical ? 'w-full' : 'h-full'}`} style={{ height: vertical ? height : 'auto' }}>
        <motion.div
            className={`${vertical ? 'w-[2px] h-full' : 'h-[2px] w-full'} bg-[var(--glass-border)] relative overflow-hidden`}
        >
            {active && (
                <motion.div
                    className={`absolute inset-0 bg-[var(--neon-cyan)]`}
                    initial={{ x: vertical ? 0 : '-100%', y: vertical ? '-100%' : 0 }}
                    animate={{ x: 0, y: 0 }}
                    transition={{ duration: 0.5 }}
                />
            )}
        </motion.div>
    </div>
);

const GraphVisualizer = ({ state }) => {
    const { phase, branches, decision } = state;

    return (
        <div className="w-full h-full flex flex-col items-center justify-center p-10 overflow-auto">

            {/* 1. Strategy Layer */}
            <Node
                label="Strategy Generator"
                subLabel={`Strategies Generated: ${state.strategies.length}`}
                status={phase === 'strategy' ? 'active' : 'done'}
                color="var(--neon-cyan)"
            />

            <Connector vertical active={branches.some(b => b.status === 'active')} />

            {/* 2. Branch Layer (Parallel) */}
            <div className="flex gap-8 w-full justify-center">
                {branches.map((branch, i) => (
                    <div key={i} className="flex flex-col items-center gap-4 flex-1 max-w-[250px]">
                        {/* Branch Top Connector */}
                        <div className="w-[2px] h-8 bg-[var(--glass-border)] relative">
                            {/* Static line for now, dynamic later */}
                        </div>

                        {/* Hypothesis Node */}
                        <Node
                            label={`Branch ${i + 1}`}
                            subLabel="Hypothesis Agent"
                            status={branch.step === 'hypothesis' ? 'active' : (branch.step !== 'idle' ? 'done' : 'pending')}
                            color="var(--neon-green)"
                        />

                        <Connector vertical active={branch.step === 'reasoning' || branch.step === 'verification'} height={20} />

                        {/* Reasoning Node */}
                        <Node
                            label="Refiner"
                            subLabel="Reasoning Agent"
                            status={branch.step === 'reasoning' ? 'active' : (branch.step === 'verification' ? 'done' : 'pending')}
                            color="var(--neon-purple)"
                        />

                        <Connector vertical active={branch.step === 'verification'} height={20} />

                        {/* Verification Node */}
                        <Node
                            label="Red Team"
                            subLabel="Verification Agent"
                            status={branch.step === 'verification' ? 'active' : 'pending'}
                            color="var(--neon-red)"
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
