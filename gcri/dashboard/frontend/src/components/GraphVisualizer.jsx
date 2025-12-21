import React from 'react';

const GraphVisualizer = ({ activeNode }) => {
    // Nodes data with positions and styling
    const nodes = [
        { id: 'start', x: 400, y: 50, label: 'Start', color: 'var(--text-secondary)' },
        { id: 'strategy', x: 400, y: 150, label: 'Strategy\nGenerator', color: 'var(--neon-cyan)' },

        // Branch Loop Area
        { id: 'hypothesis', x: 200, y: 300, label: 'Hypothesis\n(Coder)', color: 'var(--neon-green)' },
        { id: 'reasoning', x: 400, y: 300, label: 'Reasoning\n(Refiner)', color: 'var(--neon-purple)' },
        { id: 'verification', x: 600, y: 300, label: 'Verification\n(Red Team)', color: 'var(--neon-red)' },

        { id: 'decision', x: 400, y: 450, label: 'Decision\nMaker', color: 'var(--neon-cyan)' },
    ];

    const edges = [
        { from: 'start', to: 'strategy' },
        { from: 'strategy', to: 'hypothesis' },
        { from: 'strategy', to: 'reasoning' },
        { from: 'strategy', to: 'verification' },
        { from: 'hypothesis', to: 'decision' },
        { from: 'reasoning', to: 'decision' },
        { from: 'verification', to: 'decision' },
        // Inner loop connections (conceptual)
        { from: 'hypothesis', to: 'reasoning', dashed: true },
        { from: 'reasoning', to: 'verification', dashed: true },
    ];

    const isActive = (nodeId) => activeNode === nodeId || (activeNode === 'start' && nodeId === 'start');

    return (
        <svg width="100%" height="600" viewBox="0 0 800 600" className="w-[800px] h-[600px] select-none">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="var(--glass-border)" />
                </marker>
                <marker id="arrowhead-active" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="var(--neon-cyan)" />
                </marker>
            </defs>

            {/* Edges */}
            {edges.map((edge, i) => {
                const source = nodes.find(n => n.id === edge.from);
                const target = nodes.find(n => n.id === edge.to);
                const active = isActive(edge.from) && isActive(edge.to); // Simplistic edge lighting
                return (
                    <path
                        key={i}
                        d={`M${source.x},${source.y} L${target.x},${target.y}`}
                        stroke={active ? 'var(--neon-cyan)' : 'var(--glass-border)'}
                        strokeWidth="2"
                        strokeDasharray={edge.dashed ? "5,5" : ""}
                        markerEnd={active ? "url(#arrowhead-active)" : "url(#arrowhead)"}
                        className="transition-all duration-500"
                    />
                );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
                const active = isActive(node.id);
                const color = active ? node.color : 'rgba(255,255,255,0.1)';
                const stroke = active ? node.color : 'rgba(255,255,255,0.2)';

                return (
                    <g key={node.id} className="transition-all duration-500">
                        <circle
                            cx={node.x}
                            cy={node.y}
                            r={40}
                            fill={active ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.3)'}
                            stroke={stroke}
                            strokeWidth={active ? 3 : 1}
                            filter={active ? 'url(#glow)' : ''}
                            className="transition-all duration-500"
                        />
                        {active && (
                            <circle cx={node.x} cy={node.y} r={46} stroke={color} strokeWidth="1" opacity="0.5" className="animate-ping" />
                        )}
                        <text
                            x={node.x}
                            y={node.y}
                            textAnchor="middle"
                            dy=".3em"
                            fill={active ? '#fff' : 'rgba(255,255,255,0.4)'}
                            fontSize="12"
                            fontWeight={active ? "bold" : "normal"}
                            className="pointer-events-none"
                        >
                            {node.label.split('\n').map((line, i) => (
                                <tspan x={node.x} dy={i === 0 ? '-0.5em' : '1.2em'} key={i}>{line}</tspan>
                            ))}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
};

export default GraphVisualizer;
