import React from 'react';
import { Activity, Clock, AlertCircle } from 'lucide-react';

const StructuredView = ({ data }) => {
    if (!data) return <div className="text-gray-500 italic">No data available</div>;

    // Handle specific status types
    if (data.type === 'pending') {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-gray-500 animate-pulse">
                <Clock size={48} className="mb-4 opacity-50" />
                <div className="text-lg font-bold tracking-widest">WAITING FOR INPUT...</div>
                <div className="text-xs mt-2">Dependent on previous steps</div>
            </div>
        );
    }

    if (data.type === 'processing') {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-[var(--neon-cyan)] animate-pulse">
                <Activity size={48} className="mb-4" />
                <div className="text-lg font-bold tracking-widest">AGENT WORKING...</div>
                <div className="text-xs mt-2">Processing task logic</div>
            </div>
        );
    }

    // Check if object is a Strategy (has name, description, hints)
    const isStrategy = (obj) => {
        return obj && typeof obj === 'object' && 'name' in obj && 'description' in obj;
    };

    // Render a Strategy card
    const renderStrategy = (strategy, index) => (
        <div key={index} className="bg-[rgba(255,255,255,0.03)] rounded-lg border border-[var(--glass-border)] overflow-hidden mb-3 last:mb-0">
            {/* Strategy Header */}
            <div className="bg-[rgba(0,255,255,0.1)] px-4 py-2 border-b border-[var(--glass-border)]">
                <div className="flex items-center gap-2">
                    <span className="text-[var(--neon-purple)] font-bold text-xs">#{index + 1}</span>
                    <span className="text-white font-bold text-sm">{strategy.name}</span>
                </div>
            </div>
            {/* Strategy Body */}
            <div className="p-4 space-y-3">
                {/* Description */}
                <div>
                    <div className="text-[var(--neon-cyan)] text-xs font-bold uppercase mb-1">Description</div>
                    <div className="text-gray-300 text-sm leading-relaxed">{strategy.description}</div>
                </div>
                {/* Hints */}
                {strategy.hints && strategy.hints.length > 0 && (
                    <div>
                        <div className="text-[var(--neon-cyan)] text-xs font-bold uppercase mb-2">Hints</div>
                        <ul className="space-y-1">
                            {strategy.hints.map((hint, i) => (
                                <li key={i} className="flex items-start gap-2 text-xs text-gray-400">
                                    <span className="text-[var(--neon-green)] mt-0.5">•</span>
                                    <span>{hint}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                {/* Feedback Reflection */}
                {strategy.feedback_reflection && (
                    <div className="text-xs text-gray-500 italic border-t border-[var(--glass-border)] pt-2 mt-2">
                        Feedback: {strategy.feedback_reflection}
                    </div>
                )}
            </div>
        </div>
    );

    // Recursive renderer for objects/arrays
    const renderValue = (value) => {
        if (typeof value === 'string') {
            // Check if it looks like a URL or file path
            if (value.startsWith('http') || value.startsWith('/')) {
                return <span className="text-blue-400 break-all">{value}</span>;
            }
            return <span className="text-gray-300 break-words whitespace-pre-wrap">{value}</span>;
        }
        if (typeof value === 'number') return <span className="text-[var(--neon-purple)]">{value}</span>;
        if (typeof value === 'boolean') return <span className={value ? 'text-[var(--neon-green)]' : 'text-[var(--neon-red)]'}>{value.toString()}</span>;
        if (value === null) return <span className="text-gray-600 italic">null</span>;
        if (Array.isArray(value)) {
            if (value.length === 0) return <span className="text-gray-600">[]</span>;
            // Check if array contains Strategy objects
            if (value.every(isStrategy)) {
                return (
                    <div className="space-y-0">
                        {value.map((item, i) => renderStrategy(item, i))}
                    </div>
                );
            }
            return (
                <div className="pl-4 border-l border-[var(--glass-border)] my-1">
                    {value.map((item, i) => (
                        <div key={i} className="mb-2 last:mb-0">
                            <span className="text-gray-500 text-xs mr-2">[{i + 1}]</span>
                            {renderValue(item)}
                        </div>
                    ))}
                </div>
            );
        }
        if (typeof value === 'object') {
            // Check if it's a Strategy object
            if (isStrategy(value)) {
                return renderStrategy(value, 0);
            }
            if (Object.keys(value).length === 0) return <span className="text-gray-600">{"{}"}</span>;
            return (
                <div className="pl-4 border-l border-[var(--glass-border)] my-1">
                    {Object.entries(value).map(([key, val]) => (
                        <div key={key} className="mb-2 last:mb-0">
                            <span className="text-[var(--neon-cyan)] text-xs font-bold mr-2 uppercase">{key}:</span>
                            {renderValue(val)}
                        </div>
                    ))}
                </div>
            );
        }
        return String(value);
    };

    // If root data is an object, render its keys at top level
    if (typeof data === 'object' && !Array.isArray(data)) {
        return (
            <div className="space-y-4">
                {Object.entries(data).map(([key, val]) => (
                    <div key={key} className="bg-[rgba(255,255,255,0.02)] p-3 rounded border border-[var(--glass-border)]">
                        <div className="text-[var(--neon-cyan)] font-bold text-sm mb-2 uppercase tracking-wider border-b border-[var(--glass-border)] pb-1 inline-block">
                            {key.replace(/_/g, ' ')}
                        </div>
                        <div className="text-sm font-mono mt-1">
                            {renderValue(val)}
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    return renderValue(data);
};

export default StructuredView;
