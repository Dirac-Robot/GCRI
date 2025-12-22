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
            return (
                <div className="pl-4 border-l border-[var(--glass-border)] my-1">
                    {value.map((item, i) => (
                        <div key={i} className="mb-2 last:mb-0">
                            <span className="text-gray-500 text-xs mr-2">[{i}]</span>
                            {renderValue(item)}
                        </div>
                    ))}
                </div>
            );
        }
        if (typeof value === 'object') {
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
