import React, { useEffect, useRef } from 'react';

const LogStream = ({ logs }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    const getLevelColor = (level) => {
        switch (level?.toUpperCase()) {
            case 'INFO': return 'text-[var(--neon-cyan)]';
            case 'WARNING': return 'text-yellow-400';
            case 'ERROR': return 'text-[var(--neon-red)]';
            case 'DEBUG': return 'text-[var(--text-secondary)]';
            default: return 'text-white';
        }
    };

    return (
        <div className="h-full overflow-y-auto p-4 font-mono text-xs space-y-2">
            {logs.map((log, i) => (
                <div key={i} className="break-all border-b border-[rgba(255,255,255,0.02)] pb-1 last:border-0 hover:bg-[rgba(255,255,255,0.03)] transition-colors">
                    <span className="text-[var(--text-secondary)] mr-2 select-none">
                        [{new Date(log.time.timestamp * 1000).toLocaleTimeString()}]
                    </span>
                    <span className={`font-bold mr-2 ${getLevelColor(log.level.name)}`}>
                        {log.level.name}
                    </span>
                    <span className="text-gray-300">
                        {log.message}
                    </span>
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    );
};

export default LogStream;
