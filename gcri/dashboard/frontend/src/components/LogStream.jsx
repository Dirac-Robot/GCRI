import React, { useEffect, useRef } from 'react';

const LogStream = ({ logs }) => {
    const logsEndRef = useRef(null);

    const getLogData = (log) => {
        // Handle Loguru serialized structure
        const record = log.record || log;
        const message = record.message || "";
        const level = record.level?.name || "INFO";
        let timestampStr;
        if (record.time?.repr) {
            timestampStr = new Date(record.time.repr).toLocaleTimeString();
        } else if (record.time?.timestamp) {
            timestampStr = new Date(record.time.timestamp * 1000).toLocaleTimeString();
        } else {
            timestampStr = new Date().toLocaleTimeString();
        }
        return { message, level, timestamp: timestampStr };
    };

    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="flex-1 overflow-auto p-4 font-mono text-xs space-y-1">
            {logs.map((log, i) => {
                const { message, level, timestamp } = getLogData(log);
                let color = 'text-gray-400';
                if (level === 'ERROR') color = 'text-[var(--neon-red)]';
                if (level === 'WARNING') color = 'text-yellow-400';
                if (level === 'SUCCESS') color = 'text-[var(--neon-green)]';

                return (
                    <div key={i} className="flex gap-2 hover:bg-[rgba(255,255,255,0.05)] px-1 rounded">
                        <span className="text-[var(--text-secondary)] opacity-50 shrink-0">[{timestamp}]</span>
                        <span className={`font-bold shrink-0 w-16 ${color}`}>{level}</span>
                        <span className="text-gray-300 break-all">{message}</span>
                    </div>
                );
            })}
            <div ref={logsEndRef} />
        </div>
    );
};

export default LogStream;
