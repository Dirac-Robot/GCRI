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

    const [autoScroll, setAutoScroll] = React.useState(true);

    const scrollToBottom = () => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        if (autoScroll) {
            scrollToBottom();
        }
    }, [logs, autoScroll]);

    const handleScroll = (e) => {
        const { scrollTop, scrollHeight, clientHeight } = e.target;
        // If user scrolls up (distance from bottom > 50px), disable auto-scroll
        const atBottom = scrollHeight - scrollTop - clientHeight < 50;
        setAutoScroll(atBottom);
    };

    return (
        <div
            className="h-full overflow-y-auto p-4 font-mono text-xs space-y-1 scrollbar-thin scrollbar-thumb-[var(--primary)] scrollbar-track-transparent"
            onScroll={handleScroll}
        >
            {logs.map((log, i) => {
                const { message, level, timestamp } = getLogData(log);
                let color = 'text-gray-400';
                if (level === 'ERROR') color = 'text-[var(--neon-red)]';
                if (level === 'WARNING') color = 'text-yellow-400';
                if (level === 'SUCCESS') color = 'text-[var(--neon-green)]';

                const formatMessage = (msg) => {
                    // Simple parser for Markdown links [text](url) and raw URLs
                    if (!msg) return "";
                    const parts = [];
                    let lastIndex = 0;

                    // Regex for Markdown links: [Title](URL)
                    const mdLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g;
                    let match;

                    while ((match = mdLinkRegex.exec(msg)) !== null) {
                        if (match.index > lastIndex) {
                            parts.push(msg.substring(lastIndex, match.index));
                        }
                        parts.push(
                            <a
                                key={match.index}
                                href={match[2]}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[var(--neon-cyan)] underline hover:text-white transition-colors"
                            >
                                {match[1]}
                            </a>
                        );
                        lastIndex = match.index + match[0].length;
                    }

                    // Check for raw URLs if we haven't consumed everything
                    // (Simplified: Only doing MD parsing for citations as requested, to avoid messy auto-linking of everything)
                    if (lastIndex < msg.length) {
                        parts.push(msg.substring(lastIndex));
                    }

                    return parts.length > 0 ? parts : msg;
                };

                return (
                    <div
                        className="flex gap-2 hover:bg-[rgba(255,255,255,0.05)] px-1 rounded"
                    >
                        <span className="text-[var(--text-secondary)] opacity-50 shrink-0">[{timestamp}]</span>
                        <span className={`font-bold shrink-0 w-16 ${color}`}>{level}</span>
                        <span className="text-gray-300 break-all">{formatMessage(message)}</span>
                    </div>
                );
            })}
            <div ref={logsEndRef} />
        </div>
    );
};

export default LogStream;
