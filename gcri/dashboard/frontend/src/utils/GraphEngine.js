export class GraphEngine {
    constructor() {
        this.reset();
    }

    reset() {
        this.state = {
            iteration: 0,
            phase: 'idle', // idle, strategy, execution, decision, memory
            task: '',
            strategies: [], // Array of strings or objects
            branches: [
                { id: 0, step: 'idle', status: 'pending', logs: [] },
                { id: 1, step: 'idle', status: 'pending', logs: [] },
                { id: 2, step: 'idle', status: 'pending', logs: [] }
            ],
            decision: null,
            memory: []
        };
    }

    process(log) {
        // We handle both nested and flat structures for robustness.
        if (!log) return { ...this.state };

        const msg = log.record?.message || log.message || "";

        if (!msg) return { ...this.state };

        // 1. Iteration Start / Task
        if (msg.includes('Starting Iteration')) {
            const match = msg.match(/Iteration (\d+)/);
            if (match) this.state.iteration = parseInt(match[1]);
            this.state.phase = 'strategy';
            // Reset branches for new iteration
            this.state.branches = this.state.branches.map(b => ({ ...b, step: 'idle', status: 'pending', logs: [] }));
            this.state.decision = null;
        }

        // 2. Strategy Generation
        if (msg.includes('Sampled strategy')) {
            this.state.phase = 'strategy';
            // "Sampled strategy #1: ..."
            const match = msg.match(/strategy #(\d+): (.*)/);
            if (match) {
                const idx = parseInt(match[1]) - 1;
                this.state.strategies[idx] = match[2];
            }
        }

        // 3. Branch Execution & Log Routing
        // Use heuristics to attribute logs to specific branches for the detail view
        let branchIdx = -1;
        const branchMatch = msg.match(/Branch #(\d+)|hypothesis #(\d+)|strategy #(\d+)/i);
        if (branchMatch) {
            // logical index is 0-based
            branchIdx = parseInt(branchMatch[1] || branchMatch[2] || branchMatch[3]) - 1;
        }

        if (branchIdx >= 0 && this.state.branches[branchIdx]) {
            this.state.branches[branchIdx].logs.push(msg); // Store log in branch history
        }

        // Logic for stepping phase
        if (msg.includes('Request sampling hypothesis')) {
            this.state.phase = 'execution';
            if (branchIdx >= 0) this._updateBranch(branchIdx, 'hypothesis');
        }
        else if (msg.includes('Request reasoning and refining')) {
            if (branchIdx >= 0) this._updateBranch(branchIdx, 'reasoning');
        }
        else if (msg.includes('Request verifying')) {
            if (branchIdx >= 0) this._updateBranch(branchIdx, 'verification');
        }

        // 4. Decision
        if (msg.includes('Request generating final decision')) {
            this.state.phase = 'decision';
        }
        if (msg.includes('Selected Best Branch Index')) {
            const match = msg.match(/Index: (\d+)/);
            if (match) {
                this.state.decision = {
                    result: true,
                    bestBranch: parseInt(match[1])
                };
            }
        }

        // 5. Memory
        if (msg.includes('Memory saved')) {
            this.state.phase = 'memory';
        }

        return { ...this.state }; // Return immutable copy for React
    }

    _updateBranch(idx, step) {
        if (this.state.branches[idx]) {
            this.state.branches[idx].step = step;
            this.state.branches[idx].status = 'active';
        }
    }
}
