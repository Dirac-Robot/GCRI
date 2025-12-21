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
        // Loguru serialized JSON has structure: { text: "...", record: { message: "...", ... } }
        // We handle both nested and flat structures for robustness.
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

        // 3. Branch Execution (Hypothesis -> Reasoning -> Verification)
        // Log pattern: "Iter #1 | Request sampling hypothesis for strategy #1..."
        if (msg.includes('Request sampling hypothesis')) {
            this.state.phase = 'execution';
            const match = msg.match(/strategy #(\d+)/);
            if (match) this._updateBranch(match[1], 'hypothesis');
        }
        else if (msg.includes('Request reasoning and refining')) {
            const match = msg.match(/hypothesis #(\d+)/);
            if (match) this._updateBranch(match[1], 'reasoning');
        }
        else if (msg.includes('Request verifying')) {
            const match = msg.match(/hypothesis #(\d+)/);
            if (match) this._updateBranch(match[1], 'verification');
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

    _updateBranch(idxStr, step) {
        const idx = parseInt(idxStr) - 1;
        if (this.state.branches[idx]) {
            this.state.branches[idx].step = step;
            this.state.branches[idx].status = 'active';
        }
    }
}
