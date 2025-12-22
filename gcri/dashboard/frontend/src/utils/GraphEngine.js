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
            memory: [],
            history: []
        };
    }

    process(log) {
        // We handle both nested and flat structures for robustness.
        if (!log) return { ...this.state };

        const msg = log.record?.message || log.message || "";
        const extra = log.record?.extra || log.extra || {};

        if (extra.ui_event === 'phase_change') {
            this.state.phase = extra.phase;
            if (extra.iteration !== undefined) {
                // Iteration update logic
                const newIter = extra.iteration;
                if (newIter > this.state.iteration) {
                    if (this.state.iteration > 0 || this.state.strategies.length > 0) {
                        const archivedState = JSON.parse(JSON.stringify(this.state));
                        delete archivedState.history;
                        this.state.history.push(archivedState);
                    }
                    // Reset branch states explicitly
                    this.state.branches = this.state.branches.map(b => ({ ...b, step: 'idle', status: 'pending', logs: [], nodes: {} }));
                    this.state.decision = null;
                    this.state.strategyData = null;
                }
                this.state.iteration = newIter;
            }
        }

        if (extra.ui_event === 'node_update') {
            const { node, branch, data } = extra;

            if (node === 'strategy') {
                // Phase handled by phase_change event now (or explicit set above if missed, but phase_change is primary)
                if (data.strategies) {
                    this.state.strategies = data.strategies;
                    this.state.strategyData = data;
                } else if (Array.isArray(data)) {
                    this.state.strategies = data;
                    this.state.strategyData = { strategies: data };
                }
            } else if (typeof branch === 'number') {
                if (!this.state.branches[branch]) {
                    this.state.branches[branch] = { id: branch, step: 'idle', status: 'pending', logs: [], nodes: {} };
                }
                if (!this.state.branches[branch].nodes) {
                    this.state.branches[branch].nodes = {};
                }
                this.state.branches[branch].nodes[node] = data;

                // Keep minimal step inference for branch visualization status if needed, 
                // but phase is global.
                this.state.branches[branch].step = node;
                this.state.branches[branch].status = 'active';

            } else if (node === 'decision') {
                this.state.decision = { ...this.state.decision, ...data };
            } else if (node === 'memory') {
                this.state.memory = Array.isArray(data) ? data : [];
            }
        }

        // Regex logic removed as requested for phase control.
        // We only parse msg for Branch Logs routing now.

        let branchIdx = -1;
        const branchMatch = msg.match(/Branch #(\d+)|hypothesis #(\d+)|strategy #(\d+)/i);
        if (branchMatch) {
            branchIdx = parseInt(branchMatch[1] || branchMatch[2] || branchMatch[3]) - 1;
        }

        if (branchIdx >= 0 && this.state.branches[branchIdx]) {
            this.state.branches[branchIdx].logs.push(msg);
        }

        return { ...this.state };
    }

    _updateBranch(idx, step) {
        if (this.state.branches[idx]) {
            this.state.branches[idx].step = step;
            this.state.branches[idx].status = 'active';
        }
    }
}
