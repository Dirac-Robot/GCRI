"""
Test script for MinimalThinkGenerator via full GCRI pipeline.
Uses no_reasoning mode and minimal generator.
"""
import os
import sys

# Set PYTHONPATH
sys.path.insert(0, '/Users/dirac_on/Library/Mobile Documents/com~apple~CloudDocs/Projects/GCRI')
os.chdir('/Users/dirac_on/Library/Mobile Documents/com~apple~CloudDocs/Projects/GCRI')

from dotenv import load_dotenv
load_dotenv()

from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI


def main():
    # Create separate test workspace to avoid cluttering main project
    test_workspace = '/tmp/gcri_test_workspace'
    os.makedirs(test_workspace, exist_ok=True)

    # Assign views and literals
    scope.assign('no_reasoning')
    scope.assign('num_branches=2')
    scope.assign('protocols.max_iterations=3')  # 3 iterations for harder problem
    scope.apply()
    config = scope.config
    config.project_dir = test_workspace  # Use temp workspace
    config.branches_generator_type = 'low'

    print('='*60)
    print('GCRI Hard Problem Test with MinimalThinkGenerator')
    print('='*60)
    print(f'Generator Type: {config.branches_generator_type}')
    print(f'Num Branches: {config.num_branches}')
    print(f'Max Iterations: {config.protocols.max_iterations}')
    print('='*60)

    # Create GCRI instance
    gcri = GCRI(config)

    # GCRI DEEP RESEARCH TEST: Open-ended research question with no clear answer
    # Challenge: Multiple valid perspectives exist. Must explore and synthesize.
    # Goal: Test if branches produce diverse viewpoints and Decision synthesizes best report.
    task = """
Research Question: "Is the 'Alignment Tax' in AI Development Inevitable?"

**Context:**
The "Alignment Tax" refers to the potential performance cost of making AI systems safe and 
aligned with human values. Some researchers argue safety measures inherently reduce capability,
while others claim alignment and capability can be synergistic.

**Your Task:**
Write a 1-page research brief (saved as `report.md`) that:

1. **Defines the problem** - What is the alignment tax? Where does it manifest?

2. **Presents perspectives:**
   - Pessimistic: Alignment necessarily trades off against capability
   - Optimistic: Alignment and capability are complementary
   - Pragmatic: Depends on technique and context

3. **Analyzes evidence** - What supports each view?

4. **Takes a defended position** - Which view is most defensible and why?

5. **Proposes recommendations** - What should AI labs do?

**Deliverables:**
1. `report.md` - The research brief (800-1200 words)
2. `sources.md` - Key references/concepts cited

NOTE: No "correct" answer exists. Quality of reasoning matters.
"""

    print(f'\nTask: {task}')
    print('='*60)
    print('Starting GCRI pipeline...\n')

    # Run full GCRI pipeline (auto commit to save code)
    result = gcri(task, commit_mode='auto')

    print('\n' + '='*60)
    print('Final Result:')
    print('='*60)
    print(f"Decision: {result.get('decision')}")
    print(f"Best Branch: {result.get('best_branch_index')}")
    if result.get('final_output'):
        output = result['final_output']
        if isinstance(output, str):
            print(f"Output Preview: {output[:500]}...")
        else:
            print(f"Output: {output}")

    print('\n✅ Test completed.')


if __name__ == '__main__':
    main()
