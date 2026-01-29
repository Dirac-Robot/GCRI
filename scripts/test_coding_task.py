"""
Test GCRI with a coding task to verify intelligent file merging works correctly.
"""
import sys
sys.path.insert(0, '.')

from gcri.config import scope as gcri_scope
from gcri.graphs.gcri_unit import GCRI


@gcri_scope.observe('minimal_test', priority=0, default=True)
def minimal_test(config):
    """Minimal test configuration for coding task."""
    config.protocols.max_iterations = 1
    config.branches_generator_type = 'minimal'
    config.agents.branches = [
        dict(hypothesis=dict(model_id='gpt-4.1', parameters=dict(max_tokens=8192))),
        dict(hypothesis=dict(model_id='gpt-4.1', parameters=dict(max_tokens=8192))),
    ]
    return config


@gcri_scope
def main(config):
    # Hard coding problem: LeetCode Hard Level
    coding_task = """
Problem: Sliding Window Maximum (LeetCode #239 - Hard)

You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:
Input: nums = [1], k = 1
Output: [1]

Constraints:
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- 1 <= k <= nums.length

Requirements:
1. Implement solution in Python with O(n) time complexity (use deque)
2. Create a file called 'solution.py' with a Solution class
3. Include comprehensive test cases
4. The solution must pass all test cases

Implement the solution and verify it works.
"""

    print('=' * 60)
    print('GCRI Coding Test: Sliding Window Maximum')
    print('=' * 60)
    print(f'Generator Type: {config.branches_generator_type}')
    print(f'Num Branches: {len(config.agents.branches)}')
    print('=' * 60)

    gcri = GCRI(config)
    result = gcri(coding_task)

    print('\n' + '=' * 60)
    print('Final Result:')
    print('=' * 60)
    print(f"Decision: {result.get('decision')}")
    print(f"Best Branch: {result.get('best_branch')}")
    
    # Check if solution.py was created
    output = result.get('output', '')
    print(f"\nOutput Preview: {output[:1000]}...")

    print('\n✅ Coding test completed.')


if __name__ == '__main__':
    main()
