from typing import List
import bisect

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        tails = []
        for num in nums:
            # Find the index in tails where num should be placed
            pos = bisect.bisect_left(tails, num)
            # If pos is equal to length of tails, append num
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        return len(tails)


# Comprehensive tests inside the file
if __name__ == '__main__':
    sol = Solution()
    
    # Edge cases
    assert sol.lengthOfLIS([]) == 0, "Failed on empty list"
    assert sol.lengthOfLIS([10]) == 1, "Failed on single element"
    assert sol.lengthOfLIS([7,7,7,7]) == 1, "Failed on all identical elements"
    assert sol.lengthOfLIS([5,4,3,2,1]) == 1, "Failed on strictly decreasing"

    # Typical cases
    assert sol.lengthOfLIS([10,9,2,5,3,7,101,18]) == 4, "Failed on example case"
    assert sol.lengthOfLIS([0,1,0,3,2,3]) == 4, "Failed on mixed increase"
    assert sol.lengthOfLIS([7,1,5,1,6,2,3,4]) == 4, "Failed on multiple subsequences"

    print("All tests passed.")

