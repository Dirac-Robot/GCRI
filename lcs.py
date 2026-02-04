class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        if m == 0 or n == 0:
            return 0
        # Initialize DP table with zeros
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]


# Comprehensive tests
import unittest

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()

    def test_empty_strings(self):
        self.assertEqual(self.solution.longestCommonSubsequence("", ""), 0)
        self.assertEqual(self.solution.longestCommonSubsequence("", "a"), 0)
        self.assertEqual(self.solution.longestCommonSubsequence("a", ""), 0)

    def test_single_char(self):
        self.assertEqual(self.solution.longestCommonSubsequence("a", "a"), 1)
        self.assertEqual(self.solution.longestCommonSubsequence("a", "b"), 0)

    def test_identical_strings(self):
        self.assertEqual(self.solution.longestCommonSubsequence("abc", "abc"), 3)

    def test_no_common_chars(self):
        self.assertEqual(self.solution.longestCommonSubsequence("abc", "def"), 0)

    def test_typical_cases(self):
        self.assertEqual(self.solution.longestCommonSubsequence("abcde", "ace"), 3)  # ace
        self.assertEqual(self.solution.longestCommonSubsequence("abc", "abcde"), 3) # abc
        self.assertEqual(self.solution.longestCommonSubsequence("abcde", "fghij"), 0) # no common
        self.assertEqual(self.solution.longestCommonSubsequence("abcabc", "abc"), 3) # abc repeated

if __name__ == '__main__':
    unittest.main()
