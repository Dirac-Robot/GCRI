class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)

        # If one of the strings is empty
        if m == 0:
            return n
        if n == 0:
            return m

        # dp[i][j] will be the edit distance between word1[:i] and word2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize dp for empty word transformations
        for i in range(m + 1):
            dp[i][0] = i  # delete all characters from word1
        for j in range(n + 1):
            dp[0][j] = j  # insert all characters to word1

        # Fill in dp with the minimum edit distance
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # characters match, no operation needed
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,    # delete
                        dp[i][j - 1] + 1,    # insert
                        dp[i - 1][j - 1] + 1 # replace
                    )

        return dp[m][n]


# Comprehensive tests
import unittest

class TestEditDistance(unittest.TestCase):
    def setUp(self):
        self.sol = Solution()

    def test_empty_strings(self):
        self.assertEqual(self.sol.minDistance('', ''), 0)
        self.assertEqual(self.sol.minDistance('', 'abc'), 3)
        self.assertEqual(self.sol.minDistance('abc', ''), 3)

    def test_single_chars(self):
        self.assertEqual(self.sol.minDistance('a', 'a'), 0)
        self.assertEqual(self.sol.minDistance('a', 'b'), 1)
        self.assertEqual(self.sol.minDistance('a', ''), 1)
        self.assertEqual(self.sol.minDistance('', 'a'), 1)

    def test_identical_strings(self):
        self.assertEqual(self.sol.minDistance('abc', 'abc'), 0)
        self.assertEqual(self.sol.minDistance('a', 'a'), 0)

    def test_general_cases(self):
        self.assertEqual(self.sol.minDistance('horse', 'ros'), 3)
        self.assertEqual(self.sol.minDistance('intention', 'execution'), 5)

    def test_prefix_suffix(self):
        self.assertEqual(self.sol.minDistance('abcd', 'abcde'), 1)
        self.assertEqual(self.sol.minDistance('abcde', 'abcd'), 1)

    def test_replace_only(self):
        self.assertEqual(self.sol.minDistance('abc', 'def'), 3)

    def test_insert_delete_mix(self):
        self.assertEqual(self.sol.minDistance('kitten', 'sitting'), 3)

if __name__ == '__main__':
    unittest.main()  

