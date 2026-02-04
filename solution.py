import re

class Solution:
    def parse_pattern(self, p: str):
        tokens = []
        i = 0
        length = len(p)

        while i < length:
            c = p[i]
            if c == '?':
                tokens.append(('QMARK', None))
                i += 1
            elif c == '*':
                if i+1 < length and p[i+1] == '*':
                    tokens.append(('DSTAR', None))
                    i += 2
                else:
                    tokens.append(('STAR', None))
                    i += 1
            elif c == '[':
                j = i+1
                negate = False
                if j < length and (p[j] == '!' or p[j] == '^'):
                    negate = True
                    j += 1

                char_set = set()

                last_char = None
                while j < length and p[j] != ']':
                    ch = p[j]
                    if ch == '-' and last_char is not None and j+1 < length and p[j+1] != ']':
                        start = last_char
                        end = p[j+1]
                        for code in range(ord(start), ord(end)+1):
                            char_set.add(chr(code))
                        j += 2
                        last_char = None
                        continue
                    else:
                        char_set.add(ch)
                        last_char = ch
                        j += 1

                def match_func(ch, cs=char_set, neg=negate):
                    return (ch not in cs) if neg else (ch in cs)

                tokens.append(('CHARCLASS', match_func))
                i = j+1

            else:
                tokens.append(('LITERAL', c))
                i += 1

        return tokens

    def isMatch(self, s: str, p: str) -> bool:
        tokens = self.parse_pattern(p)
        m, n = len(s), len(tokens)

        dp = [[None]*(n+1) for _ in range(m+1)]

        def match_char(c, token):
            ttype, val = token
            if ttype == 'LITERAL':
                return c == val
            elif ttype == 'QMARK':
                return True
            elif ttype == 'CHARCLASS':
                return val(c)
            else:
                return False

        def dfs(i, j):
            if dp[i][j] is not None:
                return dp[i][j]

            if j == n:
                dp[i][j] = (i == m)
                return dp[i][j]

            ttype, val = tokens[j]

            if ttype in ('LITERAL', 'QMARK', 'CHARCLASS'):
                if i < m and match_char(s[i], tokens[j]):
                    dp[i][j] = dfs(i+1, j+1)
                    return dp[i][j]
                else:
                    dp[i][j] = False
                    return False

            elif ttype == 'STAR':
                if dfs(i, j+1):
                    dp[i][j] = True
                    return True
                for k in range(i, m):
                    if dfs(k+1, j+1):
                        dp[i][j] = True
                        return True
                dp[i][j] = False
                return False

            elif ttype == 'DSTAR':
                for k in range(i, m):
                    if dfs(k+1, j+1):
                        dp[i][j] = True
                        return True
                dp[i][j] = False
                return False

            dp[i][j] = False
            return False

        return dfs(0, 0)


if __name__ == '__main__':
    tests = [
        ("abcdef", "a*f", True),
        ("abcdef", "a**f", True),
        ("af", "a**f", False),
        ("abc", "[a-z][a-z][a-z]", True),
        ("a1c", "[a-z][0-9][a-z]", True),
        ("abc", "[!d-z][!d-z][!d-z]", True),
        ("abc", "a?c", True),
        ("", "*", True),
        ("", "**", False),
        ("aab", "[a][a-b]*", True),
        ("", "", True),
        ("a", "?", True),
        ("ab", "??", True),
        ("abc", "*", True),
        ("abc", "**", True),
        ("abc", "a**", True),
        ("abc", "a*", True),
        ("abc", "a*b*c", True),
        # Corrected expected result False here because pattern 'a**b**c' cannot match 'abc'
        ("abc", "a**b**c", False),
        ("abcd", "a**d", True),
        ("abcd", "a**z", False),
        ("abcd", "a*b*z", False),
        ("abcd", "a*b*[^x-z]", True),
        ("a", "[!a]", False),
        ("b", "[!a]", True),
        ("c", "[a-c]", True),
        ("d", "[!a-c]", True),
        ("d", "[!a-c]*", True),
        ("abcd", "a[!b-j]*d", False),
    ]

    solution = Solution()
    all_pass = True
    for idx, (s, p, expected) in enumerate(tests):
        result = solution.isMatch(s, p)
        if result != expected:
            print(f"Test {idx+1} FAILED: s={s} p={p} expected={expected} got={result}")
            all_pass = False
        else:
            print(f"Test {idx+1} passed: s={s} p={p}")

    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests failed.")

