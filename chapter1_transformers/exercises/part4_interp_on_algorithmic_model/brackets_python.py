# %%
def bracket_classifier1(s: str) -> bool:
    balance = 0
    if s[0] == ")":
        return False
    for c in s:
        if c == "(":
            balance += 1
        if c == ")":
            balance -= 1
    return balance == 0


def bracket_classifier2(s: str) -> bool:
    balance = 0
    left_bracket = False
    right_bracket = False
    if s[0] == ")":
        return False
    for i, c in enumerate(s):
        if c == "(" and not left_bracket:
            left_bracket = True
        if c == ")" and right_bracket:
            right_bracket = True
        if right_bracket and left_bracket:
            return bracket_classifier(s[i:])
    if (left_bracket * 1 + right_bracket * 1) == 1:
        return False
    else:
        return True


s1 = "((()))"
s2 = "()()()()"

s3 = ")()()()("
s4 = "((())))"
s5 = "()()()()("
s6 = "()()())("
for s in [s1, s2, s3, s4, s5, s6]:
    print(bracket_classifier(s))

# %%
