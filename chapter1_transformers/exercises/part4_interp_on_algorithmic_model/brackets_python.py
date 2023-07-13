# %%
def bracket_classifier1(s: str) -> bool:
    balance = 0
    for c in s:
        if balance < 0:
            return False
        if c == "(":
            balance += 1
        if c == ")":
            balance -= 1
    return balance == 0


s1 = "((()))"
s2 = "()()()()"

s3 = ")()()()("
s4 = "((())))"
s5 = "()()()()("
s6 = "()()())("
for s in [s1, s2, s3, s4, s5, s6]:
    print(bracket_classifier1(s))

# %%
