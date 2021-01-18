import calculator

calc = calculator.calculator()
print(calc.add(5, 2))
print(calc.sub(5, 2))
print(calc.mul(5, 2))
print(calc.div(5, 2))
print(calc[1])
print(calc[4])
print(calc)
calc[1] = 2.345
calc[2] = "1234"
print(calc.cache)

# expected print

"""
7
3
10
2.5
5 * 2 = 10
Out of Range
0. 5 / 2 = 2.5
1. 5 * 2 = 10
2. 5 - 2 = 3
3. 5 + 2 = 7

[0, 2.345, 0, 0, 0]
"""