import dictionary

dic = dictionary.Dictionary()
dic["one"] = 0
dic["one"] = 1
dic["two"] = 2
dic["three"] = 3
print(dic)
print(dic["two"])
print(dic.keys())
print(dic.values())
print(dic.items())
dic.pop("two")
print(dic)
dic.clear()
dic["four"] = 4
dic["five"] = 5
print(dic)

# expected print
"""
{(one : 1), (two : 2), (three : 3)}
2
['one', 'two', 'three']
[1, 2, 3]
[('one', 1), ('two', 2), ('three', 3)]
{(one : 1), (three : 3)}
{(four : 4), (five : 5)}
"""