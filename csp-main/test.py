pairs = ["old cat", "young tiger", "red apple"]
pairs = [t.split() for t in pairs]
print(pairs)
pairs = map(tuple, pairs)
print(pairs)