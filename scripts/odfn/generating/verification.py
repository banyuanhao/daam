from utils_odfn import seeds_plus_spilt, seeds_plus_shuffled
train = set(seeds_plus_spilt('train'))
print(len(train))
val = set(seeds_plus_spilt('val'))
print(len(val))
test = set(seeds_plus_spilt('test'))
print(len(test))

train_sh = set(seeds_plus_shuffled[:17500])
val_sh = set(seeds_plus_shuffled[17500:18500])
test_sh = set(seeds_plus_shuffled[18500:])

v1 = train & train_sh
v2 = val & val_sh
v3 = test & test_sh
print(len(v1), len(v2), len(v3))

