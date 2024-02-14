import json
with open('result_0_add1.json') as f:
    data = json.load(f)
    
context_total = 0
context_time_total = 0
compare_total = 0
total = 0

for key, value in data.items():
    context_total += value['context_total']
    context_time_total += value['context_time_total']
    compare_total += value['compare_total']
    total += value['total']
    
print('context_total:', context_total)
print('context_time_total:', context_time_total)
print('compare_total:', compare_total)
print('total:', total)

print('context_total:', context_total/total)
print('context_time_total:', context_time_total/total)
print('compare_total:', compare_total/total)