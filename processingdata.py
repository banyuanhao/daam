import torch

dicdata = torch.load('/home/banyh2000/diffusion/daam/save_glasses.pt')

ten = dicdata['placehold']
print(ten.shape)

count_pos = 0
count_neg = 0
for i in range(110):
    if ten[i,30] > 0:
        count_pos += 1
        
    if ten[i,1] < 0:
        count_neg += 1
        
print(count_pos)
print(count_neg)
    
    