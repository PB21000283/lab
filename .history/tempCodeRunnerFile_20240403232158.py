
for i in range( 2):
    x_i_array = np.array(x[i])
    ##print( x_i_array)
    x_transformed= np.array([[item] for item in x_i_array])
    ##print( x_transformed)
    mask_i = noise_mask(x_transformed, 0.2, 3, "together", "random", None, True) 
    ##print( mask_i)
    mask.append(mask_i)
    ##print(mask)
##x=x[0]
##x = [np.array(item) for item in x]
##x_tensors = [torch.tensor(item) for item in x][0:1]
##print(x_tensors)
x=x[0:1]
x = [[len(item), list(item)] for item in x]
print(x[0].shape[0])
##TODO  masked_x!!
#masked_x, targets, target_masks, padding_masks = collate_unsuperv_mask(
       #tra_len , x, mask , max_len=None, vocab=None, add_cls=True)

print(masked_x[0])