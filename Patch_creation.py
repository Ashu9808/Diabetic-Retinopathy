import os
import Image


for file in os.listdir(dir_data):
    outfile=os.path.splitext(file)[0]
    
    extension=os.path.splitext(file)[1]
    
    
    
    
    if (extension==".tif"):
        continue
    
    im = Image.open(os.path.join(dir_mask+outfile+"_EX"+".tif")).convert('RGB')
    
    imd = Image.open(os.path.join(dir_data,file)).convert('RGB')

    
    
    patch_id = 0
    for i in range(10):
    	for j in range(16):
            top_y = i*256
            if (i==9):
                top_y = 2336
            top_x = j*256
            if (j==15):
                top_x = 3776
                
            im_crop = im.crop((top_x,top_y,top_x+512,top_y+512))
            imd_crop = imd.crop((top_x,top_y,top_x+512,top_y+512))
            im_crop = np.array(im_crop)
            if np.sum(im_crop)==0:                 # for calculating the number of total black patches.
                negative_patches.append(output_dir_mask+outfile+"_p"+str(patch_id)+extension)
                
            else : 
                positive_count+=1
            
            
#             im_crop = (im_crop > 50) * 1.0
            
            im_crop=PIL.Image.fromarray(im_crop)
            
            
            
            
            im_crop.save(output_dir_mask+outfile+"_p"+str(patch_id)+extension)
            imd_crop.save(output_dir_data+outfile+"_p"+str(patch_id)+extension)
            
            
            
            patch_id+=1
