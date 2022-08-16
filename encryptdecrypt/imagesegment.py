from PIL import Image
import os
import copysequencefilename as name

def  image_segemnt(savedir,filename,size,y):
    img = Image.open(filename)
    width, height = img.size
    start_pos = start_x, start_y = (0, 0)
    cropped_image_size = w, h = (size, size)
    fslist=y

    frame_num = 0
    
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            crop = img.crop((col_i, row_i, col_i + w, row_i + h))
            save_to= os.path.join(savedir, "{}.jpg")
            crop.save(save_to.format(fslist[frame_num]))
            frame_num += 1

def image_segment_values(size,no_of_images,i):
    savedir = "C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\outputsegment"
    filename = "C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\encrypted_image.jpg"
    image_size=size
    n=no_of_images
    y=name.filename(n,i)
    image_segemnt(savedir,filename,size,y)