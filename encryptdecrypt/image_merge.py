import os
from PIL import Image, ImageOps


def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
              for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)
    
    return image

# Get list of image paths
def image_size(image_hw,no_of_images):
    folder = "C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\resized_images"
    image_paths = [os.path.join(folder, f) 
               for f in os.listdir(folder) if f.endswith('.jpg')]
    
    image_array = (image_paths)
    
    size=image_hw
    n=no_of_images
    row=int(n/2)
    col=2
    
    image = concat_images(image_array, (size, size), (row, col))
    image.save("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg", 'JPEG')
    
#image_size(400,6)

