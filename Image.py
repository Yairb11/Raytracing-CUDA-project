from PIL import Image
from Vector import *
def create_image(w, h, hit_list, name):
    img = Image.new('RGB', (w, h), color=(0, 0, 0))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            r = int(min(255, hit_list[x][y].x * 255))
            g = int(min(255, hit_list[x][y].y * 255))
            b = int(min(255, hit_list[x][y].z * 255))
            pixels[x, y] = (r, g, b)
    img.save(rf'output\{name}.png')
    print("Image saved successfully!")
    
def create_image_cuda(w, h, hit_list, name):
    img = Image.new('RGB', (w, h), color=(0, 0, 0))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            r = int(min(255, hit_list[x, y, 0] * 255))
            g = int(min(255, hit_list[x, y, 1]* 255))
            b = int(min(255, hit_list[x, y, 2]* 255))
            pixels[x, y] = (r, g, b)
    img.save(rf'output\{name}.png')
    print("Image saved successfully!")