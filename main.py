from Ray import *
from Vector import *
from Triangle import *
from Light import *
from Camera import *
from PIL import Image

def create_image(w, h, hit_list):
    img = Image.new('RGB', (w, h), color=(0, 0, 0))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            if hit_list[x][y] == 1:
                pixels[x, y] = (255, 255, 255)
            elif hit_list[x][y] == 2:
                pixels[x, y] = (255, 0, 0)
            else:
                pixels[x, y] = (0, 0, 0)
    img.save('output\pure_pillow_output.png')
    print("Image saved successfully!")

def tracing_ray(light_list, triangle_list, ray, col_left):
    the_ray = ray
    for _ in range(col_left):
        min_light_time, min_light_l = -1, None
        for l in light_list:
            light_col, light_t = l.collision(the_ray)
            if light_col:
                if(min_light_time < 0):
                    min_light_time = light_t
                    min_light_l = l
                else:
                    if(min_light_time > light_t):
                        min_light_time = light_t
                        min_light_l = l     
            
        min_traingle_time, min_traingle_t = -1, None
        for t in triangle_list:
            traingle_col, tracing_t = t.collision(the_ray)
            if traingle_col:
                if(min_traingle_time < 0):
                    min_traingle_time = tracing_t
                    min_traingle_t = t
                else:
                    if(min_traingle_time > tracing_t):
                        min_traingle_time = tracing_t
                        min_traingle_t = t       
        if not min_traingle_t and not min_light_l:
            return 0
        if not min_traingle_t and min_light_l:
            return 1
        if min_traingle_t and min_light_l and min_light_time <= min_traingle_time:
            return 1
        the_ray = min_traingle_t.reflaction(the_ray, min_traingle_time)
    return 0
        

def tracing_rays(light_list, triangle_list, rays_matrix):
    hit_list = []
    for x in range(W):
        hot_list_y = []
        print(x)
        for y in range(H):
            collide = tracing_ray(light_list, triangle_list, rays_matrix[x][y], MAX_COLLISIONS)
            hot_list_y.append(collide)
        hit_list.append(hot_list_y)
    return hit_list
    
W = 1920
H = 1080
MAX_COLLISIONS = 3

a = Vector(1, -1, 1)
c = Vector(-1, -1, 1)
b = Vector(-1, 1, 1)
origin = Vector(0, -5, 5)
direction = Vector(0, 1, -1)
triangle = Triangle(a, b, c)
camera = Camera(W, H, origin= Vector(0, 0, -2), target=Vector(0, 0, 0), up_vector=Vector(0, 1, 0), FOV=90)
light = Light(Vector(0,0,10), 5)

lights = [light]
triangles = [triangle]
rays_xy = camera.all_rays()



hit_list = tracing_rays(lights, triangles, rays_xy)
create_image(W, H, hit_list)
'''
collide1, t1 = triangle1.collision(ray)
if collide1:
    print(f"{collide1}, {t1} -> {ray.position_at(t1)}")
    ray2 = triangle1.reflaction(ray, t1)
    collide2, t2 =  light.collision(ray2)
    print(f"{collide2}, {t1+t2} -> {ray2.position_at(t2)}")
else:
    print(collide1)
'''