
from Ray import *
from Vector import *
from Triangle import *
from Light import *
from Camera import *
import Blender
import Image
import CudaFunctions

def light_to_cuda(lights):
    lights_pos = []
    lights_radius = []
    for light in lights:
        lights_pos.append(light.position.to_list())
        lights_radius.append(light.radius)
    return lights_pos, lights_radius
    
W = 640
H = 640
MAX_DEPTH = 5

def start_engine(camera, lights, name, adding_to_image):
    vertices, faces = Blender.extract_triangles(rf"input\{name}.obj")
    triangles = Blender.create_all_triangles_cuda(vertices, faces)
    ray_origin_list, ray_dir_list = camera.all_rays_cuda()
    lights_position, lights_radius = light_to_cuda(lights)
    print("start")
    hit_list = CudaFunctions.render_scene_gpu(16, ray_origin_list, ray_dir_list, triangles, lights_position, lights_radius)
    Image.create_image_cuda(W, H, hit_list, f"{name}_{adding_to_image}")

if __name__ == "__main__":
    name = "monkey"
    adding_to_image = "Test_cuda_1"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=120)
    lights = [Light(Vector(0,10,0), 1), Light(Vector(10,0,0), 1), Light(Vector(-10,0,0), 1), Light(Vector(0,-10,0), 1), Light(Vector(0,0, 10), 1), Light(Vector(0,0, -10), 1)]
    start_engine(camera, lights, name, adding_to_image)