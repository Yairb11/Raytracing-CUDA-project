
from Classes.Ray import *
from Classes.Vector import *
from Classes.Triangle import *
from Classes.Light import *
from Classes.Camera import *
import Classes.Blender as Blender
import Classes.Image as Image
import Classes.CudaFunctions as CudaFunctions

def light_to_cuda(lights):
    lights_pos = []
    lights_radius = []
    lights_color = []
    for light in lights:
        lights_pos.append(light.position.to_list())
        lights_radius.append(light.radius)
        lights_color.append(light.color.to_list())
    return lights_pos, lights_radius, lights_color


def start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image):
    vertices, faces = Blender.extract_triangles(rf"input\{name}.obj")
    triangles_points, triangles_color = Blender.create_all_triangles_cuda(vertices, faces)
    ray_origin_list, ray_dir_list = camera.all_rays_cuda()
    lights_position, lights_radius, lights_color = light_to_cuda(lights)
    print("start")
    hit_list = CudaFunctions.render_scene_gpu(16, MAX_DEPTH, 
                                              ray_origin_list, ray_dir_list, 
                                              triangles_points, triangles_color, 
                                              lights_position, lights_radius, lights_color)
    Image.create_image_cuda(W, H, hit_list, f"{name}_{adding_to_image}")


def main():
    W = 640
    H = 640
    MAX_DEPTH = 5
    name = "monkey"
    adding_to_image = "Test_CUDA"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(10,0,0), 1), Light(Vector(-10,0,0), 1), Light(Vector(0,-10,0), 1), Light(Vector(0,10,0), 1), Light(Vector(0, 0, 10), 1), Light(Vector(0, 0, -10), 1)]
    start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image)
    
if __name__ == "__main__":
    main()