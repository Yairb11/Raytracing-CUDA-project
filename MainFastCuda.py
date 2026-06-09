
from Classes.Ray import *
from Classes.Vector import *
from Classes.Triangle import *
from Classes.Light import *
from Classes.Camera import *
import Classes.Blender as Blender
import Classes.Image as Image
import Classes.FastCudaFunctions as FastCudaFunctions
import Classes.BVHNode as BVHNode
import time 

def light_to_cuda(lights):
    lights_pos = []
    lights_radius = []
    lights_color = []
    for light in lights:
        lights_pos.append(light.position.to_list())
        lights_radius.append(light.radius)
        lights_color.append(light.color.to_list())
    return lights_pos, lights_radius, lights_color

def setup_engine(camera, lights, name):
    vertices, faces = Blender.extract_triangles(rf"input\{name}.obj")
    triangles = Blender.create_all_triangles(vertices, faces)
    bvh_triangles = BVHNode.build_bvh(triangles)
    bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color = BVHNode.flatten_bvh(bvh_triangles)
    lights_position, lights_radius, lights_color = light_to_cuda(lights)
    worm_up = FastCudaFunctions.render_scene_gpu(16, -1, 
                                        camera, 
                                        [bvh_border[0]], bvh_child, [bvh_triangles_indexes[0]], [bvh_ordered_triangles_points[0]], [bvh_ordered_triangles_color[0]], 
                                        [lights_position[0]], [lights_radius[0]], [lights_color[0]])
    return bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color
    
def engine(camera, 
           bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color, 
           lights):
    
    lights_position, lights_radius, lights_color = light_to_cuda(lights)   
    hit_list = FastCudaFunctions.render_scene_gpu(16, MAX_DEPTH, 
                                        camera, 
                                        bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color, 
                                        lights_position, lights_radius, lights_color)
    
    
    return hit_list


def start_engine(W, H, camera, lights, name, adding_to_image):
    bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color = setup_engine(camera, lights, name)

    hit_list = engine(camera, 
           bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color, 
           lights)
    print("image")
    Image.create_image_cuda(W, H, hit_list, f"{name}_{adding_to_image}")

MAX_DEPTH = 5
def main():
    W = 1270
    H = 720
    name = "monkey"
    adding_to_image = "CUDA_1"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(0,0,-4), 1)]
    start_engine(W, H, camera, lights, name, adding_to_image)
    
if __name__ == "__main__":
    main()