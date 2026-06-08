
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


def start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image):
    vertices, faces = Blender.extract_triangles(rf"input\{name}.obj")
    triangles = Blender.create_all_triangles(vertices, faces)
    bvh_triangles = BVHNode.build_bvh(triangles)
    bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color = BVHNode.flatten_bvh(bvh_triangles)
    lights_position, lights_radius, lights_color = light_to_cuda(lights)
    print("not start")
    hit_list = FastCudaFunctions.render_scene_gpu(16, 0, 
                                            camera, 
                                            bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color, 
                                            lights_position, lights_radius, lights_color)
    print("start")
    start_time = time.time()
    hit_list = FastCudaFunctions.render_scene_gpu(16, MAX_DEPTH, 
                                            camera, 
                                            bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color, 
                                            lights_position, lights_radius, lights_color)
    end_time = time.time()
    print("image")
    Image.create_image_cuda(W, H, hit_list, f"{name}_{adding_to_image}_{end_time-start_time}")
    print(f"\n\n\nInfo: {W} x {H} -> Depth: {MAX_DEPTH}, Faces: {len(triangles)}\nProcessing: {end_time-start_time} seconds\n")


def main():
    W = 1920
    H = 1080
    MAX_DEPTH = 5
    name = "monkey"
    adding_to_image = "CUDA_"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(10,0,0), 1), Light(Vector(-10,0,0), 1), Light(Vector(0,-10,0), 1), Light(Vector(0,10,0), 1), Light(Vector(0, 0, 10), 1), Light(Vector(0, 0, -10), 1)]
    start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image)
    
if __name__ == "__main__":
    main()