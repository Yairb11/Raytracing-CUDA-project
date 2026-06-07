from Classes.Light import *
from Classes.Camera import *
import main_regular
import main_cuda
import main_bvh


if __name__ == "__main__":
    W = 640
    H = 640
    MAX_DEPTH = 5
    name = "monkey"
    adding_to_image = "Test_cuda_1"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(10,0,0), 1), Light(Vector(-10,0,0), 1), Light(Vector(0,-10,0), 1)]
    # Test 
    print("CUDA:")
    main_cuda.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_CUDA")
    print("BVH:")
    main_bvh.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_BVH")
    print("REGULAR:")
    main_regular.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_regular")
    