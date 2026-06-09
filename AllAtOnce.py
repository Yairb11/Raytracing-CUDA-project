from Classes.Light import *
from Classes.Camera import *
import MainRegular
import MainCuda
import MainBVH
import MainFastCuda
import time


if __name__ == "__main__":
    W = 1920
    H = 1080
    MAX_DEPTH = 20
    name = "monkey"
    adding_to_image = "Test_light_1"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(0,100,0), 1), Light(Vector(-0,0,-10), 1)]
    # Tests
    print("Faster CUDA:")
    start_time = time.time()
    MainFastCuda.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_FASTCUDA")
    end_time = time.time()
    print(f"{end_time-start_time} seconds")
    
    print("CUDA:")
    start_time = time.time()
    MainCuda.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_CUDA")
    end_time = time.time()
    print(f"{end_time-start_time} seconds")
    
    print("BVH:")
    start_time = time.time()
    MainBVH.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_BVH")
    end_time = time.time()
    print(f"{end_time-start_time} seconds")
    
    print("REGULAR:")
    start_time = time.time()
    MainRegular.start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image + "_regular")
    end_time = time.time()
    print(f"{end_time-start_time} seconds")
    