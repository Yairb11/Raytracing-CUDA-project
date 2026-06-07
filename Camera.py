from Vector import *
from Ray import *
import math
class Camera:
    def __init__(self, W, H, origin, target, up_vector, FOV):
        self.w = W
        self.h = H
        self.origin = origin
        self.forword = (target - origin).normalize()
        self.right = up_vector.cross(self.forword).normalize()
        self.up = self.forword.cross(self.right).normalize()
        self.FOV = FOV
    def all_rays(self):
        aspect_ratio = self.w/self.h * math.tan(math.radians(self.FOV) / 2)
        NDCx = [(2 * (x+0.5)/self.w - 1) * aspect_ratio for x in range(self.w)]
        NDCy = [(-2 * (y+0.5)/self.h + 1) * aspect_ratio for y in range(self.h)]
        rays_xy = []
        for x in range(self.w):
            rays_y = []
            right_part = self.right * NDCx[x]
            for y in range(self.h):
                up_part = self.up * NDCy[y]
                dir = right_part + up_part + self.forword
                ray = Ray(self.origin, dir)
                rays_y.append(ray)
            rays_xy.append(rays_y)
        return rays_xy
    
    def all_rays_cuda(self):
        
        aspect_ratio = self.w/self.h * math.tan(math.radians(self.FOV) / 2)
        NDCx = [(2 * (x+0.5)/self.w - 1) * aspect_ratio for x in range(self.w)]
        NDCy = [(-2 * (y+0.5)/self.h + 1) * aspect_ratio for y in range(self.h)]
        ray_origin_list = []
        ray_dir_list = []
        for x in range(self.w):
            ray_origin_y = []
            ray_dir_y = []
            right_part = self.right * NDCx[x]
            for y in range(self.h):
                up_part = self.up * NDCy[y]
                dir = right_part + up_part + self.forword
                ray_origin = self.origin.to_list()
                ray_dir = dir.to_list()
                ray_origin_y.append(ray_origin)
                ray_dir_y.append(ray_dir)
            ray_origin_list.append(ray_origin_y)
            ray_dir_list.append(ray_dir_y)
        return ray_origin_list, ray_dir_list