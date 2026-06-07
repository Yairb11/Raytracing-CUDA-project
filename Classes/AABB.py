from Classes.Vector import *
from Classes.Ray import *
import math

class AABB:
    def __init__(self, bmin, bmax):
        self.bmin = bmin
        self.bmax = bmax
        
    def Union(self, other):
        return AABB(self.bmin.min(other.bmin), self.bmax.max(other.bmax))
    
    def intersect(self, ray):
        ray_dir_inv_x = 1.0 / (ray.direction.x if ray.direction.x != 0 else 1e-8)
        ray_dir_inv_y = 1.0 / (ray.direction.y if ray.direction.y != 0 else 1e-8)
        ray_dir_inv_z = 1.0 / (ray.direction.z if ray.direction.z != 0 else 1e-8)
        t1_x = (self.bmin.x - ray.origin.x) * ray_dir_inv_x
        t1_y = (self.bmin.y - ray.origin.y) * ray_dir_inv_y
        t1_z = (self.bmin.z - ray.origin.z) * ray_dir_inv_z
        t2_x = (self.bmax.x - ray.origin.x) * ray_dir_inv_x
        t2_y = (self.bmax.y - ray.origin.y) * ray_dir_inv_y
        t2_z = (self.bmax.z - ray.origin.z) * ray_dir_inv_z
        t_enter = max(min(t1_x, t2_x), min(t1_y, t2_y), min(t1_z, t2_z))
        t_exit = min(max(t1_x, t2_x), max(t1_y, t2_y), max(t1_z, t2_z))
        return t_enter <= t_exit and t_exit > 0
        