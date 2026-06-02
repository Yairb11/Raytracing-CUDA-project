from Vector import *
from Ray import *
EPSILON = 0.000001
ZERO_VECTOR = Vector(0, 0, 0)
class Triangle:
    def __init__(self, point_a, point_b, point_c):
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        self.ba = point_b - point_a
        self.ca = point_c - point_a
    
    def collision(self, ray):
        normal_plane =  self.ba.cross(self.ca)
        if normal_plane == ZERO_VECTOR:
            return False, -1
        parallel_dot = ray.direction * normal_plane
        if(abs(parallel_dot) < EPSILON):
            return False, -1
        f = 1.0 / parallel_dot
        s = self.point_a - ray.origin
        u = ((-1.0) * f) * (ray.direction * (s.cross(self.ca)))
        v = (f) * (ray.direction * (s.cross(self.ba)))
        t = (f) * (s * normal_plane)
        if u > 1.0 or u < 0.0 or v + u > 1.0 or v < 0.0 or t <= EPSILON :
            return False, -1
        return True, t
        
    