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
        self.normal = self.ba.cross(self.ca)
    
    def collision(self, ray):
        if self.normal == ZERO_VECTOR:
            return False, -1
        parallel_dot = ray.direction * self.normal
        if(abs(parallel_dot) < EPSILON):
            return False, -1
        f = 1.0 / parallel_dot
        s = self.point_a - ray.origin
        u = ((-1.0) * f) * (ray.direction * (s.cross(self.ca)))
        v = (f) * (ray.direction * (s.cross(self.ba)))
        t = (f) * (s * self.normal)
        if u > 1.0 or u < 0.0 or v + u > 1.0 or v < 0.0 or t <= EPSILON :
            return False, -1
        return True, t
    
    def reflaction(self, ray, t):
        normal = self.normal.normalize()
        if(ray.direction * normal < 0):
            
            position = ray.position_at(t) - (normal * EPSILON)
        else:
            position = ray.position_at(t) + (normal * EPSILON)
        d = ray.direction
        dot = (d * normal)
        dot *= 2 
        n = normal * dot
        ref =  d - n
        return Ray(position, ref)