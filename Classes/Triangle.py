from Classes.Vector import *
from Classes.Ray import *
from Classes.AABB import *
EPSILON = 1e-8
EPSILON_MOVE = 1e-4
ZERO_VECTOR = Vector(0, 0, 0)
class Triangle:
    def __init__(self, point_a, point_b, point_c, color = Vector(1, 1, 1)):
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        self.color = color
        self.ba = point_b - point_a
        self.ca = point_c - point_a
        self.normal = (self.ba.cross(self.ca))
        
        self.bmin = point_a.min(point_b.min(point_c))
        self.bmax = point_a.max(point_b.max(point_c))
        self.aabb = AABB(self.bmin, self.bmax)
        self.centroid = (point_a + point_b + point_c) * (1.0 / 3.0)
    
    def __str__(self):
        str_a = str(self.point_a)
        str_b = str(self.point_b)
        str_c = str(self.point_c)
        return f"A: {str_a}, B: {str_b}, C: {str_c}"
    
    
    
    def collision(self, ray):
        d = ray.direction
        a = d * (-1 * self.normal)
        if -EPSILON < a < EPSILON:
            return False, None
        f = 1.0 / a
        s = ray.origin - self.point_a
        
        u = (f * d) * (s.cross(self.ca))
        if u > 0.0 or u < -1.0:
            return False, None
        
        v = (f * d) * (s.cross(self.ba))
        if v < 0.0 or v-u > 1.0:
            return False, None
        
        t = (f * s) * self.normal
        if t > EPSILON:
            return True, t
        return False, None
    
    def normal_from_ray(self, ray):
        triangle_face = ray.direction * self.normal
        if(triangle_face > 0):
            return (-1.0) * self.normal.normalize()
        else:
            return self.normal.normalize()
    
    def reflection(self, ray, t, normal):
        hit_point = ray.position_at(t) + EPSILON_MOVE * normal  
        dot_product = ray.direction * normal
        reflected_direction = ray.direction - (normal * (2.0 * dot_product))
        return Ray(hit_point, reflected_direction)
    
    def to_list(self):
        triangle_list = []
        triangle_list.append(self.point_a.to_list())
        triangle_list.append(self.point_b.to_list())
        triangle_list.append(self.point_c.to_list())
        return triangle_list