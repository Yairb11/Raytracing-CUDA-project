EPSILON = 0.000001
import math
from Vector import *
from Ray import *
class Light:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
    
    def collision(self, ray):
        l = ray.origin - self.position 
        d = ray.direction
        a = d * d
        b = (d * l) * 2
        c = (l * l) - (self.radius * self.radius + EPSILON)
        discriminant = b * b - 4 * a * c
        if(discriminant < 0):
            return False, -1
        t= (- b - math.sqrt(discriminant)) / (2 * a)
        if t < 0:
            return False, -1
        return True, t
            