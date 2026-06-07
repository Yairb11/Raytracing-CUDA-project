
import math
from Classes.Vector import *
from Classes.Ray import *
EPSILON = 1e-8
EPSILON_MOVE = 1e-4
class Light:
    def __init__(self, position, radius, color = Vector(1, 1, 1)):
        self.position = position
        self.radius = radius
        self.color = color
    
    def collision(self, ray):
        L = self.position - ray.origin
        tca = L * ray.direction
        if tca < 0:
            return False, None
        d2 = (L * L) - (tca * tca)
        radius2 = self.radius * self.radius
        if d2 > radius2:
            return False, None
        thc = math.sqrt(radius2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        epsilon = 1e-5
        if t0 > epsilon:
            return True, t0
        if t1 > epsilon:
            return True, t1
        return False, None
            