import math
EPSILON = 0.000001
class Vector:
    def __init__(self, x, y, z):
        self.x = round(x / EPSILON) * EPSILON
        self.y = round(y / EPSILON) * EPSILON
        self.z = round(z / EPSILON) * EPSILON
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        return Vector(self.x * other, self.y * other, self.z * other)
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __eq__(self, other):
        return abs(self.x - other.x) <= EPSILON and abs(self.y - other.y) <= EPSILON and abs(self.z- other.z) <= EPSILON
    
    def cross(self,other):
        return Vector((self.y * other.z - self.z * other.y), (self.z * other.x - self.x * other.z), (self.x * other.y - self.y * other.x))
    
    def normalize(self):
        amp = 1 / math.sqrt(self * self)
        return self * amp 
    def magnitude(self):
        return math.sqrt(self * self)
    
    def min(self, other):
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        min_z = min(self.z, other.z)
        return Vector(min_x, min_y, min_z)
    
    def max(self, other):
        max_x = max(self.x, other.x)
        max_y = max(self.y, other.y)
        max_z = max(self.z, other.z)
        return Vector(max_x, max_y, max_z)
    
    def to_list(self):
        return [self.x, self.y, self.z]