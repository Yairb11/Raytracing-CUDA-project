import math
EPSILON = 0.000001
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
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
    
    def __eq__(self, other):
        return abs(self.x - other.x) <= EPSILON and abs(self.y - other.y) <= EPSILON and abs(self.z- other.z) <= EPSILON
    
    def cross(self,other):
        return Vector((self.y * other.z - self.z * other.y), (self.z * other.x - self.x * other.z), (self.x * other.y - self.y * other.x))
    
    def square(self):
        return math.sqrt(self * self)