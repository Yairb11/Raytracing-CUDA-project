from Vector import *
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    def position_at(self, t):
        return self.origin + self.direction * t
    def __str__(self):
        return f"Origin: {str(self.origin)}\nDirection: {str(self.direction)}"