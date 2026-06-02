from Ray import *
from Vector import *
from Triangle import *

a = Vector(-1, -1, 3)
b = Vector(1, -1, 3)
c = Vector(0, 3, 3)
origin = Vector(0, 0, 0)
direction = Vector(0, 1, 1)

ray = Ray(origin, direction)
triangle = Triangle(a, b, c)

collide, t = triangle.collision(ray)
if collide:
    print(f"{collide} -> {ray.position_at(t)}")
else:
    print(collide)