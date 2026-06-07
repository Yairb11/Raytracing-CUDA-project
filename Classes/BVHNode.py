from Classes.AABB import *
from Classes.Triangle import *
from Classes.Vector import *
import math

class BVHNode:
    def __init__(self):
        self.aabb = None
        self.left = None
        self.right = None
        self.triangles = []
        self.is_leaf = False
        
def build_bvh(triangles, max_triangles_per_leaf = 2):
    node = BVHNode()
    
    total_bmin = triangles[0].bmin
    total_bmax = triangles[0].bmax
    for triangle in triangles[1:]:
        total_bmin = total_bmin.min(triangle.bmin)
        total_bmax = total_bmax.max(triangle.bmax)
    
    node.aabb = AABB(total_bmin, total_bmax)
    if len(triangles) <= max_triangles_per_leaf:
        node.is_leaf = True
        node.triangles = triangles
        return node
    
    extent = total_bmax - total_bmin
    if(extent.z >= max(extent.y, extent.x)):
        triangles.sort(key=lambda triangle: triangle.centroid.x)
    elif (extent.y >= max(extent.x, extent.z)):
        triangles.sort(key=lambda triangle: triangle.centroid.y)
    else:
        triangles.sort(key=lambda triangle: triangle.centroid.z)
    
    mid = len(triangles) // 2
    left_triangles = triangles[:mid]
    right_triangles = triangles[mid:]
    node.left = build_bvh(left_triangles, max_triangles_per_leaf)
    node.right = build_bvh(right_triangles, max_triangles_per_leaf)   
    return node

def hit_bvh(root_nood, ray):
    stack_node = [root_nood]
    t_max = float("inf")
    hit_object = None
    closest_t = t_max
    while(stack_node):
        node = stack_node.pop()
        if not node.aabb.intersect(ray):
            continue
    
        if node.is_leaf:
            for triangle in node.triangles :
                collides_triangle, t_triangle = triangle.collision(ray)
                if collides_triangle and t_triangle < closest_t:
                    closest_t = t_triangle
                    hit_object = triangle
        else:
            if node.right:
                stack_node.append(node.right)
            if node.left:
                stack_node.append(node.left)
    return hit_object, closest_t
    
        