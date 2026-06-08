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

def flatten_bvh(root_node):
    if root_node is None:
        return None
    flat_border = []
    flat_child = []
    flat_triangle_indexes = []
    
    ordered_triangles_points = []   
    ordered_triangles_color = []   
    stack = [(root_node, -1, False)]
    while len(stack) > 0:
        node, parent_index, is_left_child = stack.pop()
        current_index = len(flat_border)
        if parent_index != -1:
            if is_left_child:
                flat_child[parent_index][0] = current_index
            else:
                flat_child[parent_index][1] = current_index
        
        flat_border.append([node.aabb.bmin.to_list(), node.aabb.bmax.to_list()])
        flat_child.append([-1, -1])
        flat_triangle_indexes.append([])
        
        if node.is_leaf:
            start_idx = len(ordered_triangles_points)
            flat_triangle_indexes[current_index] = [start_idx, len(node.triangles)]
            for triangle in node.triangles:
                ordered_triangles_points.append(triangle.to_list())
                ordered_triangles_color.append(triangle.color.to_list())
        else:
            flat_triangle_indexes[current_index] = [-1, -1]
            if node.right:
                stack.append((node.right, current_index, False))
            if node.left:
                stack.append((node.left, current_index, True))
    return flat_border, flat_child, flat_triangle_indexes, ordered_triangles_points, ordered_triangles_color