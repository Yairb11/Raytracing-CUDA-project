
from Ray import *
from Vector import *
from numba import cuda
import numpy as np
import math

@cuda.jit(device=True)
def dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@cuda.jit(device=True)
def cross_product(out, v1, v2):
    out[0] = v1[1]*v2[2] - v1[2]*v2[1]
    out[1] = v1[2]*v2[0] - v1[0]*v2[2]
    out[2] = v1[0]*v2[1] - v1[1]*v2[0]

@cuda.jit(device=True)
def normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length > 0:
        v[0] /= length
        v[1] /= length
        v[2] /= length
    
@cuda.jit(device = True)
def intersect_triangle(ray_origin, ray_direction, a, b, c):
    ba = cuda.local.array(3, dtype=np.float32)
    ca = cuda.local.array(3, dtype=np.float32)
    normal = cuda.local.array(3, dtype=np.float32)
    s = cuda.local.array(3, dtype=np.float32)
    sba = cuda.local.array(3, dtype=np.float32)
    sca = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        ba[i] = b[i] - a[i]
        ca[i] = c[i] - a[i]
        s[i] = ray_origin[i] - a[i]
    cross_product(normal, ba, ca)
    a = -1 * dot_product(ray_direction, normal)
    if -0.00001 < a < 0.00001:
        return -1.0
    
    f = 1.0 / a
    cross_product(sba, s, ba)
    cross_product(sca, s, ca)
    u = f * dot_product(ray_direction, sca)
    if u > 0.0 or u < -1.0 :
        return -1.0
    v = f * dot_product(ray_direction, sba)
    if v < 0.0 or v-u > 1.0:
        return -1.0
    t = f * dot_product(s, normal)
    if t > 0.00001:
        return t
    return -1.0

@cuda.jit
def raytrace_kernel(ray_origins, ray_dirs, triangles, lights_pos, lights_radius, output_colors):
    x, y = cuda.grid(2)
    if x >= len(output_colors) or y >= len(output_colors[0]):
        return
    closest_t = 9999999.0
    hit = -1
    
    origin = ray_origins[x, y]
    dir = ray_dirs[x, y]
    for i in range(len(triangles)):
        a = triangles[i, 0]
        b = triangles[i, 1]
        c = triangles[i, 2]
        t = intersect_triangle(origin, dir, a, b, c)
        if t > 0 and t < closest_t:
            closest_t = t
            hit = i
    
    if hit > -1:
        output_colors[x, y, 0] = 1
        output_colors[x, y, 1] = 1
        output_colors[x, y, 2] = 1
    else:
        output_colors[x, y, 0] = 0
        output_colors[x, y, 1] = 0
        output_colors[x, y, 2] = 0
    
def render_scene_gpu(n, ray_origin_list, ray_dir_list, triangle_list, light_pos_list, light_radius_list):
    width, height = len(ray_origin_list), len(ray_origin_list[0])
    output_colors = []
    for x in range(width):
        output = []
        for y in range(height):
            color = []
            for z in range(3):
                color.append(0)
            output.append(color)
        output_colors.append(output)
    
    np_ray_origin_list = np.array(ray_origin_list, dtype= np.float32)
    np_ray_dir_list = np.array(ray_dir_list, dtype= np.float32)
    np_triangle_list = np.array(triangle_list, dtype= np.float32)
    np_light_pos_list = np.array(light_pos_list, dtype= np.float32)
    np_light_radius_list = np.array(light_radius_list, dtype= np.float32)
    np_output_colors = np.array(output_colors, dtype= np.float32)
    
    
    d_origins = cuda.to_device(np_ray_origin_list)
    d_dirs = cuda.to_device(np_ray_dir_list)
    d_triangles = cuda.to_device(np_triangle_list)
    d_lights_position = cuda.to_device(np_light_pos_list)
    d_lights_radius = cuda.to_device(np_light_radius_list)
    d_output = cuda.to_device(np_output_colors)
    
    threads_per_block = (n, n)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    raytrace_kernel[blocks_per_grid, threads_per_block](
        d_origins, d_dirs, d_triangles, d_lights_position, d_lights_radius, d_output
    )
    
    cuda.synchronize()
    d_output.copy_to_host(np_output_colors)
    return np_output_colors
    
    