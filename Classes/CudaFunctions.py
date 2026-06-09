
from Classes.Ray import *
from Classes.Vector import *
from Classes.Camera import *
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
def copy(input, output):
    output[0] = input[0]
    output[1] = input[1]
    output[2] = input[2] 
    
@cuda.jit(device = True)
def normal_from_ray(ray_dir, a, b, c, output):
    output[0] = (b[1] - a[1])*(c[2] - a[2]) - (b[2] - a[2])*(c[1] - a[1])
    output[1] = (b[2] - a[2])*(c[0] - a[0]) - (b[0] - a[0])*(c[2] - a[2])
    output[2] = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    if dot_product(output, ray_dir) > 0:
        output[0] *= -1
        output[1] *= -1
        output[2] *= -1
    normalize(output)

@cuda.jit(device= True)
def reflection_ray(normal, dir):
    normalize(dir)
    dot_prodact = dot_product(normal, dir)
    dir[0] -= normal[0] * 2.0 * dot_prodact
    dir[1] -= normal[1] * 2.0 * dot_prodact
    dir[2] -= normal[2] * 2.0 * dot_prodact
    normalize(dir)
    
@cuda.jit(device = True)
def invert(input, output):
    if abs(input[0]) < 0.0001:
        output[0] = 1.0 / 0.0001
    else:
        output[0] = 1.0 / input[0]
    if abs(input[1]) < 0.0001:
        output[1] = 1.0 / 0.0001
    else:
        output[1] = 1.0 / input[1]
    if abs(input[2]) < 0.0001:
        output[2] = 1.0 / 0.0001
    else:
        output[2] = 1.0 / input[2]
    
@cuda.jit(device = True)
def position_at(ray_origin, ray_dir, t, output):
    output[0] = ray_origin[0] + ray_dir[0] * t
    output[1] = ray_origin[1] + ray_dir[1] * t
    output[2] = ray_origin[2] + ray_dir[2] * t
    
@cuda.jit(device = True)
def shifted_from_hit(hit_point, normal, output):
    output[0] = hit_point[0] + normal[0] * 0.00001
    output[1] = hit_point[1] + normal[1] * 0.00001
    output[2] = hit_point[2] + normal[2] * 0.00001

@cuda.jit(device=True)
def normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length > 0:
        v[0] /= length
        v[1] /= length
        v[2] /= length

@cuda.jit(device=True)
def intersect_box(origin, dir, bvh_border):
    inv_dir = cuda.local.array(3, dtype=np.float32)
    invert(dir, inv_dir)
    tx1 = (bvh_border[0, 0] - origin[0]) * inv_dir[0]
    tx2 = (bvh_border[1, 0] - origin[0]) * inv_dir[0]
    
    tmin = min(tx1, tx2)
    tmax = max(tx1, tx2)
    
    ty1 = (bvh_border[0, 1] - origin[1]) * inv_dir[1]
    ty2 = (bvh_border[1, 1] - origin[1]) * inv_dir[1]
    tmin = max(tmin, min(ty1, ty2))
    tmax = min(tmax, max(ty1, ty2))
    
    tz1 = (bvh_border[0, 2] - origin[2]) * inv_dir[2]
    tz2 = (bvh_border[1, 2] - origin[2]) * inv_dir[2]
    tmin = max(tmin, min(tz1, tz2))
    tmax = min(tmax, max(tz1, tz2))

    
    return tmax >= max(tmin, 0.0)

@cuda.jit(device = True)
def intersect_light(ray_origin, ray_direction, p, r):
    l = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        l[i] = p[i] - ray_origin[i]
    tca = dot_product(l, ray_direction)
    if tca < 0:
        return -1.0
    d2 = dot_product(l, l) - (tca * tca)
    radius2 = r * r
    if d2 > radius2:
        return -1.0
    thc = math.sqrt(radius2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    if t0 > 0.00001:
        return t0
    if t1 > 0.00001:
         return t1
    return -1.0
  
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

@cuda.jit(device=True)
def traverse_bvh(origin, dir, bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, output):
    stack = cuda.local.array(32, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    
    output[0] = 9999999.0 #current_t
    output[1] = -1.0 #index
    
    while stack_ptr >= 0:
        current_node = stack[stack_ptr]
        stack_ptr -= 1
        if intersect_box(origin, dir, bvh_border[current_node]):
            if bvh_child[current_node, 0] == -1:
                start = bvh_triangles_indexes[current_node, 0]
                end = bvh_triangles_indexes[current_node, 1]
                for i in range(start, start + end):
                    a = bvh_ordered_triangles_points[i, 0]
                    b = bvh_ordered_triangles_points[i, 1]
                    c = bvh_ordered_triangles_points[i, 2]
                    t = intersect_triangle(origin, dir, a, b, c)
                    if t > 0 and t < output[0]:
                        output[0] = t * 1.0
                        output[1] = i * 1.0  
            else:
                stack_ptr += 1
                stack[stack_ptr] = bvh_child[current_node, 0]
                stack_ptr += 1
                stack[stack_ptr] = bvh_child[current_node, 1]

@cuda.jit
def raytrace_kernel(max_depth, ray_origins, ray_dirs, triangles_points, triangles_color, lights_pos, lights_radius, lights_color, output_colors):
    x, y = cuda.grid(2)
    if x >= len(output_colors) or y >= len(output_colors[0]):
        return
    output_colors[x, y, 0] = 0
    output_colors[x, y, 1] = 0
    output_colors[x, y, 2] = 0
    
    origin = cuda.local.array(3, dtype=np.float32)
    dir = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        origin[i] = ray_origins[x, y, i]
        dir[i] = ray_dirs[x, y, i]
    
    ambient_intensity = 0.2
    attenuation = 1.0
    reflection = 0.3
    diffusion = 1.0 - reflection
    
    for depth in range(max_depth):
        closest_t = 9999999.0
        hit_index = -1
        hit_type = ""
        for i in range(len(triangles_points)):
            a = triangles_points[i, 0]
            b = triangles_points[i, 1]
            c = triangles_points[i, 2]
            t = intersect_triangle(origin, dir, a, b, c)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_index= i 
                hit_type = "t"
        
        for i in range(len(lights_pos)):
            p = lights_pos[i]
            r = lights_radius[i]
            t = intersect_light(origin, dir, p, r)
            if t > 0 and t < closest_t:
                closest_t = t
                hit_index = i
                hit_type = "l"
                
        if hit_type == "l":
            output_colors[x, y, 0] += lights_color[hit_index, 0] * attenuation
            output_colors[x, y, 1] += lights_color[hit_index, 1] * attenuation
            output_colors[x, y, 2] += lights_color[hit_index, 2] * attenuation
            break
        elif hit_type == "t":
            hit_point = cuda.local.array(3, dtype=np.float32)
            normal = cuda.local.array(3, dtype=np.float32)
            shadow_origin = cuda.local.array(3, dtype=np.float32)
            color = cuda.local.array(3, dtype=np.float32)
            shadow_dir = cuda.local.array(3, dtype=np.float32)
            position_at(origin, dir, closest_t, hit_point)
            normal_from_ray(dir, triangles_points[hit_index, 0], triangles_points[hit_index, 1], triangles_points[hit_index, 2], normal)
            shifted_from_hit(hit_point, normal, shadow_origin)
            color[0] = triangles_color[hit_index, 0] * ambient_intensity
            color[1] = triangles_color[hit_index, 1] * ambient_intensity
            color[2] = triangles_color[hit_index, 2] * ambient_intensity
            triangle_color = triangles_color[hit_index]
            for i2 in range(len(lights_pos)):
                shadow_dir[0] = lights_pos[i2, 0] - shadow_origin[0]
                shadow_dir[1] = lights_pos[i2, 1] - shadow_origin[1]
                shadow_dir[2] = lights_pos[i2, 2] - shadow_origin[2]
                normalize(shadow_dir)
                t_light = intersect_light(shadow_origin, shadow_dir, lights_pos[i2], lights_radius[i2])
                in_shadow = False
                for j in range(len(triangles_points)):
                    a = triangles_points[j, 0]
                    b = triangles_points[j, 1]
                    c = triangles_points[j, 2]
                    t_shadow = intersect_triangle(shadow_origin, shadow_dir, a, b, c)
                    if t_shadow > 0 and t_shadow < t_light:
                        in_shadow = True
                        break
                
                if not in_shadow:
                    diffuse_intensity = max(0.0, dot_product(normal, shadow_dir))
                    color[0] += triangle_color[0] * lights_color[i2, 0] * (diffuse_intensity  / len(lights_pos))
                    color[1] += triangle_color[1] * lights_color[i2, 1] * (diffuse_intensity / len(lights_pos))
                    color[2] += triangle_color[2] * lights_color[i2, 2] * (diffuse_intensity  / len(lights_pos))
                    output_colors[x, y, 0] += (diffusion * attenuation * color[0])
                    output_colors[x, y, 1] += (diffusion * attenuation * color[1])
                    output_colors[x, y, 2] += (diffusion * attenuation * color[2])
        else:
            break
        attenuation = attenuation * reflection
        copy(hit_point, origin)
        reflection_ray(normal, dir)
        
    output_colors[x, y, 0] = min(output_colors[x, y, 0] * 255, 255)
    output_colors[x, y, 1] = min(output_colors[x, y, 1] * 255, 255)
    output_colors[x, y, 2] = min(output_colors[x, y, 2] * 255, 255)

def render_scene_gpu(n, max_depth,  ray_origin_list, ray_dir_list, triangle_points_list, triangles_color, light_pos_list, light_radius_list, lights_color):
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
    np_triangle_points_list = np.array(triangle_points_list, dtype= np.float32)
    np_triangles_color = np.array(triangles_color, dtype= np.float32)
    np_light_pos_list = np.array(light_pos_list, dtype= np.float32)
    np_light_radius_list = np.array(light_radius_list, dtype= np.float32)
    np_lights_color = np.array(lights_color, dtype= np.float32)
    np_output_colors = np.array(output_colors, dtype= np.float32)
    
    
    d_origins = cuda.to_device(np_ray_origin_list)
    d_dirs = cuda.to_device(np_ray_dir_list)
    d_triangles_points = cuda.to_device(np_triangle_points_list)
    d_triangles_colors = cuda.to_device(np_triangles_color)
    d_lights_position = cuda.to_device(np_light_pos_list)
    d_lights_radius = cuda.to_device(np_light_radius_list)
    d_np_lights_color = cuda.to_device(np_lights_color)
    d_output = cuda.to_device(np_output_colors)
    
    threads_per_block = (n, n)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    raytrace_kernel[blocks_per_grid, threads_per_block](
        int(max_depth), d_origins, d_dirs, d_triangles_points, d_triangles_colors , d_lights_position, d_lights_radius, d_np_lights_color, d_output
    )
    
    cuda.synchronize()
    d_output.copy_to_host(np_output_colors)
    return np_output_colors
    
   