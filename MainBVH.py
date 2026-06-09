from Classes.Ray import *
from Classes.Vector import *
from Classes.Triangle import *
from Classes.Light import *
from Classes.Camera import *
import Classes.BVHNode as BVHNode
import Classes.Blender as Blender
import Classes.Image as Image

def tracing_ray_bvh(light_list, triangle_bvh, ray, MAX_DEPTH):
    current_ray = ray
    final_color = Vector(0, 0, 0)
    attenuation = 1.0
    ambient_intensity = 0.2
    epsilon = 1e-4
    reflection = 0.3
    diffusion = 1.0 - reflection
    
    for depth in range(MAX_DEPTH):
        closest_t = float('inf')
        hit_object = None
        hit_type = None
        
        collides_triangle, t_triangle = BVHNode.hit_bvh(triangle_bvh , current_ray)
        if collides_triangle:
            closest_t = t_triangle
            hit_object = collides_triangle
            hit_type = 'triangle'
        
        for light in light_list:
            collides, t = light.collision(current_ray)
            if collides and epsilon < t < closest_t:
                closest_t = t
                hit_object = light
                hit_type = 'light'
            
        if hit_object is None:
            break
          
        if hit_type == 'light':
            final_color = final_color + (hit_object.color * attenuation)
            break
            
        elif hit_type == 'triangle':
            hit_point = current_ray.position_at(closest_t)
            normal = hit_object.normal_from_ray(current_ray)
            shifted_point = hit_point + (normal * epsilon)
            color = hit_object.color * ambient_intensity
            for light in light_list:
                shadow_ray = Ray(origin=shifted_point, direction=light.position - shifted_point)
                
                collides_shadow_triangle, t_shadow_triangle = BVHNode.hit_bvh(triangle_bvh , shadow_ray)
                collides_light, t_light = light.collision(shadow_ray)
                in_shadow = collides_shadow_triangle and t_shadow_triangle < t_light

                if not in_shadow:
                    diffuse_intensity = max(0.0, normal * (shadow_ray.direction))
                    reflected_color = Vector(hit_object.color.x * light.color.x,
                                            hit_object.color.y * light.color.y,
                                            hit_object.color.z * light.color.z)
                    color += reflected_color * (diffuse_intensity / len(light_list))
                    final_color += (color * diffusion * attenuation)
        current_ray = hit_object.reflection(current_ray, closest_t, normal)
        attenuation = attenuation * reflection
    return final_color
        

def tracing_rays_bvh(light_list, triangle_bvh, rays_matrix, W, H, MAX_DEPTH):
    hit_list = []
    for x in range(W):
        hot_list_y = []
        for y in range(H):
            collide = tracing_ray_bvh(light_list, triangle_bvh, rays_matrix[x][y], MAX_DEPTH)
            hot_list_y.append(collide)
        hit_list.append(hot_list_y)
    return hit_list


def start_engine(W, H, MAX_DEPTH, camera, lights, name, adding_to_image):
    vertices, faces = Blender.extract_triangles(rf"input\{name}.obj")
    triangles = Blender.create_all_triangles(vertices, faces)
    triangles_bvh = BVHNode.build_bvh(triangles, 2)
    rays_xy = camera.all_rays()
    hit_list = tracing_rays_bvh(lights, triangles_bvh, rays_xy, W, H, MAX_DEPTH)
    Image.create_image(W, H, hit_list, f"{name}_{adding_to_image}")

def main():
    W = 1920
    H = 1080
    MAX_DEPTH = 1
    name = "monkey"
    adding_to_image = "Test_BVH"
    camera = Camera(W, H, origin= Vector(0, 0, -3), target=Vector(0, 0, -2), up_vector=Vector(0, 1, 0), FOV=90)
    lights = [Light(Vector(10,0,0), 1), Light(Vector(-10,0,0), 1), Light(Vector(0,-10,0), 1), Light(Vector(0,10,0), 1), Light(Vector(0, 0, 10), 1), Light(Vector(0, 0, -10), 1)]
    start_engine(W, H, MAX_DEPTH,  camera, lights, name, adding_to_image)

if __name__ == "__main__":
    main()