import MainFastCuda
import pygame
import numpy as np
import sys
import time
from Classes.Camera import *
from Classes.Light import *
from Classes.Vector import *
import math

def rotate_vector(vector, axis, angle):
    axis = axis.normalize()
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cross = axis.cross(vector)
    dot = axis * vector
    return vector * cos_angle + cross * sin_angle + dot * (1-cos_angle) * axis

W =1280
H = 720
NAME = "donut"
UNIVERSAL_UP_VECTOR = Vector(0, 1, 0)
MOVE_SPEED = 0.05
SENSITIVITY = 0.002

position_vecotr = Vector(0, 0, -3)
forward_vector = Vector(0, 0, 1)
up_vector = Vector(0, 1, 0)
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"Python Raytracer: {NAME}")
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)


camera = Camera(W, H, origin= position_vecotr, target=position_vecotr + forward_vector, up_vector=up_vector, FOV=90)
lights = [camera.create_light_behind(1)]
bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color = MainFastCuda.setup_engine(camera, lights, NAME)

running = True
while running:
    mouse_dx, mouse_dy = pygame.mouse.get_rel()
    if mouse_dx != 0:
        yaw_angle = mouse_dx * SENSITIVITY
        forward_vector = rotate_vector(forward_vector, UNIVERSAL_UP_VECTOR, yaw_angle)
    if mouse_dy != 0:
        pitch_angle = -mouse_dy * SENSITIVITY
        proposed_forward = rotate_vector(forward_vector, right_vector, pitch_angle)
        if abs(np.dot(proposed_forward, UNIVERSAL_UP_VECTOR)) < 0.98:
            forward_vector = proposed_forward
    
    forward_vector = forward_vector.normalize()
    up_vector = up_vector.normalize()
    right_vector = forward_vector.cross(up_vector).normalize()  
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        position_vecotr += forward_vector * MOVE_SPEED
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        position_vecotr -= forward_vector * MOVE_SPEED
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        position_vecotr += right_vector * MOVE_SPEED
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        position_vecotr -= right_vector * MOVE_SPEED
    if keys[pygame.K_SPACE]:
        position_vecotr += UNIVERSAL_UP_VECTOR * MOVE_SPEED
    if keys[pygame.K_LCTRL]:
        position_vecotr -= UNIVERSAL_UP_VECTOR * MOVE_SPEED    
    if keys[pygame.K_ESCAPE]:
        running = False  

    camera = Camera(W, H, origin= position_vecotr, target=position_vecotr + forward_vector, up_vector=up_vector, FOV=90)
    lights = [camera.create_light_behind(1)]
    frame_grid = MainFastCuda.engine(camera,
                                bvh_border, bvh_child, bvh_triangles_indexes, bvh_ordered_triangles_points, bvh_ordered_triangles_color,
                                lights)
    np_frame_grid = np.array(frame_grid)
    np_frame_grid = (np_frame_grid).astype(np.uint8)
    pygame.surfarray.blit_array(screen, np_frame_grid)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

pygame.quit()
sys.exit()

