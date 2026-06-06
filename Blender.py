from Vector import *
from Triangle import *
def extract_triangles(path):
    vertices = []
    faces = []
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            if not parts: 
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                parts = line.split()
                a_idx = int(parts[1].split('/')[0]) - 1
                b_idx = int(parts[2].split('/')[0]) - 1
                c_idx = int(parts[3].split('/')[0]) - 1
                faces.append([a_idx, b_idx, c_idx])
                if len(parts) == 5:
                    d_idx = int(parts[4].split('/')[0]) - 1
                    faces.append([a_idx, c_idx, d_idx])
    return vertices, faces

def create_all_triangles(vertices, faces):
    positions = []
    for _, v in enumerate(vertices):
        position = Vector(v[0], v[1], v[2])
        positions.append(position)
    triangles = [] 
    colors = [Vector(1, 0, 0), Vector(0, 1, 0)]
    j = 0
    i = 0
    for _, f in enumerate(faces):
        triangle = Triangle(positions[f[0]], positions[f[1]], positions[f[2]], color = Vector(1,1,1))
        triangles.append(triangle)
        i += 1
        if(i >= len(faces) // 4):
            i = 0
            j += 1
    return triangles 
    
    