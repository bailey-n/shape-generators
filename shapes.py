import numpy as np
import math

def create_truncated_tetrahedron():
    tetrahedron = [
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1, -1.0, -1.0])
    ]

    faces = [
        [tetrahedron[0], tetrahedron[1], tetrahedron[2]],
        [tetrahedron[1], tetrahedron[0], tetrahedron[2]],
        [tetrahedron[3], tetrahedron[2], tetrahedron[1]],
        [tetrahedron[2], tetrahedron[3], tetrahedron[0]],
    ]

    edges_by_face = [
        [[0, 1], [1, 2], [2, 0]],
        [[0, 2], [2, 3], [3, 0]],
        [[0, 3], [3, 1], [1, 0]],
        [[1, 3], [3, 2], [2, 1]]
    ]

    vertex_exclusions = [3, 1, 2, 0]
    triangle_vertices = []
    triangle_indices = []
    hexagon_vertices = []
    for vertex, edges in zip(vertex_exclusions, edges_by_face):
        triangle = []
        for edge in edges[::-1]:
            e_coords = tetrahedron[edge[0]]
            v_coords = tetrahedron[vertex]
            triangle.append((2 * v_coords + e_coords))
        triangle_vertices.append(triangle)

        hexagon = []
        for edge in edges:
            root = tetrahedron[edge[0]]
            tip = tetrahedron[edge[1]]
            hexagon.append((2 * root + tip))
            hexagon.append((2 * tip + root))
        hexagon_vertices.append(hexagon)

    hexagon_centers = []
    for hexagon in hexagon_vertices:
        center = np.array([0.0, 0.0, 0.0])
        for vertex in hexagon:
            center += vertex
        center /= 6.0
        hexagon_centers.append(center)

    for triangle in triangle_vertices:
        print(triangle)
    for hexagon in hexagon_vertices:
        print(hexagon)
    for center in hexagon_centers:
        print(center)

    # Make obj
    with open("truncated_tetrahedron.obj", "w") as f:
        # Write vertices
        triangle_vertex_string = ""
        for triangle in triangle_vertices:
            for vertex in triangle[::-1]:
                triangle_vertex_string += f"v {' '.join(str(val) for val in vertex)}\n"
        f.write(triangle_vertex_string)
        hexagon_center_string = ""
        for center in hexagon_centers:
            hexagon_center_string += f"v {' '.join(str(val) for val in center)}\n"
        f.write(hexagon_center_string)
        hexagonal_vertex_string = ""
        for hexagon in hexagon_vertices:
            for vertex in hexagon[::-1]:
                hexagonal_vertex_string += f"v {' '.join(str(val) for val in vertex)}\n"
        f.write(hexagonal_vertex_string)

        # Write uv information
        f.write("\nvt 0.0 0.0 0.0\n")

        # Write normal information
        f.write("\nvn 1.0 0.0 0.0\n")

        # Write face information
        f.write("\n")
        triangle_face_string = "f 1/1/1 2/1/1 3/1/1\nf 4/1/1 5/1/1 6/1/1\nf 7/1/1 8/1/1 9/1/1\nf 10/1/1 11/1/1 12/1/1\n"
        f.write(triangle_face_string)

        hexagon_face_string = ""
        for i in range(len(hexagon_vertices)):
            center_index = i + 13
            hexagon_root_vertex = 17 + (6 * i)
            for j in range(5):
                hexagon_face_string += f"f {hexagon_root_vertex + j}/1/1 {hexagon_root_vertex + j + 1}/1/1 {center_index}/1/1\n"
            hexagon_face_string += f"f {hexagon_root_vertex + 5}/1/1 {hexagon_root_vertex}/1/1 {center_index}/1/1\n"
        f.write(hexagon_face_string)

def create_rectangular_prism():
    vertices, faces = create_prism(4)

    width = 1.0
    height = 1.0
    length = 5.0

    for vertex in vertices:
        vertex[0] *= width
        vertex[1] *= length
        vertex[2] *= height

    create_obj("rectangular_prism.obj", vertices, faces)

def create_cylinder():
    vertices, faces = create_prism(100)

    width = 1.0
    height = 1.0
    length = 5.0

    for vertex in vertices:
        vertex[0] *= width
        vertex[1] *= length
        vertex[2] *= height

    create_obj("cylinder.obj", vertices, faces)

def create_prism(sides: int):
    base_points = [(math.cos(i * 2 * math.pi / sides), math.sin(i * 2 * math.pi / sides)) for i in range(sides)]
    vertices = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])] # start w/ top and bottom centers
    faces = []
    num_vertices = 2 * len(base_points)
    for i, point in enumerate(base_points):
        vertices.append(np.array([point[0], 1.0, point[1]])) # top
        vertices.append(np.array([point[0], -1.0, point[1]])) # bottom
        tl = 2 * i + 2
        bl = 2 * i + 3
        tr = (2 * i + 2) % num_vertices + 2
        br = (2 * i + 3) % num_vertices + 2
        faces.append([tl, tr, bl]) # top-left side
        faces.append([br, bl, tr]) # bottom-left side
        faces.append([tr, tl, 0])  # top base slice
        faces.append([bl, br, 1]) # bottom base slice
    return vertices, faces

def create_obj(name: str, vertices: list[np.ndarray], faces: list[list[int]]):
    with open(name, "w") as f:
        for vertex in vertices:
            f.write(f"v {' '.join(str(val) for val in vertex)}\n")
        f.write("\nvt 0.0 0.0 0.0\n")
        f.write("\nvn 1.0 0.0 0.0\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {' '.join(str(index + 1) + '/1/1' for index in face)}\n")
        f.close()

if __name__ == "__main__":
    create_truncated_tetrahedron()
    create_rectangular_prism()
    create_cylinder()
