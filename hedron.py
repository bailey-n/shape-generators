import random

import numpy as np
import math

def generate_icosahedron() -> (np.ndarray, np.ndarray):
    faces_per_vertex = 5
    critical_angle = 2.0 * math.pi / faces_per_vertex
    cos_crit = math.cos(critical_angle)
    side_length = math.sqrt(2.0 - 2.0 * cos_crit)
    print(f"Side length: {side_length}\n")
    polar_offset = math.sqrt(1.0 - 2.0 * cos_crit)
    equatorial_offset = 0.5 * math.sqrt(2.0 * math.sqrt(0.5 + 0.5 * cos_crit) - 2.0 * cos_crit)

    unit_scale_factor = 1.0 / (polar_offset + equatorial_offset)

    # dimension = 3, vertex count = 12
    vertices = np.zeros((12, 3))
    # Set pole y values
    vertices[0][1] = unit_scale_factor * (polar_offset + equatorial_offset)
    # vertices[0][1] = unit_scale_factor * (equatorial_offset)
    vertices[11][1] = -vertices[0][1]
    # Set upper pentagon values
    for n in range(5):
        vertices[n + 1][0] = unit_scale_factor * math.cos(n * critical_angle)
        vertices[n + 1][1] = unit_scale_factor * equatorial_offset
        vertices[n + 1][2] = unit_scale_factor * math.sin(n * critical_angle)
    # Set lower pentagon values
    lower_offset_angle = critical_angle / 2
    for n in range(5):
        vertices[n + 6][0] = unit_scale_factor * math.cos(n * critical_angle + lower_offset_angle)
        vertices[n + 6][1] = unit_scale_factor * (- equatorial_offset)
        vertices[n + 6][2] = unit_scale_factor * math.sin(n * critical_angle + lower_offset_angle)

    # Create face indices
    indices = np.ones((20, 3), int)
    # Layer 1 triangles
    base_index: int = 1
    for n in range(5):
        indices[n][0] = base_index
        indices[n][1] = base_index + n + 1
        indices[n][2] = base_index + ((n + 1) % 5) + 1
    # Layer 2 triangles
    total_faces: int = 5
    for n in range(5):
        indices[n + total_faces][0] = base_index + n + 1
        indices[n + total_faces][1] = base_index + n + 6
        indices[n + total_faces][2] = base_index + ((n + 1) % 5) + 1
    # Layer 3 triangles
    total_faces = 10
    for n in range(5):
        indices[n + total_faces][0] = base_index + n + 1
        indices[n + total_faces][1] = base_index + ((n - 1) % 5) + 6
        indices[n + total_faces][2] = base_index + n + 6
    # Layer 4 triangles
    total_faces = 15
    for n in range(5):
        indices[n + total_faces][0] = base_index + 11
        indices[n + total_faces][1] = base_index + n + 6
        indices[n + total_faces][2] = base_index + ((n - 1) % 5) + 6

    return vertices, indices

def generate_dual(vertices: np.ndarray, indices: np.ndarray) -> (np.ndarray, np.ndarray):
    planes = assign_faces_to_planes(vertices, indices)
    polygons = extract_plane_edges(planes, indices)
    # random.shuffle(polygons)
    vertex_faces = find_shared_faces(polygons)
    # vertex_faces = {vert: face for vert, face in list(vertex_faces.items())[:2]}
    new_indices: np.ndarray = triangulate_vertex_faces(vertex_faces)
    new_vertices = find_polygon_centers(vertices, polygons)
    print(planes)
    print(polygons)
    print(vertex_faces)
    print(new_vertices)
    print(new_indices)
    return new_vertices, new_indices

def assign_faces_to_planes(vertices: np.ndarray, indices: np.ndarray) -> list[list[int]]:
    planes: list[list[int]] = []
    normals: list[np.ndarray] = []
    for i, face in enumerate(indices):
        A = vertices[face[0] - 1]
        B = vertices[face[1] - 1]
        C = vertices[face[2] - 1]
        new_normal = np.cross(B - A, C - A)
        new_normal /= np.linalg.norm(new_normal)
        found_match: bool = False
        for j, old_normal in enumerate(normals):
            # print(new_normal, old_normal)
            if all(np.isclose(new_normal, old_normal)):
                found_match = True
                planes[j].append(i)
                break
        if not found_match:
            normals.append(new_normal)
            planes.append([i])
    return planes

def extract_plane_edges(planes: list[list[int]], indices: np.ndarray) -> list[list[int]]:
    plane_edges: list[list[int]] = []
    for plane in planes:
        edges: list[tuple[int, int]] = []
        for face_index in plane:
            vertex_indices = indices[face_index]
            potential_edges: list[tuple[int, int]] = [
                (vertex_indices[0], vertex_indices[1]),
                (vertex_indices[1], vertex_indices[2]),
                (vertex_indices[2], vertex_indices[0])
            ]
            for edge in potential_edges:
                if edge in edges in edges:
                    edges.remove(edge)
                    continue
                if edge[::-1] in edges:
                    edges.remove(edge[::-1])
                    continue
                edges.append(edge)
        starting_index = edges[0][0]
        current_index = edges[0][1]
        directed_edges: list[int] = [starting_index]
        edges.remove(edges[0])
        while current_index != starting_index:
            directed_edges.append(current_index)
            for edge in edges:
                if edge[0] == current_index:
                    current_index = edge[1]
                    break
            edges.remove((directed_edges[-1], current_index))
        plane_edges.append(directed_edges)
    return plane_edges

def find_polygon_centers(vertices: np.ndarray, polygons: list[list[int]]) -> np.ndarray:
    dimensions = (len(polygons), np.size(vertices, axis=1))
    polygon_centers = np.ndarray(dimensions)
    for i, polygon in enumerate(polygons):
        for axis in range(3):
            polygon_centers[i][axis] = np.average([vertices[index - 1][axis] for index in polygon])
    return polygon_centers

def find_shared_faces(polygons: list[list[int]]) -> dict[int, list[int]]:
    unique_vertices = get_unique_vertex_references(polygons)
    shared_faces: dict[int, list[int]] = dict()
    # Find the faces
    for vertex in unique_vertices:
        incident_faces = []
        for i, polygon in enumerate(polygons):
            if vertex in polygon:
                incident_faces.append(i)
        shared_faces[vertex] = incident_faces
    # Order them to maintain counterclockwise directionality
    for vertex, faces in shared_faces.items():
        # Get vertex counterclockwise from the target vertex in the first face
        cc_vertex = polygons[faces[0]][polygons[faces[0]].index(vertex) - 1]
        for i in range(len(faces) - 1):
            current_face_index = i + 1
            # Find the face which shares this vertex
            for j, face_index in enumerate(shared_faces[vertex][current_face_index:]):
                if cc_vertex in polygons[face_index]:
                    # Get the next target vertex in the sequence
                    cc_vertex = polygons[face_index][polygons[face_index].index(vertex) - 1]
                    if j != 0:
                        # Swap the faces if it is not in the correct order
                        shared_faces[vertex][current_face_index], shared_faces[vertex][current_face_index + j] = \
                            shared_faces[vertex][current_face_index + j], shared_faces[vertex][current_face_index]
                    break
    return shared_faces

def get_unique_vertex_references(polygons: list[list[int]]) -> list[int]:
    unique_indices = []
    for polygon in polygons:
        for index in polygon:
            if index not in unique_indices:
                unique_indices.append(index)
    return unique_indices

def triangulate_vertex_faces(vertex_faces: dict[int, list[int]]) -> np.ndarray:
    # Number of triangles is sum of (two less than the number of vertices in each polygon)
    num_triangles = sum([len(indices) for indices in vertex_faces.values()]) - 2 * len(vertex_faces.items())
    triangles = np.ndarray((num_triangles, 3), int)
    triangle_count = 0
    for polygon in vertex_faces.values():
        triangles[triangle_count][0] = polygon[0] + 1
        triangles[triangle_count][1] = polygon[1] + 1
        triangles[triangle_count][2] = polygon[2] + 1
        triangle_count += 1
        if len(polygon) < 4:
            continue
        vertex_count = 3
        while vertex_count < len(polygon):
            triangles[triangle_count][0] = polygon[0] + 1
            triangles[triangle_count][1] = polygon[vertex_count - 1] + 1
            triangles[triangle_count][2] = polygon[vertex_count] + 1
            triangle_count += 1
            vertex_count += 1
    return triangles


def compute_norm(vtx: np.ndarray, face_idx: np.ndarray) -> np.ndarray:
    # For ABC, B-A x C-A
    A = vtx[face_idx[0] - 1]
    B = vtx[face_idx[1] - 1]
    C = vtx[face_idx[2] - 1]
    # print("Face side lengths: ")
    # print(np.linalg.norm(B-A))
    # print(np.linalg.norm(C-A))
    # print(np.linalg.norm(C-B))
    # print()
    normal_vector = np.cross(B-A, C-A)
    return normal_vector / np.linalg.norm(normal_vector)

def create_obj(name: str, vtx: np.ndarray, idx: np.ndarray):
    with open(name, "w") as f:
        for vertex in vtx:
            f.write(f"v {' '.join(str(val) for val in vertex)}\n")
        f.write("\nvt 0.0 0.0 0.0\n")
        f.write("\n")
        for face in idx:
            norm = compute_norm(vtx, face)
            f.write(f"vn {' '.join(str(val) for val in norm)}\n")
        f.write("\n")
        for i, face in enumerate(idx):
            f.write(f"f {' '.join(str(index) + '/1/' + str(i + 1) for index in face)}\n")
        f.close()

if __name__ == "__main__":
    verts, inds = generate_icosahedron()
    create_obj("icosahedron.obj", verts, inds)
    # np.random.shuffle(inds)
    dod_verts, dod_inds = generate_dual(verts, inds)
    create_obj("dodecahedron.obj", dod_verts, dod_inds)
    # np.random.shuffle(dod_inds)
    ico2_verts, ico2_inds = generate_dual(dod_verts, dod_inds)
    create_obj("icosahedron2.obj", ico2_verts, ico2_inds)