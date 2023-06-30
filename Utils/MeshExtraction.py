import torch
import mcubes
import open3d as o3d
import numpy as np


class SimpleMeshNeRF:
    """
    This class implements a simple meshing system for the NeRF model
    """
    def __init__(self, device):
        self.device = device

    @staticmethod
    def generate_grid(grid_res, grid_scale):
        """
        A function for generation of the meshing grid for the ray marching algorithm
        :param grid_res: an int parameter that defines the resolution of the grid
        :param grid_scale: a float parameter that defines the scale of the grid
        :return: a torch tensor with generated coordinates
        """

        grid_x = torch.linspace(-grid_scale, grid_scale, grid_res)
        grid_y = torch.linspace(-grid_scale, grid_scale, grid_res)
        grid_z = torch.linspace(-grid_scale, grid_scale, grid_res)
        x, y, z = torch.meshgrid((grid_x, grid_y, grid_z))
        xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)

        return xyz

    def generate_mesh(self, model, grid_res, grid_scale, mcube_filter_thres):
        """
        A function for generating the mesh from NeRF model using ray marching algorithm
        :param model: a torch.nn.Module object with trained model
        :param grid_res: an int parameter that defines the resolution of the grid
        :param grid_scale: a float parameter that defines the scale of the grid
        :param mcube_filter_thres: a float multiplier that controls the noise reduction during meshing using marching cubes
        :return: a numpy ndarray (nx3) with mesh vertices and a numpy ndarray (nx3) with mesh triangular faces
        """

        xyz = self.generate_grid(grid_res, grid_scale)
        with torch.no_grad():
            _, density = model.forward(xyz.to(self.device), torch.zeros_like(xyz).to(self.device))

        density = density.detach().cpu().numpy().reshape(grid_res, grid_res, grid_res)
        vertices, triangles = mcubes.marching_cubes(density, mcube_filter_thres * np.mean(density))

        # cleaning the mesh
        print("[INFO] Filtering the mesh.")
        vertices = o3d.utility.Vector3dVector(vertices)
        triangles = o3d.utility.Vector3iVector(triangles)
        mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
        mesh.compute_vertex_normals()
        idxs, count, _ = mesh.cluster_connected_triangles()
        max_cluster_idx = np.argmax(count)

        triangles_to_remove = [i for i in range(len(triangles)) if idxs[i] != max_cluster_idx]
        mesh.remove_triangles_by_index(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.triangles).astype(np.int32)

        return vertices, triangles

    @staticmethod
    def render_mesh(vertices, triangles):
        """
        A function for rendering the mesh using Open3D functionality
        :param vertices: a numpy ndarray (nx3) with mesh vertices
        :param triangles: a numpy ndarray (nx3) with mesh triangular faces
        """

        vertices = o3d.utility.Vector3dVector(vertices)
        triangles = o3d.utility.Vector3iVector(triangles)
        mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
