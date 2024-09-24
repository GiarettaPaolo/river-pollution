from mesh import Triangulation
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class FactoryMesh:
    def __init__(self, L: float, W: float, FACTORY_START: float, FACTORY_END: float, h: float, h_bound: float, h_factory: float, std_dev_boundary: float, std_dev_factory: float):
        self.L = L
        self.W = W
        self.FACTORY_START = FACTORY_START
        self.FACTORY_END = FACTORY_END
        self.h = h
        self.h_bound = h_bound
        self.h_factory = h_factory
        self.std_dev_boundary = std_dev_boundary
        self.std_dev_factory = std_dev_factory
        self.mesh = self.generate_mesh()
        self.masks = self.generate_masks(self.mesh)
        self.masks_up, self.masks_down, self.masks_top, self.masks_bottom, self.masks_factory, self.masks_side = self.masks

    def generate_mesh(self) -> Triangulation:
        dist_from_boundary = lambda x, y: np.minimum.reduce([x, self.L - x, y, self.W - y])  # Distance from boundary
        dist_from_factory = lambda x, y: np.sqrt((y - self.W) ** 2 + np.maximum.reduce([self.FACTORY_START-x, x-self.FACTORY_END, np.zeros_like(x)]) ** 2)  # Distance from factory
        gaussian = lambda x, std_dev: np.exp(-x ** 2 / (2 * std_dev ** 2))

        mesh_size = lambda dim, tag, x, y, z, _: self.h + np.minimum((self.h_bound - self.h) * gaussian(dist_from_boundary(x, y), self.std_dev_boundary), \
                                  (self.h_factory - self.h) * gaussian(dist_from_factory(x, y), self.std_dev_factory)) 

        mesh_points = np.array([[0, 0], [self.L, 0], [self.L, self.W], [0, self.W]])

        return Triangulation.from_polygon(mesh_points, mesh_size=mesh_size)

    def generate_masks(self, mesh: Triangulation) -> List[np.ndarray]:
        lines = mesh.lines
        points = mesh.points[lines]

        up_mask = (np.abs(points[:, :, 0]) < 1e-10).all(axis=1)
        down_mask = (np.abs(points[:, :, 0] - self.L) < 1e-10).all(axis=1) 
        top_mask = (np.abs(points[:, :, 1] - self.W) < 1e-10).all(axis=1)
        bottom_mask = (np.abs(points[:, :, 1]) < 1e-10).all(axis=1)
        factory_mask = top_mask &  np.logical_and(points[:, :, 0] >= self.FACTORY_START - 1e-10, points[:, :, 0] <= self.FACTORY_END + 1e-10).all(axis=-1)
        side_mask = ~(up_mask | down_mask | factory_mask)

        return up_mask, down_mask, top_mask, bottom_mask, factory_mask, side_mask

    def plot(self, block: bool = False):
        lines = self.mesh.lines
        triangles = self.mesh.triangles

        _, ax = plt.subplots()

        ax.set_aspect('equal')
        up_mask, down_mask, _, _, factory_mask, side_mask = self.masks
        ax.triplot(*self.mesh.points.T, triangles=triangles, color='blue', linewidth=0.5)
        ax.plot(*self.mesh.points[lines[up_mask]].T, color='red', linewidth=2)
        ax.plot(*self.mesh.points[lines[down_mask]].T, color='yellow', linewidth=2)
        ax.plot(*self.mesh.points[lines[factory_mask]].T, color='green', linewidth=2)
        ax.plot(*self.mesh.points[lines[side_mask]].T, color='black', linewidth=2)

        plt.show()
