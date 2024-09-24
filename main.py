# Project file - Paolo Giaretta
import numpy as np
from util import np, _
from quad import seven_point_gauss_6
from FactoryMesh import FactoryMesh
import matplotlib.pyplot as plt
from integrate import mass_with_reaction_iter, stiffness_with_diffusivity_iter, \
                        assemble_matrix_from_iterables, transport_matrix_iter, \
                        assemble_rhs_from_iterables, streamline_diffusion_stabilisation_iter, \
                        assemble_neumann_mass_matrix, assemble_neumann_rhs
from solve import solve_with_dirichlet_data
import torch

plt.close('all')

# Parameters ######################################################################
L           = 10          # River length [m]
W           = 2           # River width [m]
FACTORY_START = 1         # Factory position from inlet [m]
FACTORY_END = 3           # Factory width [m]  
C_UP        = 1           # Upstream concentration [g/m^3]
C_DRY       = 1           # Natural concentration [g/m^3]
ALPHA       = 0.1         # Proportionality constant
U_M         = 10          # Maximum velocity [m/s]
SIG         = 0.5         # Reaction rate
MU          = 1e-6        # Diffusivity of pollutant

U_1  = np.vectorize(lambda y: U_M * (W - y) * y)                          # Inlet velocity field (streamwise)
C_IN = np.vectorize(lambda x: C_UP + 15 * (1- np.cos(np.pi * (x- 1))))    # Inlet concentration                                                

# plot U_1 and C_in
x = np.linspace(0, W, 100)
plt.figure(figsize=(6, 5))
plt.plot(x, U_1(x))
plt.xlabel('y')
plt.ylabel('$U_1$')
plt.title('Upstream velocity field')

x = np.linspace(FACTORY_START, FACTORY_END, 100)
plt.figure(figsize=(6, 5))
plt.plot(x, C_IN(x))
plt.xlabel('x')
plt.ylabel('$C_{in}$')
plt.title('Inlet concentration')

# Mesh Generation ################################################################
h = 2e-1                     # Mesh discretization parameter
h_bound = 2.5e-2             # Boundary discretization parameter
h_factory = 1e-2             # Factory discretization parameter
std_dev_boundary = 0.3       # Standard deviation for the Gaussian function
std_dev_factory  = 0.3       # Standard deviation for the Gaussian function

mesh_points = np.array([[0, 0], 
                        [L, 0], 
                        [L, W], 
                        [W, 0]])

# Quadrule ########################################################################
quadrule = seven_point_gauss_6()

# FUNCTIONS ####################################################################### 
def generate_mesh(h, h_bound, h_factory, std_dev_boundary, std_dev_factory):
    return FactoryMesh(L, W, FACTORY_START, FACTORY_END, h, h_bound, h_factory, std_dev_boundary, std_dev_factory)
    
def solve_no_stabilization(meshObj, quadrule, plot=False):
    # Boundary conditions ############################################################
    # Impose Dirichlet boundary conditions on the upstream and factory boundaries
    mesh = meshObj.mesh
    bindices = mesh.boundary_indices                                                         # all boundary indices
    data = np.zeros(bindices.shape, dtype=float)                                             # data array of same length containing (initially) only zeros
    data[meshObj.masks_factory] = C_IN(mesh.points[bindices[meshObj.masks_factory]][:, 0])   # set data array to C_IN on the upstream boundary
    data[meshObj.masks_up] = C_UP                                                            # set data array to C_UP on the factory boundary
    bindices = np.unique(bindices[meshObj.masks_factory | meshObj.masks_up])                 # the boundary vertices are the unique indices of the mesh's boundary edges restricted to the Dirichlet boundary
    data = data[meshObj.masks_factory | meshObj.masks_up]                                    # corresponding boundary points

    # Solve problem without stabilization ##############################################
    # Mass matrix
    Miter = mass_with_reaction_iter(mesh, quadrule, lambda _: np.array([SIG]))
    Mneumann = assemble_neumann_mass_matrix(mesh, meshObj.masks_side, ALPHA)

    # Stiffness matrix
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule, lambda _: np.array([MU]))

    # Transport matrix
    beta = lambda x: np.concatenate([U_1(x[:, 1, _]), np.zeros_like(x[:, 1, _])], axis=-1)
    Biter = transport_matrix_iter(mesh, quadrule, beta)

    # Right-hand side
    rhs = assemble_neumann_rhs(mesh, meshObj.masks_side, ALPHA * C_DRY)

    # Solve
    S = assemble_matrix_from_iterables(mesh, Miter, Aiter, Biter)
    solution = solve_with_dirichlet_data(S + Mneumann, rhs, bindices, data)

    if plot:
        # Plot solution
        plt.figure()
        plt.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles, solution, shading='flat', cmap='jet', edgecolors='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solution with no stabilization')
        plt.colorbar(label='Concentration', orientation='horizontal')
        # plt.clim(0.5, 1.5)
        plt.gca().set_aspect('equal')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
    return solution

def solve_SUPG_stabilization(meshObj, quadrule, gamma=5, plot=False):
    # Boundary conditions ############################################################
    # Impose Dirichlet boundary conditions on the upstream and factory boundaries
    mesh = meshObj.mesh
    bindices = mesh.boundary_indices                                                         # all boundary indices
    data = np.zeros(bindices.shape, dtype=float)                                             # data array of same length containing (initially) only zeros
    data[meshObj.masks_factory] = C_IN(mesh.points[bindices[meshObj.masks_factory]][:, 0])   # set data array to C_IN on the upstream boundary
    data[meshObj.masks_up] = C_UP                                                            # set data array to C_UP on the factory boundary
    bindices = np.unique(bindices[meshObj.masks_factory | meshObj.masks_up])                 # the boundary vertices are the unique indices of the mesh's boundary edges restricted to the Dirichlet boundary
    data = data[meshObj.masks_factory | meshObj.masks_up]                                    # corresponding boundary points

    # Solve problem without stabilization ##############################################
    # Mass matrix
    Miter = mass_with_reaction_iter(mesh, quadrule, lambda _: np.array([SIG]))
    Mneumann = assemble_neumann_mass_matrix(mesh, meshObj.masks_side, ALPHA)

    # Stiffness matrix
    Aiter = stiffness_with_diffusivity_iter(mesh, quadrule, lambda _: np.array([MU]))

    # Transport matrix
    beta = lambda x: np.concatenate([U_1(x[:, 1, _]), np.zeros_like(x[:, 1, _])], axis=-1)
    Biter = transport_matrix_iter(mesh, quadrule, beta)

    # Right-hand side
    rhs_neumann = iter(assemble_neumann_rhs(mesh, meshObj.masks_side, ALPHA * C_DRY))

    # Stabilization
    Stabiter = streamline_diffusion_stabilisation_iter(mesh, quadrule, beta, lambda _: np.array([SIG]), gamma=gamma)
    # rhs_stabiter = supg_rhs_iter(mesh, quadrule, lambda????, beta, gamma)

    # Solve
    rhs = assemble_rhs_from_iterables(mesh, rhs_neumann)
    S = assemble_matrix_from_iterables(mesh, Miter, Aiter, Biter, Stabiter)
    solution = solve_with_dirichlet_data(S + Mneumann, rhs, bindices, data)

    if plot:
        # Plot solution
        plt.figure()
        plt.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles, solution, shading='flat', cmap='jet', edgecolors='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solution with SUPG stabilization')
        plt.colorbar(label='Concentration', orientation='horizontal')
        # plt.clim(0.5, 1.5)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
    return solution

def integrate_concentration(solution, mesh):
    triangles = mesh.triangles
    detBK = mesh.detBK
    return (detBK[..., _] * solution[triangles]).sum() / 6


if __name__ == '__main__':

    meshObj = generate_mesh(h, h_bound, h_factory, std_dev_boundary, std_dev_factory)
    print('Number of vertices:', meshObj.mesh.points.shape[0])
    meshObj.plot()

    solution = solve_no_stabilization(meshObj, quadrule, plot=True)
    print('Concentration:', integrate_concentration(solution, meshObj.mesh))

    solution = solve_SUPG_stabilization(meshObj, quadrule, gamma=1, plot=True)
    print('Concentration:', integrate_concentration(solution, meshObj.mesh))
    # raise ValueError
    h_vect = np.linspace(1e-1, 1, 20)
    h_bound_vect = h_vect * 2.5 / 20
    h_factory_vect = h_vect / 20
    Qh = np.empty_like(h_vect)
    for i, (h, h_bound, h_factory) in enumerate(zip(h_vect, h_bound_vect, h_factory_vect)):
        meshObj = generate_mesh(h, h_bound, h_factory, std_dev_boundary, std_dev_factory)
        solution = solve_SUPG_stabilization(meshObj, quadrule)
        Qh[i] = integrate_concentration(solution, meshObj.mesh)
        print(f"Concentration for h={h: .4f}, h_bound={h_bound: .4f}, h_factory={h_factory: .4f}: Concentration: {Qh[i]: .4f}")

    plt.figure()
    plt.loglog(h_vect[1:], np.abs(Qh[1:]-Qh[0]), 'o-', label='error with final estimate')
    plt.loglog(h_vect[1:], 3*h_vect[1:], '--', label='first-order')
    plt.loglog(h_vect[1:], 10*h_vect[1:]**2, '--', label='second-order')
    plt.xlabel('h')
    plt.ylabel('Error')
    plt.title('Concentration-based error')
    plt.legend()
    plt.show()