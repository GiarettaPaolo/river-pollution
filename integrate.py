from util import np, _
from quad import QuadRule
from mesh import Triangulation

from scipy import sparse
from typing import Iterable, Callable


def shape2D_LFE(quadrule: QuadRule) -> np.ndarray:
  """
    Return the shape functions evaluated in the quadrature points
    associated with ``quadrule`` for the three local first order
    lagrangian finite element basis functions (hat functions) over the
    unit triangle.

    Parameters
    ----------
    quadrule: :class: QuadRule with quadrule.simplex_type == 'triangle'

    Returns
    -------
    x: np.ndarray
      evaluation of the three local basis functions in the quad points.
      Has shape (nquadpoints, 3)
  """

  assert quadrule.simplex_type == 'triangle'
  points = quadrule.points

  # points = [x, y], with shape (npoints, 2)
  # shapeF0 = 1 - x - y
  # shapeF1 = x
  # shapeF2 = y
  # so we construct the (npoints, 3) matrix shapeF = [1 - x - y, x, y]
  # simply by concatenating 1 - x - y with points = [x, y] along axis 1
  return np.concatenate([ (1 - points.sum(1)).reshape([-1, 1]), points ], axis=1)


def grad_shape2D_LFE(quadrule: QuadRule) -> np.ndarray:
  """
    Return the local gradient of the shape functions evaluated in
    the quadrature points associated with ``quadrule`` for the three local
    first order lagrangian finite element basis functions (hat functions)
    over the unit triangle.

    Parameters
    ----------
    quadrule: :class: `QuadRule` with quadrule.simplex_type == 'triangle'

    Returns
    -------
    x : :class: `np.ndarray`
      evaluation of the three local basis functions in the quad points.
      Has shape (nquadpoints, 3, 2), where the first axis refers to the index
      of the quadrature point, the second axis to the index of the local
      basis function and the third to the component of the gradient.

      Example: the gradient of shape (2,) of the 2nd local basis function
               in the third quadrature point is given by x[2, 1, :]
               (0-based indexing).
  """
  assert quadrule.simplex_type == 'triangle'

  # number of quadrature points
  nP, = quadrule.weights.shape
  ones = np.ones((nP,), dtype=float)
  zeros = np.zeros((nP,), dtype=float)
  return np.moveaxis( np.array([ [-ones, -ones],
                                 [ones, zeros],
                                 [zeros, ones] ]), -1, 0)


def assemble_matrix_from_iterables(mesh: Triangulation, *system_matrix_iterables) -> sparse.csr_matrix:
  """ Assemble sparse matrix from triangulation and system matrix iterables.
      For examples, see end of the script. """

  triangles = mesh.triangles
  ndofs = len(mesh.points)

  A = sparse.lil_matrix((ndofs,)*2)

  for tri, *system_mats in zip(triangles, *system_matrix_iterables):

    # this line is equivalent to
    # for mat in system_mats:
    #   A[np.ix_(*tri,)*2] += mat
    A[np.ix_(*(tri,)*2)] += np.add.reduce(system_mats)

  return A.tocsr()


def assemble_rhs_from_iterables(mesh: Triangulation, *rhs_iterables) -> np.ndarray:
  """ Assemble right hand side from triangulation and local load vector iterables.
      For examples, see end of the script. """

  triangles = mesh.triangles
  ndofs = len(mesh.points)

  rhs = np.zeros((ndofs,), dtype=float)

  for tri, *local_rhss in zip(triangles, *rhs_iterables):
    rhs[tri] += np.add.reduce(local_rhss)

  return rhs


def mass_with_reaction_iter(mesh: Triangulation, quadrule: QuadRule, freact: Callable = None) -> Iterable:
  """
    Iterator for the mass matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    freact: :class: `Callable`
      Function representing the reaction term. Must take as argument a single
      array of shape quadrule.points.shape and return a :class: `np.ndarray`
      object either of shape arr.shape == quadrule.weights.shape or
                             arr.shape == (1,)
      The latter usually means that freact is constant.

    Example
    -------
    For an example, see the end of the script.
  """

  # freact not passed => take it to be constant one.
  if freact is None:
    freact = lambda x: np.array([1])

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for (a, b, c), BK, detBK in zip(mesh.points_iter(), mesh.BK, mesh.detBK):

    # define the global points by pushing forward the local quadrature points
    # from the reference element onto the current triangle
    x = qpoints @ BK.T + a[_]

    # this line is equivalent to
    # outer[i, j] = (weights * shapeF[:, i] * shapeF[:, j] * freact(x)).sum()
    # it's a tad faster because it's vectorised
    outer = (weights[:, _, _] * shapeF[..., _] * shapeF[:, _] * freact(x)[:, _, _]).sum(0)
    yield outer * detBK


def stiffness_with_diffusivity_iter(mesh: Triangulation, quadrule: QuadRule, fdiffuse: Callable = None) -> Iterable:
  """
    Iterator for the stiffness matrix, to be passed into `assemble_matrix_from_iterables`.

    Parameters
    ----------

    Exactly the same as in `mass_with_reaction_iter`.
    freact -> fdiffuse and has to be implemented in the exact same way.

    Example
    -------
    For an example, see the end of the script.
  """

  if fdiffuse is None:
    fdiffuse = lambda x: np.array([1])

  weights = quadrule.weights
  qpoints = quadrule.points
  grad_shapeF = grad_shape2D_LFE(quadrule)

  # loop over all points (a, b, c) per triangle and the correponding
  # Jacobi matrix and measure
  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]

    # evaluate the diffusivity in the global points.
    fdiffx = fdiffuse(x)

    # below an implementation using two for loops

    """
      mat = np.zeros((3, 3), dtype=float)

      for i in range(3):
        for j in range(i, 3):
          # y = grad_shapeF[:, i] is of shape (nquadpoints, 2) and wherein
          # y[j, k] represents the k-th component of \hat{\nabla} phi_i on the
          # j-th quadrature point.
          # The integral of \nabla phi_i \cdot \nabla phi_j is given by
          # \int BK^{-T} @ (\hat{\nabla} \phi_i) \cdot BK^{-T} @ (\hat{\nabla} phi_j) detBK dxi
          Gi = grad_shapeF[:, i] @ BKinv
          Gj = grad_shapeF[:, j] @ BKinv
          mat[i, j] = (weights * fdiffx * (Gi * Gj).sum(1)).sum() * detBK

      # add strictly upper triangular part transposed to mat
      mat += np.triu(mat, k=1).T
    """

    # these two lines are equivalent to all of the above
    grad_glob = (BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1)
    mat = ((weights * fdiffx)[:, _, _] * (grad_glob[..., _, :] * grad_glob[:, _]).sum(-1)).sum(0) * detBK

    yield mat


def transport_matrix_iter(mesh: Triangulation, quadrule: QuadRule, beta: Callable) -> Iterable:
  r"""
    \int -(\beta \cdot \nabla v) u

    Parameters
    ----------
    The same as for `mass_with_reaction_iter` and `stiffness_with_diffusivity_iter`
    but now the (nonoptional) `Callable` beta takes as input an array of shape
    (nquadpoints, 2) and either returns an array of shape (nquadpoints, 2)
    or (1, 2) where the latter means that the vector beta is constant.
  """

  weights = quadrule.weights
  qpoints = quadrule.points

  shapeF = shape2D_LFE(quadrule)
  grad_shapeF = grad_shape2D_LFE(quadrule)

  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]
    bx = beta(x)

    grad_glob_in_beta = ((BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1) * bx[:, _]).sum(-1)

    yield (weights[:, _, _] * grad_glob_in_beta[:, _] * shapeF[..., _]).sum(0) * detBK

def assemble_neumann_rhs(mesh: Triangulation, mask: np.ndarray, g: float = 1.00) -> np.ndarray:
  mask = np.asarray(mask, dtype=np.bool_)
  mask.shape == mesh.lines.shape[:1]

  local_neumann_load = g / 2.0 * np.ones(2, dtype=float)

  rhs = np.zeros(len(mesh.points), dtype=float)

  # retain only the boundary edges mesh.lines[i] if mask[i] is True
  neumann_lines = mesh.lines[mask]
  
  # loop over each line [index_of_a, index_of_b] and the corresponding points (a, b)
  for line, (a, b) in zip(neumann_lines, mesh.points[neumann_lines]):
    rhs[line] += local_neumann_load * np.linalg.norm(b - a) 

  return rhs

def assemble_neumann_mass_matrix(mesh: Triangulation, mask: np.ndarray, g: float = 1.00) -> np.ndarray:
  
  ndofs = len(mesh.points)
  A = sparse.lil_matrix((ndofs,)*2)
  
  local_neumann_load = g / 6.0 * np.array([[2, 1], [1, 2]], dtype=float)
  neumann_lines = mesh.lines[mask]
  for line, (a, b) in zip(neumann_lines, mesh.points[neumann_lines]):
    A[np.ix_(*(line, )*2)] += local_neumann_load * np.linalg.norm(b - a)    

  return A.tocsr()                               



def streamline_diffusion_stabilisation_iter(mesh: Triangulation, quadrule: QuadRule, beta: Callable, sigma: Callable, gamma: float = 1) -> Iterable:
  r"""
    Iterator assembling the streamline diffusion stabilisation matrix.

    \sum_K \int_K d_k (\beta \cdot \nabla phi_i) (\beta \cdot \nabla phi_j) 

    where d_K = \gamma h_K / |beta|_\infty

    Parameters
    ----------
    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    beta : :class: `Callable`
      The (nonoptional) `Callable` beta takes as input an array of shape
      (nquadpoints, 2) and either returns an array of shape (nquadpoints, 2)
      or (1, 2) where the latter means that the vector beta is constant.
    gamma : :class: `float`
      The (optional) stabilisation parameter gamma > 0 for use in `d_k` defaults to 1.
  """
  assert gamma > 0
  weights = quadrule.weights
  qpoints = quadrule.points

  grad_shapeF = grad_shape2D_LFE(quadrule)
  shapeF = shape2D_LFE(quadrule)

  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]
    bx = beta(x)
    sigx = sigma(x)

    hK = np.sqrt(detBK) / 2
    binf = np.abs(bx).max()
    dk = gamma * hK / binf

    grad_glob_in_beta = ((BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1) * bx[:, _]).sum(-1)

    yield dk * (weights[:, _, _] * (grad_glob_in_beta[:, _] + sigx[_, _] * shapeF[:, _]) * grad_glob_in_beta[..., _]).sum(0) * detBK


def transport_matrix_with_stabilisation_iter(mesh: Triangulation, quadrule: QuadRule, beta: Callable, gamma: float = 1) -> Iterable:
  """
    Same as `transport_matrix_iter` but now adds artificial streamline diffusion for stabilisation.

    Parameters
    ----------
    Same as before.
    gamma: float
      Stabilisation parameter. Defaults to one. Bigger => more artificial diffusion.
  """

  weights = quadrule.weights
  qpoints = quadrule.points

  shapeF = shape2D_LFE(quadrule)
  grad_shapeF = grad_shape2D_LFE(quadrule)

  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]
    bx = beta(x)

    grad_glob_in_beta = ((BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1) * bx[:, _]).sum(-1)
    hK = np.sqrt(detBK) / 2
    binf = np.abs(bx).max()
    
    # goes in front of the stabilisation integral
    C = gamma * hK / binf

    yield (weights[:, _, _] * (-grad_glob_in_beta[:, _] * shapeF[..., _] + C * grad_glob_in_beta[:, _] * grad_glob_in_beta[..., _])).sum(0) * detBK


def poisson_rhs_iter(mesh: Triangulation, quadrule: QuadRule, f: Callable) -> Iterable:

  """
    Iterator for assembling the right-hand side corresponding to
    \int f(x) phi_i dx.

    To be passed into the `assemble_rhs_from_iterables` function.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    f : :class: `Callable`
      Function representing the right hand side as a function of the position.
      Must take as input a vector of shape (nquadpoints, 2) and return either
      a vector of shape (nquadpoints,) or (1,).
      The latter means f is constant.
  """

  weights = quadrule.weights
  qpoints = quadrule.points
  shapeF = shape2D_LFE(quadrule)

  for (a, b, c), BK, detBK in zip(mesh.points_iter(), mesh.BK, mesh.detBK):

    # push forward of the local quadpoints (c.f. mass matrix with reaction term).
    x = qpoints @ BK.T + a[_]

    # rhs function f evaluated in the push-forward points
    fx = f(x)

    yield (shapeF * (weights * fx)[:, _]).sum(0) * detBK


def supg_rhs_iter(mesh: Triangulation, quadrule: QuadRule, f: Callable, beta: Callable, gamma: float = 1) -> Iterable:
  """
    Iterator for assembling the SUPG right-hand side corresponding to
    sum_K \int_K d_k f(x) \nabla phi_i \cdot \beta dx.

    where d_k as defined before.

    To be passed into the `assemble_rhs_from_iterables` function.

    Parameters
    ----------

    mesh : :class:`Triangulation`
      An instantiation of the `Triangulation` class, representing the mesh.
    quadrule : :class: `QuadRule`
      Instantiation of the `QuadRule` class with fields quadrule.points and
      quadrule.weights. quadrule.simplex_type must be 'triangle'.
    f : :class: `Callable`
      Function representing the right hand side as a function of the position.
      Must take as input a vector of shape (nquadpoints, 2) and return either
      a vector of shape (nquadpoints,) or (1,).
      The latter means f is constant.
  """
  assert gamma > 0
  weights = quadrule.weights
  qpoints = quadrule.points

  grad_shapeF = grad_shape2D_LFE(quadrule)

  for (a, b, c), BK, BKinv, detBK in zip(mesh.points_iter(), mesh.BK, mesh.BKinv, mesh.detBK):

    x = qpoints @ BK.T + a[_]
    bx = beta(x)
    fx = f(x)

    hK = np.sqrt(detBK) / 2
    binf = np.abs(bx).max()
    dk = gamma * hK / binf

    grad_glob_in_beta = ((BKinv.T[_, _] * grad_shapeF[..., _, :]).sum(-1) * bx[:, _]).sum(-1)

    yield dk * (grad_glob_in_beta * (weights * fx)[:, _]).sum(0) * detBK


if __name__ == '__main__':
  from matplotlib import pyplot as plt
  from quad import seven_point_gauss_6
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=0.1)
  mesh.plot()

  quadrule = seven_point_gauss_6()

  # assemble mass matrix with reaction term r(x) = 1 + ||x||^2
  # we do this by passing the corresponding iterator into the
  # ``assemble_matrix_from_iterators`` function.

  # create the iterator
  M_iter = mass_with_reaction_iter(mesh, quadrule, freact=lambda x: 1 + (x**2).sum(1))

  # pass into the assembly routine
  M = assemble_matrix_from_iterables(mesh, M_iter)

  plt.spy(M.todense())
  plt.show()

  # we can also pass more than one iterator into ``assemble_matrix_from_iterators``.
  # He will then take the sum of the system matrices generated by the iterators.
  S_iter = stiffness_with_diffusivity_iter(mesh, quadrule)
  M_iter = mass_with_reaction_iter(mesh, quadrule)
  S = assemble_matrix_from_iterables(mesh, S_iter, M_iter)

  plt.spy(S.todense())
  plt.show()
