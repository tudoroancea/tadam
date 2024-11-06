from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def norm(x):
    return np.sqrt(np.sum(x**2, axis=-1))


def normalize(x):
    return x / norm(x)[:, None]


def simple_retraction(x: np.ndarray, u: np.ndarray):
    return normalize(x + u)


def cayley_retraction(x: np.ndarray, u: np.ndarray):
    """
    :param x: shape (n,)
    :param u: shape (B,n)
    """
    x = x[None, :]
    O = u[:, :, None] * x[:, None, :] - x[:, :, None] * u[:, None, :]
    C = np.linalg.solve(np.eye(x.shape[1]) - 0.5 * O, np.eye(x.shape[1]) + 0.5 * O)
    return np.squeeze(C @ x[:, :, None])
    # O = np.outer(u, x) - np.outer(x, u)
    # C = np.linalg.solve(np.eye(len(x)) - 0.5 * O, np.eye(len(x)) + 0.5 * O)
    # ic(O, C)
    # return C @ x


def plot_sphere(ax: Axes):
    # set matpltolib 3d view to elevation 28°, azimuth -74°
    ax.view_init(elev=20, azim=16)
    # theta = np.linspace(0.0, np.pi, 50)
    # phi = np.linspace(0.0, 2 * np.pi, 50)
    # theta, phi = np.meshgrid(theta, phi)
    # theta = theta.ravel()
    # phi = phi.ravel()
    # x = np.sin(theta) * np.cos(phi)
    # y = np.sin(theta) * np.sin(phi)
    # z = np.cos(theta)
    # ax.scatter(x, y, z, s=0.5, alpha=0.9)
    # ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1)

    # now also plot wireframe
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)


def plot_vector(origin: np.ndarray, vector: np.ndarray, ax: Axes, *args, **kwargs):
    # plot a vector from origin to origin + vector
    plt.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], *args, **kwargs)


def main():
    # n = 3
    # x = normalize(np.random.randn(n))
    # u = (np.eye(3) - np.outer(x, x)) @ np.random.randn(n)
    t = np.linspace(0, 2, 50)
    x = np.array([0, 0, 1])
    us = np.array([1, 0, 0])
    uc = np.array([0, 1, 0])

    xs = simple_retraction(x, t[:, None] * us[None, :])
    xc = cayley_retraction(x, t[:, None] * uc[None, :])
    assert np.allclose(norm(xs), 1.0)
    assert np.allclose(norm(xc), 1.0)

    # plot everything in 3d
    plt.figure().add_subplot(projection="3d")
    ax = plt.gca()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")
    ax.scatter(x[0], x[1], x[2], label="x", s=10)
    plot_vector(x, us, ax, color="g", label="simple")
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], s=10, c=t, cmap="jet")
    plot_vector(x, uc, ax, color="r", label="cayley")
    ax.scatter(xc[:, 0], xc[:, 1], xc[:, 2], s=10, c=t, cmap="jet")
    # plot the surface of the sphere
    plot_sphere(plt.gca())
    # add color map
    plt.colorbar()
    # other
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
