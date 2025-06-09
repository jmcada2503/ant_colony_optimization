import numpy as np
import numpy.typing as ntp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass


@dataclass
class Ant:
    path: list[int]
    path_length: float
    visited: ntp.ArrayLike


def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


points = np.array([
    [0.13549547, 0.66202045],
    [0.47046747, 0.02456593],
    [0.34690738, 0.98025478],
    [0.41461822, 0.32904476],
    [0.34483581, 0.8041709],
    [0.22544901, 0.19767214],
    [0.59941989, 0.19204002],
    [0.21255983, 0.63456189],
    [0.72419519, 0.93991517],
    [0.7139072, 0.07865282]
])
n_ants = 10
n_iterations = 50
alpha = 1
beta = 1
evaporation_rate = 0.5
Q = 1

n_points = len(points)
pheromone = np.ones((n_points, n_points))
best_path = []
best_path_length = np.inf

iteration_paths: list[list[list[int]]] = []

for iteration in range(n_iterations):
    paths = []
    path_lengths = []

    # Initialize the ants
    ants = []
    for ant in range(n_ants):
        visited = np.array([False]*n_points)
        visited[0] = True
        ants.append(Ant(
            path=[0],
            path_length=0,
            visited=visited
        ))

    for _ in range(n_points-1):
        for ant in ants:
            unvisited = np.where(np.logical_not(ant.visited))[0]
            probabilities = np.zeros(len(unvisited))

            for i, unvisited_point in enumerate(unvisited):
                probabilities[i] = (
                    pheromone[ant.path[-1], unvisited_point]**alpha /
                    distance(
                        points[ant.path[-1]],
                        points[unvisited_point]
                    )**beta
                )

            probabilities /= np.sum(probabilities)

            next_point = np.random.choice(unvisited, p=probabilities)
            ant.path_length += distance(
                points[ant.path[-1]],
                points[next_point]
            )
            ant.path.append(next_point)
            ant.visited[next_point] = True

        iteration_paths.append([ant.path.copy() for ant in ants])

    for ant in ants:
        paths.append(ant.path)
        path_lengths.append(ant.path_length)

        if ant.path_length < best_path_length:
            best_path = ant.path
            best_path_length = ant.path_length

    pheromone *= evaporation_rate

    for path, path_length in zip(paths, path_lengths):
        for i in range(n_points-1):
            pheromone[path[i], path[i+1]] += Q/path_length
        pheromone[path[-1], path[0]] += Q/path_length


def update(frame):
    ax.clear()
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='red', marker='o')

    # Plot each ant's path at this iteration
    for path in iteration_paths[frame]:
        path_coords = points[path]
        x_coords = path_coords[:, 0]
        y_coords = path_coords[:, 1]
        ax.plot(x_coords, y_coords, linestyle='-', alpha=0.5)
    ax.set_title(f"Frame {frame+1}")
    return ax


fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig,
    update,  # type: ignore
    frames=len(iteration_paths),
    interval=200,
    blit=False,
    repeat=False
)

plt.show()
