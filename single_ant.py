import numpy as np
import matplotlib.pyplot as plt


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
n_ants = 1
n_iterations = 5
alpha = 1
beta = 1
evaporation_rate = 0.5
Q = 1

n_points = len(points)
pheromone = np.ones((n_points, n_points))
best_path = []
best_path_length = np.inf

for _ in range(n_iterations):
    paths = []
    path_lengths = []

    for ant in range(n_ants):
        visited = np.array([False]*n_points)
        current_point = np.random.randint(n_points)
        visited[current_point] = True
        path = [current_point]
        path_length = 0

        while False in visited:
            unvisited = np.where(np.logical_not(visited))[0]
            probabilities = np.zeros(len(unvisited))

            distances = []
            pheromones = []
            for i, unvisited_point in enumerate(unvisited):
                distances.append(
                    distance(
                        points[current_point],
                        points[unvisited_point]
                    )
                )
                pheromones.append(
                    pheromone[current_point, unvisited_point]
                )
                probabilities[i] = (
                    pheromone[current_point, unvisited_point]**alpha /
                    distance(
                        points[current_point],
                        points[unvisited_point]
                    )**beta
                )

            probabilities /= np.sum(probabilities)
            distances /= np.max(distances)
            pheromones = np.array([float(i) for i in pheromones])

            next_point = np.random.choice(unvisited, p=probabilities)
            path.append(next_point)
            path_length += distance(
                points[current_point],
                points[next_point]
            )
            visited[next_point] = True
            current_point = next_point
            plt.scatter(
                points[unvisited][:, 0],
                points[unvisited][:, 1],
                s=pheromones*500,
                c='orange', marker='o', alpha=0.3
            )
            plt.scatter(
                points[unvisited][:, 0],
                points[unvisited][:, 1],
                s=distances*500,
                c='blue', marker='o', alpha=0.3
            )
            print(f"{path=}")
            plt.plot(
                points[path][:, 0],
                points[path][:, 1],
                c='g', linestyle='-', linewidth=2, marker='o'
            )
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.show()

        paths.append(path)
        path_lengths.append(path_length)

        if path_length < best_path_length:
            best_path = path
            best_path_length = path_length

    pheromone *= evaporation_rate

    for path, path_length in zip(paths, path_lengths):
        for i in range(n_points-1):
            pheromone[path[i], path[i+1]] += Q/path_length
        pheromone[path[-1], path[0]] += Q/path_length

best_path_points = points[best_path]
plt.scatter(points[:, 0], points[:, 1], c='r', marker='o')
plt.plot(
    best_path_points[:, 0],
    best_path_points[:, 1],
    c='g', linestyle='-', linewidth=2, marker='o'
)

plt.show()
