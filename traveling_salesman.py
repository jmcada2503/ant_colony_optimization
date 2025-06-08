import numpy as np
import matplotlib.pyplot as plt


def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def ant_colony_optimization(
    points,
    n_ants,
    n_iterations,
    alpha,
    beta,
    evaporation_rate,
    Q
):
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

                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = (
                        pheromone[current_point, unvisited_point]**alpha /
                        distance(
                            points[current_point],
                            points[unvisited_point]
                        )**beta
                    )

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(
                    points[current_point],
                    points[next_point]
                )
                visited[next_point] = True
                current_point = next_point

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

    plt.scatter(points[:, 0], points[:, 1], c='r', marker='o')

    for i in range(n_points-1):
        plt.plot(
            [points[best_path[i], 0], points[best_path[i+1], 0]],
            [points[best_path[i], 1], points[best_path[i+1], 1]],
            c='g', linestyle='-', linewidth=2, marker='o'
        )

    plt.show()


points = np.random.rand(10, 2)  # Generate 10 random 2D points
ant_colony_optimization(
    points,
    n_ants=10,
    n_iterations=1,
    alpha=1,
    beta=1,
    evaporation_rate=0.5,
    Q=1
)
