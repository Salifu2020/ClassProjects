# ----------------------------------- Fuseini Salifu ----------------------------------------

# ---------------------------------------------
#
#
# -- CSC315 / DSC615 Project 3
#
#
# ------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import random
import time

pi = 3.14159265358979323846264


# -----------------------------------
# GenPoints   generates a set of random 2D points
#             containing 5 planted circles
#             within a set of uniformly distributed
#             outliers
#
#     Input:   None
#     Output:  xy    [700 2]  numpy array of 700 2d points
#
# -----------------------------------
def GenPoints():
    n_circle = 5
    point_per_circle = 100
    noise_points = 200
    total_points = n_circle * point_per_circle + noise_points
    xy = np.random.uniform((-10.0, -10.0), (10.0, 10.0), (total_points, 2))

    for c in range(n_circle):
        # Randomly generate a circle
        xpos = random.uniform(-8.0, 8.0)
        ypos = random.uniform(-8.0, 8.0)
        radius = random.uniform(0.5, 2.0)
        spread = random.uniform(0.05, 0.2)

        thetas = np.random.uniform(0.0, 2 * pi, (point_per_circle,))
        rhos = radius + np.random.normal(radius, spread, (point_per_circle,))

        circle_xy = np.zeros((point_per_circle, 2), dtype='float64')
        circle_xy[:, 0] = xpos + rhos * np.cos(thetas)
        circle_xy[:, 1] = ypos + rhos * np.sin(thetas)

        # Fill the portion of the numpy array
        sidx = c * point_per_circle
        eidx = (c + 1) * point_per_circle
        xy[sidx:eidx] = circle_xy

    for i in range(total_points):
        j = random.randint(i, total_points - 1)
        temp = np.array(xy[i])
        xy[i] = xy[j]
        xy[j] = temp

    return xy


# Function to calculate the circumcircle from 3 points
def find_circle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Check for collinearity to avoid division by zero
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-10:  # Points are collinear
        return None, None, None

    # Calculate center (x0, y0)
    x0 = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    y0 = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d

    # Calculate radius
    radius = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    return x0, y0, radius


# RANSAC algorithm to detect one circle
def ransac_circle(points, n_iterations=1000, distance_threshold=0.2, min_inliers=30):
    best_inliers = 0
    best_circle = None
    best_inlier_mask = None

    n_points = len(points)
    if n_points < 3:
        return None, None

    for _ in range(n_iterations):
        # Randomly select 3 points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Find the circle through these 3 points
        x0, y0, radius = find_circle(p1, p2, p3)
        if x0 is None:  # Skip if points are collinear
            continue

        # Vectorized distance calculation
        distances = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
        inlier_mask = np.abs(distances - radius) < distance_threshold
        n_inliers = np.sum(inlier_mask)

        if n_inliers > best_inliers and n_inliers >= min_inliers:
            best_inliers = n_inliers
            best_circle = (x0, y0, radius)
            best_inlier_mask = inlier_mask

    return best_circle, best_inlier_mask


# Main simulation
n_runs = 5
n_circles = 5
colors = ['red', 'blue', 'green', 'orange', 'yellow']  # Colors for each circle

for run in range(n_runs):
    start_time = time.time()

    # Generate points
    xy = GenPoints()
    points = xy.copy()  # Working copy of points

    # Prepare the plot
    plt.figure(figsize=(8.0, 8.0))
    plt.scatter(xy[:, 0], xy[:, 1], c='black', s=10)
    plt.title(f'Run {run + 1}')

    # Detect circles one by one
    for circle_idx in range(n_circles):
        # Run RANSAC to detect one circle
        circle, inlier_mask = ransac_circle(points, n_iterations=1000, distance_threshold=0.2, min_inliers=30)

        if circle is not None:
            x0, y0, radius = circle

            # Plot the detected circle
            theta = np.linspace(0, 2 * pi, 100)
            x = x0 + radius * np.cos(theta)
            y = y0 + radius * np.sin(theta)
            plt.plot(x, y, c=colors[circle_idx], linestyle='--', linewidth=2)

            # Plot the inlier threshold boundaries
            x_inner = x0 + (radius - 0.2) * np.cos(theta)
            y_inner = y0 + (radius - 0.2) * np.sin(theta)
            x_outer = x0 + (radius + 0.2) * np.cos(theta)
            y_outer = y0 + (radius + 0.2) * np.sin(theta)
            plt.plot(x_inner, y_inner, c=colors[circle_idx], linestyle=':', linewidth=1)
            plt.plot(x_outer, y_outer, c=colors[circle_idx], linestyle=':', linewidth=1)

            # Remove inlier points to avoid detecting the same circle again
            points = points[~inlier_mask]

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

    # Check timing
    elapsed_time = time.time() - start_time
    print(f'Run {run + 1} took {elapsed_time:.2f} seconds')

print('Done!')
print('I hope you enjoyed my implementation 🙏')
