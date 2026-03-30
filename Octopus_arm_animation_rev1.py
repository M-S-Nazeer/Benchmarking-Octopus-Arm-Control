import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb  # For Bézier curve calculation
import time
from scipy.interpolate import interp1d


def plot_tapered_tentacle(ax, base_pos, tip_pos, base_radius, tip_radius,
                          num_path_points=50, num_circle_points=20,
                          control_points=None, color='cornflowerblue'):
    """
    Plots a 3D tapered shape representing a tentacle on a given Matplotlib 3D axis.

    Args:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The 3D subplot to plot on.
        base_pos (np.ndarray): A 1x3 numpy array for the [x, y, z] base position.
        tip_pos (np.ndarray): A 1x3 numpy array for the [x, y, z] tip position.
        base_radius (float): The radius of the tentacle at its base.
        tip_radius (float): The radius of the tentacle at its tip.
        num_path_points (int): Number of segments along the tentacle's length.
        num_circle_points (int): Number of points to define the circumference.
        control_points (list of np.ndarray, optional): A list of 1x3 numpy arrays
            representing Bézier control points for a curved path. If None, a
            straight line is used.
        color (str): The color of the tentacle surface.
    """
    # --- 1. Define the Path (Spine of the tentacle) ---
    path_points = np.linspace(0, 1, num_path_points)

    if control_points is None:
        # Straight line path
        path = np.array([base_pos + t * (tip_pos - base_pos) for t in path_points])
    else:
        # Curved path using a Bézier curve
        # Combine all control points: base, user-provided, tip
        all_control_points = [base_pos] + control_points + [tip_pos]
        n = len(all_control_points) - 1

        # Calculate Bernstein basis polynomials
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        path = np.zeros((num_path_points, 3))
        for i in range(num_path_points):
            t = path_points[i]
            path[i] = sum(bernstein_poly(j, n, t) * np.array(all_control_points[j]) for j in range(n + 1))

    # --- 2. Define the Radius Profile ---
    # Linearly decreasing radius from base to tip
    radii = np.linspace(base_radius, tip_radius, num_path_points)

    # --- 3. Generate the 3D Surface ---
    # Create arrays to hold the surface coordinates
    X = np.zeros((num_path_points, num_circle_points))
    Y = np.zeros((num_path_points, num_circle_points))
    Z = np.zeros((num_path_points, num_circle_points))

    # Generate points for a unit circle
    theta = np.linspace(0, 2 * np.pi, num_circle_points)
    u = np.cos(theta)
    v = np.sin(theta)

    for i in range(num_path_points):
        # For each point on the path, we need to define a plane perpendicular to the path
        if i < num_path_points - 1:
            tangent = path[i + 1] - path[i]
        else:  # For the last point, use the previous tangent
            tangent = path[i] - path[i - 1]

        tangent = tangent / np.linalg.norm(tangent)

        # Find a vector not parallel to the tangent (an arbitrary 'up' vector)
        not_parallel = np.array([0, 0, 1]) if (np.abs(tangent[0]) > 0.1 or np.abs(tangent[1]) > 0.1) else np.array(
            [0, 1, 0])

        # Create two perpendicular vectors to form the plane of the circle
        normal1 = np.cross(tangent, not_parallel)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.cross(tangent, normal1)
        normal2 /= np.linalg.norm(normal2)

        # Generate the circle points in 3D space
        for j in range(num_circle_points):
            circle_point = path[i] + radii[i] * (u[j] * normal1 + v[j] * normal2)
            X[i, j] = circle_point[0]
            Y[i, j] = circle_point[1]
            Z[i, j] = circle_point[2]

    # --- 4. Plot the Surface ---
    ax.plot_surface(X, Y, Z, color=color, alpha=0.7, rstride=1, cstride=1)


def generate_tentacle_surface(base_pos, tip_pos, base_radius, tip_radius,
                              num_path_points=50, num_circle_points=20,
                              control_points=None):
    """
    Generates the X, Y, Z surface coordinates for a tapered tentacle.
    This function contains the core math and is separated for clarity.
    """
    # 1. Define Path using Bézier curve
    path_t = np.linspace(0, 1, num_path_points)
    all_control_points = [base_pos] + control_points + [tip_pos]
    n = len(all_control_points) - 1

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**i) * ((1 - t)**(n - i))

    path = np.zeros((num_path_points, 3))
    for i in range(num_path_points):
        t = path_t[i]
        path[i] = sum(bernstein_poly(j, n, t) * np.array(all_control_points[j]) for j in range(n + 1))

    # 2. Define Radius Profile
    radii = np.linspace(base_radius, tip_radius, num_path_points)

    # 3. Generate 3D Surface
    X, Y, Z = (np.zeros((num_path_points, num_circle_points)) for _ in range(3))
    theta = np.linspace(0, 2 * np.pi, num_circle_points)
    u_circ, v_circ = np.cos(theta), np.sin(theta)

    for i in range(num_path_points):
        tangent = path[i+1] - path[i] if i < num_path_points - 1 else path[i] - path[i-1]
        tangent /= np.linalg.norm(tangent)

        not_parallel = np.array([0, 0, 1]) if (np.abs(tangent[0]) > 0.1 or np.abs(tangent[1]) > 0.1) else np.array([0, 1, 0])
        normal1 = np.cross(tangent, not_parallel)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.cross(tangent, normal1)
        normal2 /= np.linalg.norm(normal2)

        for j in range(num_circle_points):
            circle_point = path[i] + radii[i] * (u_circ[j] * normal1 + v_circ[j] * normal2)
            X[i, j], Y[i, j], Z[i, j] = circle_point

    return X, Y, Z


def update_plot(ax, base_pos, tip_pos, control_points, base_radius, tip_radius):
    """
    Clears the axes and redraws the entire tentacle scene.
    """
    # Clear the previous plot to prevent drawing over it
    ax.cla()

    # Generate the new tentacle surface
    X, Y, Z = generate_tentacle_surface(
        base_pos, tip_pos, base_radius, tip_radius,
        control_points=control_points
    )
    # Plot the new surface
    ax.plot_surface(X, Y, Z, color='deepskyblue', alpha=0.6, rstride=1, cstride=1)

    # Re-plot the control points, base, and tip
    all_points = [base_pos] + [tip_pos] + control_points
    point_colors = ['red', 'darkred'] + ['green'] * len(control_points)
    ax.scatter(
        [p[0] for p in all_points],
        [p[1] for p in all_points],
        [p[2] for p in all_points],
        s=100, c=point_colors, depthshade=False
    )

    # Re-apply plot labels and limits
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Manually Controlled Tentacle')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    # Re-set consistent axes limits for a stable view
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 15)

    # Refresh the canvas
    plt.draw()
    plt.pause(0.01)  # A brief pause to allow the plot to update


def get_user_input(prompt):
    """
    Robustly gets a list of three floats from the user.
    """
    while True:
        try:
            user_str = input(prompt)
            # Check for quit command
            if user_str.lower() in ['q', 'quit', 'exit']:
                return None
            # Split the string by comma or space and convert to float
            coords = [float(x) for x in user_str.replace(',', ' ').split()]
            if len(coords) != 3:
                print("Error: Please enter exactly three numbers separated by spaces or commas.")
                continue
            return np.array(coords)
        except ValueError:
            print("Error: Invalid input. Please enter numbers only (e.g., '5 2 4').")

if __name__ == '__main__':

    # # --- Define the 3D plot ---
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # --- Tentacle 1: Straight ---
    # base_pos_1 = np.array([0, 0, 0])
    # tip_pos_1 = np.array([5, 5, 8])
    # plot_tapered_tentacle(ax, base_pos=base_pos_1, tip_pos=tip_pos_1,
    #                       base_radius=1.0, tip_radius=0.1, color='lightcoral')
    # # Mark base and tip for clarity
    # ax.scatter(*base_pos_1, color='red', s=100, label='Base 1')
    # ax.scatter(*tip_pos_1, color='darkred', s=50, label='Tip 1')
    #
    # # --- Tentacle 2: Curved ---
    # base_pos_2 = np.array([-8, 0, 0])
    # tip_pos_2 = np.array([0, -8, 6])
    # # Define control points to create a curve
    # # The curve will be pulled towards these points
    # control_points_2 = [
    #     np.array([-8, -6, 4]),  # Pulls the tentacle this way
    #     np.array([-2, -8, 2])  # Then pulls it this way
    # ]
    # plot_tapered_tentacle(ax, base_pos=base_pos_2, tip_pos=tip_pos_2,
    #                       control_points=control_points_2,
    #                       base_radius=1.2, tip_radius=0.15, color='deepskyblue')
    # # Mark base and tip for clarity
    # ax.scatter(*base_pos_2, color='blue', s=100, label='Base 2')
    # ax.scatter(*tip_pos_2, color='darkblue', s=50, label='Tip 2')
    #
    # # --- Plot Customization ---
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_title('Tapered Tentacle Shapes')
    # ax.legend()
    # # Set aspect ratio to be equal for a more realistic view
    # ax.set_box_aspect([1, 1, 1])  # For matplotlib >= 3.3
    # # For older versions, you might need a more complex aspect ratio solution
    # # x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    # # x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    # # y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    # # z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    # # plot_radius = 0.5*max([x_range, y_range, z_range])
    # # ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    # # ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    # # ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    #
    # plt.show()

    # Enable interactive mode in Matplotlib
    plt.ion()

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Initial positions for the tentacle
    base_pos = np.array([0, 0, 0])
    tip_pos = np.array([2, 8, 10])
    control_points = [
        np.array([5, 2, 4]),
        np.array([-2, 6, 8])
    ]
    base_radius = 1.0
    tip_radius = 0.1

    # Initial plot
    update_plot(ax, base_pos, tip_pos, control_points, base_radius, tip_radius)

    # --- Manual Control Loop ---
    while True:
        print("\n--- Update Tentacle Control Points ---")
        print("Current Base: [Not Editable]")
        print("Current Tip:  [Not Editable]")
        for i, p in enumerate(control_points):
            print(f"Current CP {i + 1}: {np.round(p, 2)}")

        print("\nEnter 'q' or 'quit' to exit.")

        # Ask user which point to modify
        try:
            choice_str = input(f"Which Control Point to modify (1 to {len(control_points)}): ")
            if choice_str.lower() in ['q', 'quit', 'exit']:
                break

            choice_idx = int(choice_str) - 1  # Convert to 0-based index
            if not (0 <= choice_idx < len(control_points)):
                print(f"Invalid choice. Please enter a number between 1 and {len(control_points)}.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        # Get new coordinates for the chosen point
        new_coords = get_user_input(f"Enter new X Y Z for CP {choice_idx + 1} (e.g., '5.5 3 -2'): ")

        if new_coords is None:  # User chose to quit
            break

        # Update the control point's position
        control_points[choice_idx] = new_coords

        # Redraw the entire scene with the new point position
        update_plot(ax, base_pos, tip_pos, control_points, base_radius, tip_radius)

    # --- Cleanup ---
    print("\nExiting. Close the plot window to finish.")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot visible until the user closes it




