import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
from scipy.interpolate import interp1d
import matplotlib.animation as animation


# --- Core Function to Generate Piecewise Bézier Path ---
def generate_piecewise_path(base_pos, mid_pos, tip_pos, cp1, cp2, num_points_per_segment=25):
    """
    Generates a smooth path from two connected quadratic Bézier curves.
    Curve 1: base -> cp1 -> mid_pos
    Curve 2: mid_pos -> cp2 -> tip_pos
    """
    path_segments = []
    segment1_points = [base_pos, cp1, mid_pos]
    segment2_points = [mid_pos, cp2, tip_pos]
    all_segments = [segment1_points, segment2_points]

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    for segment_ctrl_pts in all_segments:
        n = 2  # Degree of a quadratic Bézier curve
        path_t = np.linspace(0, 1, num_points_per_segment)
        segment_path = np.zeros((num_points_per_segment, 3))
        for i, t in enumerate(path_t):
            segment_path[i] = sum(bernstein_poly(j, n, t) * np.array(segment_ctrl_pts[j]) for j in range(n + 1))

        if not path_segments:
            path_segments.append(segment_path)
        else:
            path_segments.append(segment_path[1:])  # Avoid duplicating the join point

    return np.vstack(path_segments)


# --- Core Function to Generate Tentacle Surface (Unchanged) ---
def generate_tentacle_surface(path, radius_profile_func, num_circle_points=20):
    """Generates the X, Y, Z surface coordinates for a given path and radius profile."""
    num_path_points = path.shape[0]
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    radii = radius_profile_func(arc_lengths)

    X, Y, Z = (np.zeros((num_path_points, num_circle_points)) for _ in range(3))
    theta = np.linspace(0, 2 * np.pi, num_circle_points)
    u_circ, v_circ = np.cos(theta), np.sin(theta)

    for i in range(num_path_points):
        if radii[i] <= 1e-6: continue
        tangent = path[i + 1] - path[i] if i < num_path_points - 1 else path[i] - path[i - 1]
        tangent /= np.linalg.norm(tangent)
        not_parallel = np.array([0, 0, 1]) if (np.abs(tangent[0]) > 0.1 or np.abs(tangent[1]) > 0.1) else np.array(
            [0, 1, 0])
        normal1 = np.cross(tangent, not_parallel)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.cross(tangent, normal1)
        normal2 /= np.linalg.norm(normal2)
        for j in range(num_circle_points):
            circle_point = path[i] + radii[i] * (u_circ[j] * normal1 + v_circ[j] * normal2)
            X[i, j], Y[i, j], Z[i, j] = circle_point

    return X, Y, Z


# --- Animation Update Function (Using your control point logic) ---
def update(frame_num, ax, xt_trajectory, base_pos, tip_pos, radius_profile_func, pull_factor):
    """
    Function called for each frame of the animation.
    Calculates control points to guarantee the path passes through x_t.
    """
    ax.cla()  # Clear the previous frame

    # Get the current position for the point of interest (x_t)
    current_xt = xt_trajectory[frame_num]
    current_tip_pos = tip_pos[frame_num]
    # --- Use your specified logic to calculate dynamic control points ---
    # This creates a C-shaped curve where the control points lie on the
    # lines connecting the endpoints to the midpoint (x_t).
    control_point_1 = base_pos * (1 - pull_factor) + current_xt * pull_factor
    control_point_2 = current_tip_pos * (1 - pull_factor) + current_xt * pull_factor

    # Generate the path from two connected curves: base->CP1->x_t and x_t->CP2->tip
    path = generate_piecewise_path(base_pos, current_xt, current_tip_pos, control_point_1, control_point_2)

    # Generate the surface mesh based on the new path
    X, Y, Z = generate_tentacle_surface(path, radius_profile_func)

    # Plot the new surface
    ax.plot_surface(X, Y, Z, color='mediumseagreen', alpha=0.6, rstride=1, cstride=1, linewidth=0.1, edgecolors='k')

    # Plot the key points and control cage for this frame
    ax.scatter(*base_pos, color='blue', s=150, label='Base', depthshade=False, zorder=10)
    ax.scatter(*current_tip_pos, color='purple', s=150, label='Tip', depthshade=False, zorder=10)
    ax.scatter(*current_xt, color='red', s=150, label='Point of Interest (x_t)', marker='X', depthshade=False,
               zorder=10)
    ax.scatter(*control_point_1, color='orange', s=80, label='Control Points', marker='^', depthshade=False, zorder=10)
    ax.scatter(*control_point_2, color='orange', s=80, marker='^', depthshade=False, zorder=10)
    ax.plot(*zip(base_pos, control_point_1, current_xt, control_point_2, current_tip_pos), color='gray', linestyle=':',
            alpha=0.7)

    # Plot the trajectory of x_t so far
    ax.plot(xt_trajectory[:frame_num + 1, 0], xt_trajectory[:frame_num + 1, 1], xt_trajectory[:frame_num + 1, 2],
            color='red', linestyle='--', alpha=0.5)

    # Re-apply plot settings
    ax.set_xlabel('X-axis (mm)')
    ax.set_ylabel('Y-axis (mm)')
    ax.set_zlabel('Z-axis (mm)')
    ax.set_title(f'Octopus Tentacle Animation (Frame {frame_num + 1})')
    ax.legend(loc='upper left')
    ax.set_xlim(-100, 200)
    ax.set_ylim(-200, 150)
    ax.set_zlim(-700, 50)
    ax.set_box_aspect([1, 1, 1])


if __name__ == "__main__":
    # --- 1. Define Static Information ---
    base_pos = np.array([60., -2., -528.33])

    # Define the pull factor for the control points. This is now the main "shape" parameter.
    # 0.5 = CPs are halfway between endpoints and x_t.
    # > 0.5 = curve is "tighter" towards x_t.
    # < 0.5 = curve is "looser" and wider.
    PULL_FACTOR = 0.8

    length_to_xt = 338.0
    total_length = 510.0
    dia_base = 50.0
    dia_mid = 29.5
    dia_xt = 16.2
    dia_tip = 1.0

    # --- 2. Create the Radius Profile Function (once) ---
    radius_data_points = [
        (0.0, dia_base / 2.0),
        (total_length / 2.0, dia_mid / 2.0),
        # (length_to_xt, dia_xt / 2.0),
        (total_length, dia_tip / 2.0)
    ]
    radius_data_points.sort(key=lambda item: item[0])
    known_lengths = [p[0] for p in radius_data_points]
    known_radii = [p[1] for p in radius_data_points]
    radius_profile_func = interp1d(known_lengths, known_radii, kind='linear', fill_value="extrapolate")

    # --- 3. Create the Animation Trajectory for x_t ---
    # **REPLACE THIS with your array of x_t positions**
    num_frames = 200
    center_xt = np.array([63.91, -5.03, -190.33])
    radius_xt_path = 70.0
    angle = np.linspace(0, 2 * np.pi, num_frames)
    xt_trajectory = np.zeros((num_frames, 3))
    xt_trajectory[:, 0] = center_xt[0] + radius_xt_path * np.cos(angle)
    xt_trajectory[:, 1] = center_xt[1] + radius_xt_path * np.sin(angle)
    xt_trajectory[:, 2] = center_xt[2]

    tip_pos = np.zeros((num_frames, 3))
    tip_pos[:, 0] = xt_trajectory[:, 0] - 10.0
    tip_pos[:, 1] = xt_trajectory[:, 1] - 10.0
    tip_pos[:, 2] = xt_trajectory[:, 2] + 172.0

    print("Tip position: ", tip_pos)

    # --- 4. Set up the Plot and Animation ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # `fargs` now includes the static pull_factor
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  fargs=(ax, xt_trajectory, base_pos, tip_pos, radius_profile_func, PULL_FACTOR),
                                  interval=50, blit=False, repeat=True)

    # To save the animation:
    # ani.save('octopus_pull_factor_animation.mp4', writer='ffmpeg', dpi=150)
    plt.show()
