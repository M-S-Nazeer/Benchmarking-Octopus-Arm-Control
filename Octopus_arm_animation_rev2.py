import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
from scipy.interpolate import interp1d
import matplotlib.animation as animation


def generate_tentacle_surface(base_pos, tip_pos, radius_profile_func, control_points=None,
                              num_path_points=50, num_circle_points=20):
    """Generates the X, Y, Z surface coordinates for a tapered tentacle."""
    path_t = np.linspace(0, 1, num_path_points)
    all_control_points = [base_pos] + control_points + [tip_pos]
    n = len(all_control_points) - 1
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**i) * ((1 - t)**(n - i))
    path = np.zeros((num_path_points, 3))
    for i in range(num_path_points):
        t = path_t[i]
        path[i] = sum(bernstein_poly(j, n, t) * np.array(all_control_points[j]) for j in range(n + 1))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    radii = radius_profile_func(arc_lengths)
    X, Y, Z = (np.zeros((num_path_points, num_circle_points)) for _ in range(3))
    theta = np.linspace(0, 2 * np.pi, num_circle_points)
    u_circ, v_circ = np.cos(theta), np.sin(theta)
    for i in range(num_path_points):
        if radii[i] <= 1e-6: continue
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


# def octopus_update(frame_num, ax, xt_trajectory, base_pos, tip_pos, radius_profile_func, trial_ref_positions=None):
#     """
#     Function called for each frame of the animation.
#     """
#     # Clear the previous frame
#     ax.cla()
#     if trial_ref_positions is not None:
#         ax.scatter(trial_ref_positions[:, 0], trial_ref_positions[:, 1], trial_ref_positions[:, 2],
#                    c='k', s=100, marker='*')
#
#     # Get the current position for the point of interest (x_t)
#     current_xt = xt_trajectory[frame_num]
#     current_tip_pos = tip_pos[frame_num]
#
#     # --- Dynamically calculate control points based on current_xt ---
#     # This makes the arm bend "naturally" as x_t moves.
#     # CP1 is on the line from base to x_t, CP2 is on the line from tip to x_t.
#     pull_factor = 0.8  # Determines how "stiff" the curve is. <1 bends less, >1 bends more.
#     control_point_1 = base_pos * (1 - pull_factor) + current_xt * pull_factor
#     control_point_2 = current_tip_pos * (1 - pull_factor) + current_xt * pull_factor
#     control_points = [control_point_1, control_point_2]
#
#     # Generate the new tentacle surface data
#     X, Y, Z = generate_tentacle_surface(base_pos, current_tip_pos, radius_profile_func, control_points)
#
#     # Plot the new surface
#     ax.plot_surface(X, Y, Z, color='mediumseagreen', alpha=0.6, rstride=1, cstride=1, linewidth=0.1, edgecolors='k')
#
#     # Plot the key points for this frame
#     ax.scatter(*base_pos, color='blue', s=150, label='Base Position', depthshade=False, zorder=10)
#     ax.scatter(*current_tip_pos, color='purple', s=20, label='Passive Tip', depthshade=False, zorder=10)
#     ax.scatter(*current_xt, color='red', s=20, label='Active Tip', depthshade=False, zorder=10)
#
#     # Plot the trajectory of x_t so far
#     ax.plot(xt_trajectory[:frame_num + 1, 0], xt_trajectory[:frame_num + 1, 1], xt_trajectory[:frame_num + 1, 2],
#             color='red', linestyle=':', alpha=0.5)
#
#     # Re-apply plot settings for each frame to keep them consistent
#     ax.set_xlabel('X-axis (mm)')
#     ax.set_ylabel('Y-axis (mm)')
#     ax.set_zlabel('Z-axis (mm)')
#     ax.set_title(f'Octopus Tentacle Animation (Frame {frame_num + 1})')
#     # ax.legend(loc='upper left')
#
#     # Maintain consistent viewing limits
#     ax.set_xlim(-200, 200)
#     ax.set_ylim(-200, 200)
#     ax.set_zlim(-700, 50)
#     ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio equal


# if __name__ == "__main__":
#     # --- 1. Define Static Information ---
#     base_pos = np.array([63., -5., -528.33])
#
#     length_to_xt = 338.0
#     total_length = 510.0
#
#     dia_base = 50.0
#     dia_mid = 29.5
#     dia_xt = 16.2
#     dia_tip = 1.0
#
#     # --- 2. Create the Radius Profile Function (once) ---
#     radius_data_points = [
#         (0.0, dia_base / 2.0),
#         (total_length / 2.0, dia_mid / 2.0),
#         (length_to_xt, dia_xt / 2.0),
#         (total_length, dia_tip / 2.0)
#     ]
#     radius_data_points.sort(key=lambda item: item[0])
#     known_lengths = [p[0] for p in radius_data_points]
#     known_radii = [p[1] for p in radius_data_points]
#     radius_profile_func = interp1d(known_lengths, known_radii, kind='linear', fill_value="extrapolate")
#
#     # --- 3. Create the Animation Trajectory for the point of interest (x_t) ---
#     # **REPLACE THIS with your 1500x3 array of x_t positions**
#     # For this example, we'll create a circular path for x_t.
#     num_frames = 200  # Number of animation frames
#     # Center of the circle is the initial x_t position
#     center_xt = np.array([63.91, -5.03, -190.33])
#     radius_xt_path = 50.0
#     angle = np.linspace(0, 2 * np.pi, num_frames)
#
#     xt_trajectory = np.zeros((num_frames, 3))
#     xt_trajectory[:, 0] = center_xt[0] + radius_xt_path * np.cos(angle)  # X moves
#     xt_trajectory[:, 1] = center_xt[1] + radius_xt_path * np.sin(angle)  # Y moves
#     xt_trajectory[:, 2] = center_xt[2]  # Z stays constant
#
#     tip_pos = np.zeros((num_frames, 3))
#     tip_pos[:, 0] = xt_trajectory[:, 0] - 3.0
#     tip_pos[:, 1] = xt_trajectory[:, 0] - 2.0
#     tip_pos[:, 2] = xt_trajectory[:, 0] - 172.
#
#     # --- 4. Set up the Plot and Animation ---
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     # Create the animation object.
#     # `fargs` passes the necessary arguments to the `update` function.
#     # `interval` is the delay between frames in milliseconds.
#     # `blit=False` is necessary for 3D animations and for plots where elements are cleared.
#     ani = animation.FuncAnimation(fig, update, frames=num_frames,
#                                   fargs=(ax, xt_trajectory, base_pos, tip_pos, radius_profile_func),
#                                   interval=50, blit=False, repeat=True)
#
#     # To save the animation, you need a writer like ffmpeg or imagemagick
#     # Make sure ffmpeg is installed on your system: sudo apt-get install ffmpeg
#     # print("Saving animation... this may take a moment.")
#     # ani.save('octopus_animation.mp4', writer='ffmpeg', dpi=150)
#     # print("Animation saved to octopus_animation.mp4")
#
#     plt.show()
