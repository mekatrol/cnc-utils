from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D support is loaded
import numpy as np
import matplotlib.pyplot as plt
from geometry.GeometryInt import GeometryInt


class Viewer3d:
    @staticmethod
    def _set_axes_equal_3d(ax):
        """Make 3D axes have equal scale so geometry isn't distorted."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def visualize(geom: GeometryInt, show_axes: bool = True):
        """Render integer polylines at z=0 in a 3D view (rotatable)."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each polyline. Convert back to floats (unscale) for display.
        inv = 1.0 / float(geom.scale if geom.scale != 0 else 1)
        for pl in geom.polylines:
            xs = [p.x * inv for p in pl.pts]
            ys = [p.y * inv for p in pl.pts]
            zs = [0.0 for _ in pl.pts]
            ax.plot(xs, ys, zs, linewidth=1.0)

        if show_axes:
            # Add simple XY axes in the plane
            minx, miny, maxx, maxy = geom.bounds()
            invx = [minx * inv, maxx * inv]
            ax.plot(invx, [0, 0], [0, 0])
            invy = [miny * inv, maxy * inv]
            ax.plot([0, 0], invy, [0, 0])

        ax.set_xlabel("X (SVG units)")
        ax.set_ylabel("Y (SVG units)")
        ax.set_zlabel("Z")
        ax.view_init(elev=30, azim=-60)
        Viewer3d._set_axes_equal_3d(ax)
        ax.grid(True)

        def on_key(event):
            if event.key == 'r':
                ax.view_init(elev=30, azim=-60)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()
