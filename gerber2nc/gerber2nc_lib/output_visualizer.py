"""Render a quick visual CAM check before G-code is written to the machine.

PCB isolation routing is sensitive to missing layers, mirrored coordinates, and
wrong offsets. This preview window deliberately overlays:
- copper that should remain,
- white isolation paths that will be milled,
- the mechanical outline,
- and drill holes.

The goal is not polished UI. It is a fast operator sanity check before a real
tool touches the workpiece.
"""

class OutputVisualizer:
    # Show the PCB and tool paths on the screen for review. In CAM workflows a
    # simple preview often catches bad imports earlier than a broken board does.
    def __init__(self, base_name: str, bounds, visualisation_config: dict):
        self.base_name = base_name
        self.bounds = bounds
        self.visualisation_config = visualisation_config
        self.offset_geometry = False
        self.holes = []
        self.scale = 25

    def load_trace_geometries(self, traces) -> None:
        self.traces = traces

    def load_holes(self, holes: list) -> None:
        self.holes = holes

    def load_trace_mill_geometry(self, offsets) -> None:
        self.trace_mill_geometry = offsets

    def load_edge_cut_geometry(self, edgecuts) -> None:
        self.edgecuts = edgecuts

    def create_tkinter_visualization(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self.canvas = None
        root.title(
            self.base_name + ":   Edge cut paths in white.  Close this window to continue"
        )

        width_mm = self.bounds.x_max - self.bounds.x_min
        height_mm = self.bounds.y_max - self.bounds.y_min

        max_window_width = root.winfo_screenwidth() * 0.9
        if self.scale * width_mm > max_window_width:
            # Fit large boards onto the screen without changing the underlying
            # machine coordinates. Only the preview scale changes.
            self.scale = max_window_width / width_mm

        canvas_width = int(width_mm * self.scale)
        canvas_height = int(height_mm * self.scale)

        self.canvas = tk.Canvas(
            root,
            width=canvas_width,
            height=canvas_height,
            bg=(
                self.visualisation_config["background_with_edgecuts_rgb"]
                if self.edgecuts
                else self.visualisation_config["background_without_edgecuts_rgb"]
            ),
        )
        self.canvas.pack(padx=5, pady=5)

        if self.edgecuts:
            # Draw the edge-cuts loop first so it becomes the reference shape.
            # Y is flipped because screen coordinates grow downward while PCB
            # and CNC coordinates usually grow upward.
            coords = []
            for x, y in self.edgecuts:
                screen_x = x * self.scale
                screen_y = canvas_height - y * self.scale
                coords.extend([screen_x, screen_y])

            self.canvas.create_polygon(
                coords[:-2],
                fill=self.visualisation_config["edgecuts_fill_rgb"],
                outline=self.visualisation_config["edgecuts_outline_rgb"],
                width=2,
            )

        for trace in self.traces.traces:
            # Red stroked geometry shows the copper tracks that should remain.
            start_x, start_y = trace[0]
            end_x, end_y = trace[1]
            width_mm = trace[2]

            x1 = start_x * self.scale
            y1 = canvas_height - start_y * self.scale
            x2 = end_x * self.scale
            y2 = canvas_height - end_y * self.scale
            line_width = max(1, int(width_mm * self.scale))

            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=self.visualisation_config["trace_copper_rgb"],
                width=line_width,
                capstyle=tk.ROUND,
            )

        for region in self.traces.regions:
            # Filled regions are copper pours or polygon fills.
            coords = []
            for x, y in region:
                screen_x = x * self.scale
                screen_y = canvas_height - y * self.scale
                coords.extend([screen_x, screen_y])

            if coords:
                self.canvas.create_polygon(
                    coords,
                    fill=self.visualisation_config["region_copper_rgb"],
                    outline=self.visualisation_config["region_copper_rgb"],
                )

        for pad in self.traces.pads:
            # Pads are highlighted separately because in the Gerber data they
            # are flashed apertures, not line strokes.
            x_mm, y_mm = pad[0]
            aperture = pad[1]
            x = x_mm * self.scale
            y = canvas_height - y_mm * self.scale

            if aperture["type"] == "circle":
                radius = (aperture["diameter"] / 2) * self.scale
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=self.visualisation_config["pad_fill_rgb"],
                    outline=self.visualisation_config["pad_outline_rgb"],
                )
            elif aperture["type"] == "rectangle":
                width = aperture["width"] * self.scale / 2
                height = aperture["height"] * self.scale / 2
                self.canvas.create_rectangle(
                    x - width,
                    y - height,
                    x + width,
                    y + height,
                    fill=self.visualisation_config["pad_fill_rgb"],
                    outline=self.visualisation_config["pad_outline_rgb"],
                )

        for shape in self.trace_mill_geometry.geoms:
            # White lines are the actual milling centerlines. Comparing them to
            # the red copper makes it easy to see whether the clearance is sane.
            coords = []
            for x, y in shape.coords:
                screen_x = x * self.scale
                screen_y = canvas_height - y * self.scale
                coords.extend([screen_x, screen_y])

            self.canvas.create_line(
                coords,
                fill=self.visualisation_config["toolpath_rgb"],
                width=2,
            )

        for hole in self.holes:
            # Draw drill hits last so they remain visible above copper fills.
            x, y, diameter = hole
            screen_x = x * self.scale
            screen_y = canvas_height - y * self.scale
            radius = diameter / 2 * self.scale

            self.canvas.create_oval(
                screen_x - radius,
                screen_y - radius,
                screen_x + radius,
                screen_y + radius,
                fill=self.visualisation_config["hole_fill_rgb"],
                outline=self.visualisation_config["hole_outline_rgb"],
                width=1,
            )

        root.mainloop()
