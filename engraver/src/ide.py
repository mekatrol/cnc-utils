import argparse

from views import AppView


def main():
    parser = argparse.ArgumentParser(description="Polygon Engraver")
    parser.add_argument("--input", dest="input_path", help="Input SVG/JSON file path")
    parser.add_argument("--scale", type=int, default=10000, help="SVG import scale")
    parser.add_argument("--tol", type=float, default=0.25, help="SVG import tolerance")
    parser.add_argument("--export-json", dest="export_json", help="Export JSON path")
    args, _unknown = parser.parse_known_args()

    app_view = AppView()
    if args.input_path:
        app_view.queue_startup_load(
            args.input_path,
            scale=args.scale,
            tol=args.tol,
            export_json=args.export_json,
        )
    app_view.mainloop()


if __name__ == "__main__":
    main()
