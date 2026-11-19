from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Run live object detection on a camera feed.")
    parser.add_argument("weights", type=Path, metavar="MODEL.pt", help="YOLO detection weights.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence between 0 and 1.")
    args = parser.parse_args()

    if args.weights.suffix.lower() != ".pt" or not args.weights.is_file():
        parser.error(f"Provide an existing .pt file: {args.weights}")
    if args.camera < 0:
        parser.error("--camera must be a nonnegative index.")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")

    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors

    model = YOLO(str(args.weights.resolve()))
    if model.task != "detect":
        parser.error("The .pt file must contain a YOLO detection model.")

    camera = cv2.VideoCapture(args.camera)
    window = "FireGuardML | Press Q or Esc to exit"
    try:
        if not camera.isOpened():
            parser.exit(1, f"Could not open camera {args.camera}.\n")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = camera.read()
            if not ok:
                parser.exit(1, f"Could not read a frame from camera {args.camera}.\n")

            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            annotator = Annotator(frame, example=str(result.names))
            for box in result.boxes.cpu():
                class_id = int(box.cls.item())
                label = f"{result.names[class_id]} {box.conf.item():.0%}"
                annotator.box_label(box.xyxy[0].tolist(), label, color=colors(class_id, bgr=True))

            cv2.imshow(window, annotator.result())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
