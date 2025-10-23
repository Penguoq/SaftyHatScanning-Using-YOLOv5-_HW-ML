"""
Simple inference script for HatScan weights.
Usage (Python):
    python scripts/predict.py --weights weights/best.pt --source assets/samples --imgsz 640
If running from a fresh environment, first install YOLOv5:
    pip install git+https://github.com/ultralytics/yolov5.git
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='assets/samples', help='file/dir/URL/glob')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    args = parser.parse_args()

    try:
        from yolov5 import detect  # YOLOv5's builtin detect.py entrypoint
    except Exception as e:
        print("YOLOv5 not found. Install with: pip install git+https://github.com/ultralytics/yolov5.git")
        sys.exit(1)

    # Build CLI args for detect.run
    detect.run(weights=args.weights, source=args.source, imgsz=args.imgsz, conf_thres=args.conf)

if __name__ == '__main__':
    main()