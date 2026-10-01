from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser(description="Mamba-YOLO Inference Script")
    parser.add_argument('--weights', type=str, default=ROOT + 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=ROOT + 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT + 'runs/predict', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Initialize YOLO model
    model = YOLO(opt.weights)

    # Run prediction
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        device=opt.device,
        show=opt.view_img,
        save=not opt.nosave,
        save_txt=opt.save_txt,
        save_conf=opt.save_conf,
        save_crop=opt.save_crop,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        augment=opt.augment,
        visualize=opt.visualize,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        line_width=opt.line_thickness,
        show_labels=not opt.hide_labels,
        show_conf=not opt.hide_conf,
        half=opt.half,
        dnn=opt.dnn,
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)