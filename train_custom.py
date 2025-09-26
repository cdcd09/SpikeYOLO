#!/usr/bin/env python3
import argparse
import os
from ultralytics import YOLO
'''


python /workspace/train_custom.py \
    --data /data/obstacles-5_yolov5/data.yaml \
    --model /workspace/ckpt/best.pt \
    --epochs 200 --batch 16 --imgsz 640 \
    --device 0,1,2,3 \
    --project obstacles


'''

def parse_args():
    p = argparse.ArgumentParser(description="Train custom Ultralytics fork with optional W&B logging")
    p.add_argument("--data", type=str, default="/data/obstacles-5_yolov5/data.yaml")
    p.add_argument("--model", type=str, default="/workspace/ckpt/best.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--project", type=str, default="obstacles")
    p.add_argument("--name", type=str, default="obstacles-yolov8-custom")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def maybe_enable_wandb(enable_flag: bool):
    if enable_flag:
        os.environ.pop("WANDB_DISABLED", None)
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            try:
                import wandb
                wandb.login(key=api_key)
            except Exception:
                pass
    else:
        os.environ["WANDB_DISABLED"] = "true"


def main():
    args = parse_args()
    maybe_enable_wandb(args.wandb)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        resume=args.resume,
        verbose=True,
    )

    # Validation metrics
    val_res = model.val(data=args.data, imgsz=args.imgsz, device=args.device if args.device else None)
    print("Validation metrics:")
    print(val_res)


if __name__ == "__main__":
    main()
