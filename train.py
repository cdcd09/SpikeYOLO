import argparse
import os
import shlex
import subprocess
import sys


def parse_args():
	p = argparse.ArgumentParser(description="Train YOLOv8 (CLI wrapper) with optional W&B logging")
	p.add_argument("--data", type=str, default="/data/obstacles-5_yolov5/data.yaml", help="path to data.yaml")
	p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model cfg or weights (e.g., yolov8n.pt or /workspace/ckpt/best.pt)")
	p.add_argument("--epochs", type=int, default=50)
	p.add_argument("--batch", type=int, default=16)
	p.add_argument("--imgsz", type=int, default=640)
	p.add_argument("--device", type=str, default="0,1,2,3", help="device string, e.g. '0', '0,1' or 'cpu'")
	p.add_argument("--project", type=str, default="runs/detect", help="Project dir for saving runs (Ultralytics default for detection)")
	p.add_argument("--name", type=str, default="obstacles-yolov8", help="Run name")
	p.add_argument("--resume", action="store_true", help="Resume last run")
	p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (requires wandb login)")
	return p.parse_args()


def run(cmd: str, extra_env=None):
	print(f"\n>> {cmd}")
	env = os.environ.copy()
	if extra_env:
		env.update(extra_env)
	p = subprocess.Popen(shlex.split(cmd), env=env)
	return p.wait()


def main():
	args = parse_args()

	# Control W&B via env var to avoid accidental hangs when not logged in
	env = {}
	if args.wandb:
		env["WANDB_DISABLED"] = "false"
	else:
		env["WANDB_DISABLED"] = "true"

	train_cmd = (
		f"yolo train model={shlex.quote(args.model)} "
		f"data={shlex.quote(args.data)} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
		f"project={shlex.quote(args.project)} name={shlex.quote(args.name)}"
	)
	if args.device:
		train_cmd += f" device={shlex.quote(args.device)}"
	if args.resume:
		train_cmd += " resume=True"

	code = run(train_cmd, extra_env=env)
	if code != 0:
		sys.exit(code)

	# Try to locate best weights and run validation for precision/recall/F1
	# Ultralytics default save path: {project}/{name}/weights/best.pt
	weights = os.path.join(args.project, args.name, "weights", "best.pt")
	if not os.path.exists(weights):
		# fallback to last.pt
		weights = os.path.join(args.project, args.name, "weights", "last.pt")

	val_cmd = (
		f"yolo val model={shlex.quote(weights)} data={shlex.quote(args.data)} imgsz={args.imgsz}"
	)
	if args.device:
		val_cmd += f" device={shlex.quote(args.device)}"

	run(val_cmd, extra_env=env)


if __name__ == "__main__":
	main()

