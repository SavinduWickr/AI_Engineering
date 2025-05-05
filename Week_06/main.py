#%%
import os
import random
import shutil
import subprocess
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
import pandas as pd
import torch
import cv2

# Adjust these two to match your setup:
BASE     = Path('/home/admin/Documents/AI_Eng/Week_06')
YOLO_DIR = BASE / 'yolov5'

# Device for inference/training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
def convert_annotations(csv_path: Path, labels_dir: Path, class_map={'Graffiti': 0}):
    """
    Convert CSV bbox annotations to YOLO txt files.
    """
    df = pd.read_csv(csv_path)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_name, group in df.groupby('filename'):
        w, h = group[['width','height']].iloc[0]
        lines = []
        for _, row in group.iterrows():
            cid = class_map[row['class']]
            x_c = ((row.xmin + row.xmax) / 2) / w
            y_c = ((row.ymin + row.ymax) / 2) / h
            bw  = (row.xmax - row.xmin) / w
            bh  = (row.ymax - row.ymin) / h
            lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        out_file = labels_dir / f"{Path(img_name).stem}.txt"
        out_file.write_text("\n".join(lines))

#%%
def prepare_dataset(src_img_train: Path,
                    src_img_test:  Path,
                    lbl_train_dir:  Path,
                    lbl_test_dir:   Path,
                    dst:            Path,
                    n_train=400,
                    n_val=40,
                    seed=None):
    """
    Sample and copy images + labels into YOLOv5 structure, then write graffiti.yaml.
    """
    random.seed(seed)
    dst_img_tr = dst/'images'/'train'
    dst_img_va = dst/'images'/'val'
    dst_lbl_tr = dst/'labels'/'train'
    dst_lbl_va = dst/'labels'/'val'
    for p in (dst_img_tr, dst_img_va, dst_lbl_tr, dst_lbl_va):
        p.mkdir(parents=True, exist_ok=True)

    all_train = list(src_img_train.glob('*.jpg'))
    all_test  = list(src_img_test.glob('*.jpg'))
    train_sel = random.sample(all_train, n_train)
    val_sel   = random.sample(all_test,  n_val)

    for imgs, dst_img, src_lbl, dst_lbl in [
        (train_sel, dst_img_tr, lbl_train_dir, lbl_tr:=dst_lbl_tr),
        (val_sel,   dst_img_va, lbl_test_dir,  lbl_va:=dst_lbl_va)
    ]:
        for img in imgs:
            shutil.copy(img, dst_img/img.name)
            txt = src_lbl / f"{img.stem}.txt"
            if txt.exists():
                shutil.copy(txt, dst_lbl/f"{img.stem}.txt")

    # Write dataset YAML
    yaml_dict = {
        'path': str(dst),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    1,
        'names': ['Graffiti']
    }
    with open(dst/'graffiti.yaml','w') as f:
        yaml.dump(yaml_dict, f)

#%%
def train_yolo(data_yaml: str,
               weights:    str,
               exp_name:   str,
               epochs=30,
               img_size=640,
               batch=16,
               cache=True):
    """
    Runs yolov5/train.py from inside YOLO_DIR and prints full logs on error.
    """
    cmd = [
        'python', 'train.py',
        '--img',   str(img_size),
        '--batch', str(batch),
        '--epochs',str(epochs),
        '--data',  data_yaml,
        '--weights', weights,
        '--name',  exp_name
    ] + (['--cache'] if cache else [])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(YOLO_DIR),
            check=True,
            capture_output=True,
            text=True
        )
        print(proc.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ train.py failed (exit {e.returncode})\n")
        print("— STDOUT —\n", e.stdout)
        print("— STDERR —\n", e.stderr)
        raise

#%%
def load_ground_truth(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        _, x_c, y_c, w, h = map(float, line.split())
        x1 = (x_c - w/2) * img_w
        y1 = (y_c - h/2) * img_h
        x2 = (x_c + w/2) * img_w
        y2 = (y_c + h/2) * img_h
        boxes.append([x1,y1,x2,y2])
    return boxes

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA)*max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

#%%
def evaluate_model(model, img_paths, lbl_dir, save_csv):
    rows = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h,w = img.shape[:2]
        preds = model(img).xyxy[0].cpu().numpy()
        gts   = load_ground_truth(lbl_dir/f"{img_path.stem}.txt", w, h)

        if len(preds)==0:
            rows.append([img_path.name, 0.0, 0.0])
        else:
            best = preds[preds[:,4].argmax()]
            x1,y1,x2,y2,conf,_ = best
            ious = [compute_iou([x1,y1,x2,y2], g) for g in gts]
            rows.append([img_path.name, float(conf), max(ious) if ious else 0.0])

    df = pd.DataFrame(rows, columns=['image_name','confidence','IoU'])
    df.to_csv(save_csv, index=False)
    return (df['IoU']>=0.9).mean()

#%%
def real_time_detection(weights_path: str, class_names=['Graffiti']):
    model = torch.hub.load('ultralytics/yolov5','custom',
                           path=weights_path,
                           force_reload=False).to(DEVICE)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        for *box, conf, cls in model(frame).xyxy[0].cpu().numpy():
            x1,y1,x2,y2 = map(int,box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"{class_names[int(cls)]} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow('det', frame)
        if cv2.waitKey(1)==ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

#%%
# 1) Convert CSV → YOLO txt
convert_annotations(BASE/'train_labels.csv', BASE/'labels'/'train')
convert_annotations(BASE/'test_labels.csv',  BASE/'labels'/'test')

#%%

# 2) Prepare dataset + graffiti.yaml
prepare_dataset(
    src_img_train=BASE/'images'/'train',
    src_img_test= BASE/'images'/'test',
    lbl_train_dir=BASE/'labels'/'train',
    lbl_test_dir= BASE/'labels'/'test',
    dst=          BASE/'dataset',
    seed=42
)


#%%

# 3) Iterative training & evaluation
data_yaml   = str(BASE/'dataset'/'graffiti.yaml')
best_weights = 'yolov5s.pt'  # initial pretrained weights

for i in range(1, 11):
    exp = f'graffiti_exp{i}'
    print(f"\n>>> Iteration {i}: training {exp}")
    train_yolo(data_yaml, best_weights, exp, epochs=30)

    # load the newly trained weights
    wpath = YOLO_DIR/'runs'/'train'/exp/'weights'/'best.pt'
    model = torch.hub.load('ultralytics/yolov5','custom',
                           path=str(wpath),
                           force_reload=False).to(DEVICE)

    # evaluate on val set
    val_imgs = list((BASE/'dataset'/'images'/'val').glob('*.jpg'))
    rate = evaluate_model(model, val_imgs,
                          BASE/'dataset'/'labels'/'val',
                          save_csv=f'iteration{i}_results.csv')
    print(f"Iteration {i}: {rate*100:.1f}% images ≥ 0.9 IoU")

    best_weights = str(wpath)
    if rate >= 0.8:
        print("Target reached — stopping loop.")
        break
#%%
