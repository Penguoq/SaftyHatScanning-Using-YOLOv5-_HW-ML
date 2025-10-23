# HatScan – YOLOv5 Safety Helmet Detection

โปรเจคนี้คือโมเดล YOLOv5 สำหรับตรวจจับหมวกนิรภัย (Safety Helmet) ที่เทรนบน Google Colab โดยใช้ชุดข้อมูล Annotated จาก Roboflow และบันทึกน้ำหนักโมเดล (`.pt`) ไว้เรียบร้อยแล้ว **คุณสามารถ clone repo นี้ แล้วรัน inference ได้ทันที** โดยไม่ต้องเทรนใหม่

> ✅ เหมาะสำหรับ: สาธิตผลลัพธ์, ใช้งาน inference, และเป็นพอร์ตงานขึ้น GitHub แบบอ่านง่ายหน้าตาสวย

---

## 🔧 โครงสร้าง Repo

```
HatScan-YOLOv5/
├─ assets/                 # รูปตัวอย่าง/ภาพผลลัพธ์ สำหรับ README
│  └─ samples/             # รูปทดสอบสำหรับ inference
├─ data/
│  └─ HatScan.yaml         # ไฟล์ dataset config (ปรับ path ให้ตรงกับของคุณ)
├─ notebooks/
│  └─ HatScan_inference.ipynb  # โน้ตบุ๊กสาธิตโหลดน้ำหนักและรันตรวจจับ
├─ scripts/
│  └─ predict.py           # สคริปต์รัน inference ผ่าน CLI (ใช้ yolov5.detect)
├─ weights/
│  └─ README.txt           # วิธีเก็บไฟล์ .pt (แนะนำใช้ Git LFS)
├─ .gitattributes          # ติดตั้ง Git LFS สำหรับไฟล์ใหญ่
├─ .gitignore
├─ requirements.txt
└─ README.md               # หน้าหลัก (ไฟล์นี้)
```

> ใส่ไฟล์น้ำหนักของคุณไว้ที่ `weights/` เช่น `weights/bestSaftyHatScanaaaaaaa.pt` (ชื่อไฟล์เปลี่ยนได้) แล้วอัปด้วย Git LFS

---

## 🚀 Quick Start (Inference อย่างเดียว)

1) ติดตั้ง YOLOv5 (ใช้เฉพาะ inference):
```bash
pip install git+https://github.com/ultralytics/yolov5.git
pip install -r requirements.txt
```

2) วางไฟล์น้ำหนักไว้ใน `weights/` ตัวอย่าง: `weights/bestSaftyHatScanaaaaaaa.pt`

3) เตรียมรูปทดสอบ วางไว้ที่ `assets/samples/` (สร้างโฟลเดอร์นี้ได้เอง)

4) รัน:
```bash
python scripts/predict.py --weights weights/bestSaftyHatScanaaaaaaa.pt --source assets/samples --imgsz 640
```
ผลลัพธ์จะถูกบันทึกไว้ในโฟลเดอร์ `runs/detect/exp*` ของ YOLOv5

---

## 🧠 Data & Classes

- Annotated ผ่าน Roboflow (train/val/test แบ่ง 80/10/10)
- จำนวนคลาส: `3`
- รายชื่อคลาส (เรียงให้ตรงกับตอนเทรน):
  1. `GapHat`
  2. `NoSafty_Hat`
  3. `SaftyHat`

แก้ไขพาธใน `data/HatScan.yaml` ให้ตรงกับที่คุณเก็บรูป/เลเบล เช่นพาธ Google Drive ของคุณ

```yaml
train: /content/drive/MyDrive/<YOUR_DATA_PATH>/images/train
val:   /content/drive/MyDrive/<YOUR_DATA_PATH>/images/val
test:  /content/drive/MyDrive/<YOUR_DATA_PATH>/images/test
```

---

## 📒 Colab Notebook (สาธิต)

เปิด `notebooks/HatScan_inference.ipynb` เพื่อ:
- ติดตั้ง YOLOv5
- โหลดน้ำหนักที่อยู่ใน `weights/`
- รันตรวจจับกับรูปตัวอย่างใน `assets/samples/`
- แสดงภาพพร้อมกล่องครอบผลลัพธ์

> หากต้องการลิงก์ปุ่มเปิดใน Colab (Badge) ให้แก้ URL ให้ชี้ไปไฟล์ใน GitHub ของคุณเองภายหลังอัปโหลด

---

## 📊 ผลลัพธ์ที่แนะนำให้โชว์ใน README

เพิ่มรูปภาพ/กราฟไว้ใน `assets/` แล้วแทรกใน README:
- `assets/hero.png` — รูปสวยๆ ของการตรวจจับ
- `assets/sample_pred1.jpg`, `assets/sample_pred2.jpg` — ภาพผลลัพธ์ตัวอย่าง
- (ถ้ามีจาก `runs/train/exp/`): `assets/results.png`, `assets/confusion_matrix.png`, `assets/PR_curve.png`

ตัวอย่างการแทรก:
```markdown
![Hero](assets/hero.png)
![Samples](assets/sample_pred1.jpg)
```

---

## 🏋️‍♀️ (ทางเลือก) คำสั่ง Train ที่ใช้

ถ้าต้องการใส่รายละเอียดการเทรน (ไม่จำเป็นต้องรันใหม่):
```bash
python train.py --img 640 --epochs 130 --data data/HatScan.yaml --weights yolov5s.pt
```
Logger ที่เคยใช้:
- Comet / ClearML / TensorBoard (ระบุวิธีตั้งค่าไว้สั้นๆ ตามที่คุณใช้งานจริง)

---

## 📦 Weights (.pt) ควรจัดการอย่างไร?

### ทางเลือก A: Git LFS
แนะนำสำหรับการเก็บใน repo โดยตรง
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add weights/bestSaftyHatScanaaaaaaa.pt
git commit -m "Add trained weights (LFS)"
git push
```

### ทางเลือก B: GitHub Releases
อัปโหลดไฟล์ `.pt` ไปที่ Releases แล้วใส่ลิงก์ดาวน์โหลดใน README แทน

---

## 📝 License
เลือกลิขสิทธิ์ที่เหมาะสมกับโปรเจคของคุณ (เช่น MIT) แล้วเพิ่มไฟล์ `LICENSE`

---

## 🙌 Credits
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- ขอบคุณ Roboflow สำหรับเครื่องมือช่วย Annotate# SaftyHatScanning-YOLOV5-
