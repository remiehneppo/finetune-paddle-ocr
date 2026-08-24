# Fine-tune PP-OCRv6 detection từ dữ liệu labeler

Sau khi bấm `Xuất nhãn detection`, có thể trộn nhiều workspace và chỉ kiểm tra
dữ liệu trước khi train:

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir ./PaddleOCR \
  --work-dir runs/vi_det_check \
  --prepare-only
```

Nếu `prepared/summary.json` và `prepared/rejected.jsonl` đã ổn, chạy fine-tune
và tự export checkpoint tốt nhất sang định dạng inference:

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir ./PaddleOCR \
  --work-dir runs/vi_det_v1 \
  --pretrained-model ./models/PP-OCRv6_medium_det_pretrained.pdparams \
  --export-after-train
```

`--dataset-dir` nhận workspace ảnh, chính thư mục
`.paddleocr-det-labeler`, hoặc đường dẫn `det_labels.txt`. Script kiểm tra ảnh
decode được, giới hạn kích thước, đường dẫn không thoát workspace, JSON/point
hợp lệ, polygon không tự cắt/không suy biến; box lỗi được loại và ghi rõ vào
`rejected.jsonl`. Ảnh trùng nội dung được khử theo SHA-256 trước khi chia
train/validation để tránh leakage. Ảnh hợp lệ được hard-link (fallback copy)
vào run nên ảnh nguồn và annotation nguồn không bị sửa.

Mặc định cho RTX 5060 Ti 16 GB là crop native `640x640`, batch 4, AMP, 4 worker,
learning rate `1e-4`, 100 epoch. Nếu CUDA OOM, giảm `--batch-size 3`, rồi `2`;
không giảm crop trước vì chữ nhỏ trên trang sẽ mất chi tiết. Với dataset nhỏ,
hãy ưu tiên bổ sung dữ liệu (PaddleOCR khuyến nghị tối thiểu khoảng 500 ảnh
detection) và xem Hmean validation thay vì tăng epoch liên tục.

Script luôn clone config chính thức
`configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml`, giữ nguyên `Architecture`,
`Loss` và thứ tự augmentation. Trước khi train, toàn bộ tensor trong file
`.pdparams` được so shape với model; inference directory sẽ bị từ chối để tránh
load nhầm weight và fine-tune một phần kiến trúc.

Output chính:

```text
runs/vi_det_v1/
├── prepared/
│   ├── images/
│   ├── train.txt
│   ├── validation.txt
│   ├── rejected.jsonl
│   └── summary.json
├── pretrained/
├── resolved_config.yml
├── output/
└── inference/best_accuracy/   # khi có --export-after-train
```

Các lý do loại dữ liệu được thống kê trong `summary.json` và ghi chi tiết theo
dòng/box trong `rejected.jsonl`. Transcription không dùng cho bài toán detection:
box thường được canonical hóa thành `text`, còn box ignore giữ `###` đúng contract
`DetLabelEncode` của PaddleOCR.

## Quy trình chạy đầy đủ

### 1. Cài môi trường và weight

Dùng virtualenv PP-OCRv6 ở root repository, không dùng môi trường ERNIEKit VL:

```bash
source .venv/bin/activate
python -m pip install -r PaddleOCR/requirements.txt
python -m pip install -r requirements.txt
./download_pretrained_models.sh det
python -c "import paddle; paddle.utils.run_check()"
```

### 2. Validate và stage dataset

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir ./PaddleOCR \
  --work-dir /media/tieubaoca/HDD1/F/finetune-output/det_prepare \
  --validation-ratio 0.10 \
  --seed 2026 \
  --prepare-only
```

Đọc `prepared/summary.json` và `prepared/rejected.jsonl`. Chỉ train khi cả
`train.txt` và `validation.txt` có sample hợp lệ, split không bị duplicate hash
và rejection đã được hiểu rõ.

### 3. Train và export

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir ./PaddleOCR \
  --work-dir /media/tieubaoca/HDD1/F/finetune-output/det_v1 \
  --pretrained-model ./models/PP-OCRv6_medium_det_pretrained.pdparams \
  --epochs 100 \
  --learning-rate 1e-4 \
  --batch-size 4 \
  --num-workers 4 \
  --eval-batch-step 200 \
  --save-epoch-step 5 \
  --export-after-train
```

Nếu OOM, giảm `--batch-size` xuống `3`, sau đó `2`. Chỉ dùng `--disable-amp`
khi AMP/runtime gây lỗi; tắt AMP thường làm tăng VRAM.

## Tham số `finetune_det.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--dataset-dir PATH [PATH ...]` | Bắt buộc | Một hoặc nhiều workspace, `.paddleocr-det-labeler` hoặc `det_labels.txt`. |
| `--paddleocr-dir PATH` | `./PaddleOCR` | Checkout PaddleOCR chứa `tools/train.py`. |
| `--work-dir PATH` | `runs/vi_det_YYYYmmdd_HHMMSS` | Run mới; script từ chối thư mục đã có nội dung. |
| `--config VALUE` | `configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml` | Config relative với checkout hoặc absolute path. Architecture/loss/transform chain phải giữ nguyên. |
| `--pretrained-model VALUE` | URL official medium det | URL hoặc file training `.pdparams`. Inference directory và suffix khác bị từ chối. |
| `--validation-ratio FLOAT` | `0.10` | Tỷ lệ validation, trong `(0, 0.5)`. Split theo source sau deduplicate SHA-256. |
| `--seed INT` | `2026` | Seed split và shuffle. |
| `--epochs INT` | `100` | Số epoch trong config resolved. |
| `--learning-rate FLOAT` | `1e-4` | Learning rate ban đầu. |
| `--batch-size INT` | `4` | Batch train mỗi card. |
| `--num-workers INT` | `4` | Data loader workers. |
| `--eval-batch-step INT` | `200` | Chu kỳ validation theo training step. |
| `--save-epoch-step INT` | `5` | Chu kỳ lưu checkpoint theo epoch. |
| `--max-image-pixels INT` | `50_000_000` | Ảnh lớn hơn giới hạn bị reject trước staging. |
| `--min-polygon-area FLOAT` | `4.0` | Polygon nhỏ hơn diện tích này bị reject. |
| `--disable-amp` | Tắt | Tắt mixed precision; mặc định AMP bật. |
| `--prepare-only` | Tắt | Chỉ validate, stage và ghi config; không chạy GPU/train. |
| `--export-after-train` | Tắt | Export best checkpoint sang `inference/best_accuracy/`. |

## Failure gates

- `--validation-ratio` phải trong `(0, 0.5)`; các count/rate phải dương.
- `--paddleocr-dir` phải có `tools/train.py`; config phải tồn tại.
- Label path không được thoát workspace; JSON, points, polygon và ảnh phải hợp lệ.
- Pretrained phải là `.pdparams` và khớp shape toàn bộ PP-OCRv6 detector.
- Script giữ nguyên `Architecture`, `Loss` và tên/thứ tự transform của config gốc.
- Nếu GradScaler không tương thích NumPy, cài dependency đã pin (`numpy<2.4`).
- `--export-after-train` fail nếu training không tạo best checkpoint.

## Artifact cần bàn giao

```text
<work-dir>/prepared/summary.json
<work-dir>/prepared/rejected.jsonl
<work-dir>/resolved_config.yml
<work-dir>/output/
<work-dir>/inference/best_accuracy/   # nếu export
```

Ghi kèm command, git SHA, pretrained path/hash, dataset manifest/hash, peak VRAM,
best Hmean/precision/recall và inference path.
