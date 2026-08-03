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
