# Fine-tune PaddleOCR-VL-1.6 tiếng Việt bằng ERNIEKit LoRA

Pipeline này dành cho ảnh crop dòng, bảng, công thức hoặc biểu đồ trên cùng
một script `finetune_vl.py`. Mỗi sample lấy ground truth từ cột `label`/`text`
của dataset đã chỉ định; script không chuyển đổi format target. Prompt theo
task (`OCR:`, `Table Recognition:`, `Formula Recognition:`,
`Chart Recognition:`) và image token được mask. Backend là stage `OCR-VL-SFT`
của ERNIEKit, không gọi `PaddleOCR/tools/train.py`.

## Môi trường tách biệt

Không cài ERNIEKit/PaddleOCR-VL vào virtualenv PP-OCRv6 hiện có.

Môi trường CPU để chuẩn bị dữ liệu:

```bash
python -m venv .venv-vl-prepare
source .venv-vl-prepare/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-vl-prepare.txt
```

Môi trường GPU để train nên được tạo trong checkout ERNIEKit `release/v1.5`
theo [hướng dẫn SFT chính thức](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft.md).
Với RTX 50/SM120, dùng Paddle CUDA 12.9 theo
[hướng dẫn Blackwell của PaddleOCR](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html),
checkout ERNIEKit ở revision ghi trong `requirements-vl-erniekit.txt`, rồi cài
đúng các version runtime trong file đó và `pip install -e .`. Script fail-fast nếu
Git revision hoặc Paddle/PaddleFormers/Transformers lệch profile đã pin. Script ưu tiên `<erniekit-dir>/.venv/bin/python`, rồi
`<erniekit-dir>/venv/bin/python`; nếu không có, nó dùng Python đang chạy.

## Model

Script không tự tải weights khi chỉ được import hoặc chạy test. Tải snapshot
đầy đủ bằng lệnh riêng sau; download có resume và mặc định pin revision đã kiểm
tra của PaddleOCR-VL-1.6:

```bash
./download_pretrained_models.sh vl ./models
```

Có thể pin revision khác rõ ràng:

```bash
./download_pretrained_models.sh vl ./models --revision <commit-or-tag>
```

Sau đó luôn truyền model local để bước prepare/train không tải weights từ Hub.
Trên máy hiện tại snapshot đã có tại:

```bash
MODEL=/home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6
```

## Chuẩn bị dữ liệu

Input là một hoặc nhiều thư mục Hugging Face `save_to_disk()` hoặc snapshot
Parquet có cột `image` và `label`/`text`. Nếu `label` rỗng, `text` được dùng làm
fallback. Có thể thêm cột `task` với giá trị `ocr|table|formula|chart` để trộn
nhiều loại trong một run; nếu thiếu cột này, dùng `--task` (mặc định `ocr`).
Target giữ nguyên từ dataset đã chỉ định. OCR vẫn flatten whitespace; layout
giữ newline. Ảnh phải giải mã thật, có hai chiều lớn hơn 1 pixel, nằm dưới
`--max-image-pixels`, và được materialize thành PNG RGB.

```bash
python finetune_vl.py \
  --dataset-dir /data/ocr_a /data/ocr_b \
  --model "$MODEL" \
  --work-dir runs/vl16_vi_prepare \
  --prepare-only
```

Mỗi nguồn giữ validation riêng. Nếu nguồn chưa có `validation`, `valid` hoặc
`dev`, script tách holdout với seed riêng cho nguồn. ERNIEKit nhận một JSONL cho
mỗi nguồn và xác suất trộn chuẩn hóa theo `sqrt(số sample)`.

### Trộn table, formula và chart

Dùng chung `finetune_vl.py` và để target nằm sẵn trong dataset. Trộn nhiều
layout trong một prepare/train để giảm nguy cơ LoRA làm hỏng các output layout
khác. Mỗi sample cần cột `task`; JSONL ERNIEKit vẫn đặt ảnh khớp
`text_info[0]`, prompt `mask`, target `no_mask`.

```bash
python finetune_vl.py \
  --dataset-dir /data/layout_mixed \
  --model "$MODEL" \
  --work-dir runs/vl16_layout_mixed_prepare \
  --prepare-only

python finetune_vl.py \
  --prepared-from runs/vl16_layout_mixed_prepare \
  --erniekit-dir /opt/ERNIE \
  --model "$MODEL" \
  --work-dir runs/vl16_layout_mixed_train
```

Nếu một dataset chỉ có một loại và không có cột `task`, đặt
`--task table|formula|chart` làm mặc định cho toàn bộ sample của run đó.

### Tạo crop layout bằng VL Layout Labeler

Tool `run_vl_layout_labeler.py` nằm ngoài `ocr_labeler/` và dùng sidecar riêng
`.paddleocr-vl-labeler`. Sidecar version 2 giữ đủ 25 class PP-DocLayoutV3 và
task VL nullable trên cùng block. Tool gửi riêng block có task
`ocr|table|formula|chart` tới OpenAI-compatible `llama-server
/v1/chat/completions` với `temperature=0`, đồng thời giữ block layout-only để
export COCO instance segmentation.

```bash
python run_vl_layout_labeler.py \
  --images /data/pages \
  --layout-model-dir /home/tieubaoca/.paddlex/official_models/PP-DocLayoutV3 \
  --vl-base-url http://127.0.0.1:8000/v1 \
  --vl-model paddleocr-vl \
  --port 8012
```

Trong UI, detect layout trước, prelabel từng vùng hoặc cả trang, sửa toàn bộ
layout label/task/text/bbox rồi bấm `Complete`. `Export HF` giữ schema
`image,text,task` và loại layout-only. `Export Layout` tạo `images/`, hai COCO
JSON train/validation và manifest; ảnh toàn trang được sao chép byte-for-byte.
`Export All` tạo nguyên tử `<root>/vl/` và `<root>/layout/`.

Nhánh VL dùng trực tiếp với `--dataset-dir <root>/vl --prepare-only`. Nhánh
layout kiểm tra/train bằng PaddleX 3.7.2 như sau:

```bash
PADDLEX_CONFIG=.venv/lib/python3.12/site-packages/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml

.venv/bin/python -c 'from paddlex.engine import Engine; Engine().run()' \
  -c "$PADDLEX_CONFIG" \
  -o Global.mode=check_dataset \
  -o Global.dataset_dir=<root>/layout

.venv/bin/python -c 'from paddlex.engine import Engine; Engine().run()' \
  -c "$PADDLEX_CONFIG" \
  -o Global.mode=train \
  -o Global.dataset_dir=<root>/layout \
  -o Train.num_classes=25
```

### Dùng lại dataset đã prepare

`--prepared-from` tạo run train mới nhưng tham chiếu trực tiếp JSONL và ảnh của
run `--prepare-only` cũ. Script kiểm tra toàn bộ record, mask contract, số lượng
sample và sự tồn tại của file ảnh; ảnh không bị giải mã lại hoặc sao chép. Vì
vậy không được xóa hay di chuyển run prepare trong khi còn dùng checkpoint này.
`summary.json` ghi `tasks`/`prompts` quan sát được; run mixed đặt
`task=mixed`. Summary cũ không có `task` vẫn được hiểu là OCR. Evaluator và merge
đọc prompt từ từng dòng JSONL, không khóa một task cho cả run.

Với dataset và model hiện có trên máy này, chạy full LoRA bằng lệnh:

```bash
source /tmp/paddleocr-vl-prepare-venv/bin/activate
python -u finetune_vl.py \
  --prepared-from /home/tieubaoca/AI/ocr/paddle-ocr/runs/vl16_vi_all_datasets_prepare \
  --erniekit-dir /tmp/paddleocr-vl-erniekit-reference \
  --model /home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6 \
  --work-dir /home/tieubaoca/AI/ocr/paddle-ocr/runs/vl16_vi_full_v2 \
  --epochs 2.5 \
  --devices 0
```

Run mới chỉ chứa `summary.json`, `resolved.yaml`, checkpoint, log và metric; nó
không có bản sao `prepared/`. Không kết hợp `--prepared-from` với
`--dataset-dir`, `--prepare-only` hoặc `--resume-from`.

## Inspect và GPU smoke test

Trước mọi lần train, script chạy preflight `do_train=false`: model được wrap
LoRA, ERNIEKit in số trainable parameters, và script dừng nếu không thấy LoRA
hoặc nếu hơn 20% base model trainable. Có thể chỉ chạy preflight:

```bash
python finetune_vl.py \
  --prepared-from runs/vl16_vi_prepare \
  --erniekit-dir /opt/ERNIE \
  --model "$MODEL" \
  --work-dir runs/vl16_vi_inspect \
  --inspect-model
```

Smoke test ba bước trên một run mới:

```bash
python finetune_vl.py \
  --prepared-from runs/vl16_vi_prepare \
  --erniekit-dir /opt/ERNIE \
  --model "$MODEL" \
  --work-dir runs/vl16_vi_smoke \
  --smoke-steps 3
```

Profile mặc định: LoRA rank 32, vision encoder đóng băng, BF16/O2,
FlashAttention, full recompute, micro-batch 1, packing 1, accumulation 16,
sequence 2048, `50,176–451,584` pixel, LR `1e-4`, cosine, warmup 3%, weight
decay `0.01`, 3 epoch, hai worker và prefetch 2. Dùng
`--no-flash-attention` nếu build/hardware không tương thích.

OCR-VL tạo `IterableDataset` không có độ dài, nên ERNIEKit bắt buộc `max_steps`
dương. Script tự tính `ceil(samples × epochs / (batch × packing × accumulation))`;
với 220.691 sample, 2,5 epoch và effective batch 16, config nhận `max_steps: 34483`.
`--smoke-steps` vẫn override giá trị này cho smoke test.

Với snapshot Hugging Face chính thức, pipeline bật `use_huggingface_model` để
LoRA phủ đủ `q/k/v/o/up/gate/down` của decoder. Hook runtime giới hạn regex vào
`model.layers.*`, vì target mặc định của ERNIEKit cũng khớp `visual.*` sau khi
vision đã freeze. Preflight sẽ dừng nếu hook này không được nạp; trước export,
adapter tiếp tục bị kiểm tra để không có bất kỳ tensor vision nào.

Peak VRAM được poll bằng `nvidia-smi` trong inspect/train và ghi vào `metrics/`.
Nếu không có `nvidia-smi`, trường này là `null` thay vì đoán.

## Resume

Resume yêu cầu checkpoint nằm trong chính `--work-dir`. `resolved.yaml` gốc
không bị sửa; script tạo `resolved-resume.yaml` mới và không bật overwrite:

```bash
python finetune_vl.py \
  --erniekit-dir /tmp/paddleocr-vl-erniekit-reference \
  --model /home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6 \
  --work-dir /home/tieubaoca/AI/ocr/paddle-ocr/runs/vl16_vi_full_v2 \
  --resume-from /home/tieubaoca/AI/ocr/paddle-ocr/runs/vl16_vi_full_v2/adapter/checkpoint-1200
```

## Artifact

```text
runs/vl16_vi_full/
├── prepared/
│   ├── images/source-*/
│   ├── train-source-*.jsonl
│   └── validation-source-*.jsonl
├── rejected.jsonl
├── summary.json
├── resolved.yaml
├── adapter/                 # LoRA + checkpoints ERNIEKit
│   └── export/              # model đã merge + verification reports
├── export.yaml
├── export_manifest.json
├── logs/
├── metrics/                 # VRAM, checkpoint selection, CER, predictions
└── tensorboard_logs/
```

PaddleFormers 0.4.0 `MergeKit` chưa nhận diện model type custom `paddleocr_vl` và
có đường làm tròn BF16 khác runtime adapter. Pipeline vì vậy dùng API đã pin
`LoRAModel.merge()`, lọc LoRA tensors khỏi state dict rồi lưu lại ở định dạng
Hugging Face. Nó copy config, tokenizer, processor, custom model code,
`chat_template.jinja`, `inference.yml` và generation config từ base local.

Trước merge, evaluator greedy chấm base, adapter cuối và các checkpoint gần nhất
bằng CER (exact match là tie-breaker), rồi chỉ merge checkpoint thắng.
`merge_verification.json` xác nhận công thức weight; `logits_verification.json`
xác nhận model reload khớp bản merge trong RAM và drift BF16 nằm trong numeric
tolerance. Pha đánh giá thứ hai so output của adapter đã chọn với merged model;
`export_manifest.json` chỉ được công nhận khi normalized edit similarity đạt 0,99.
Full run fail-fast nếu không đạt; smoke run vẫn ghi trạng thái để chẩn đoán model
chưa học đủ. CER, exact match và normalized edit distance được báo tổng thể và
theo từng nguồn.

Lưu ý về ERNIEKit `release/v1.5`: workflow `OCR-VL-SFT` chính thức hiện để
`eval_dataset=None`, nên bật `do_eval` sẽ lỗi thay vì tạo validation loss. Config
resolved vì vậy giữ `do_eval: false`, nhưng vẫn xuất validation JSONL cho đánh
giá CER deterministic. Không tuyên bố checkpoint "best validation loss" khi
backend chưa cung cấp contract đó; checkpoints được giữ để so CER sau train.

## Test

```bash
PYTHONPATH=. pytest -q tests/test_finetune_vl.py tests/test_finetune.py
PYTHONPATH=. pytest -q tests/test_finetune_vl_layout.py
bash -n download_pretrained_models.sh
```
