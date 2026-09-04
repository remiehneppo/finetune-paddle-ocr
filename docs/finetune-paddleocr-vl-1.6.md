# Fine-tune PaddleOCR-VL-1.6 tiếng Việt bằng ERNIEKit LoRA

Pipeline này dành cho ảnh crop dòng, bảng, công thức hoặc biểu đồ trên cùng
một script `finetune_vl.py`. Mỗi sample lấy ground truth từ cột `label`/`text`
của dataset đã chỉ định; script không chuyển đổi format target. Prompt theo
task (`OCR:`, `Table Recognition:`, `Formula Recognition:`,
`Chart Recognition:`) và image token được mask. Backend là stage `OCR-VL-SFT`
của ERNIEKit, không gọi `PaddleOCR/tools/train.py`.

Contract target theo tài liệu ERNIEKit chính thức: table là **OTSL**
(`<fcel>`, `<ecel>`, `<xcel>`, `<lcel>`, `<ucel>`, `<nl>`), formula là LaTeX,
chart là bảng Markdown. HTML table bị từ chối thay vì train sai schema.

## Môi trường tách biệt

Không cài ERNIEKit/PaddleOCR-VL vào virtualenv PP-OCRv6 hiện có.

Môi trường CPU để chuẩn bị dữ liệu:

```bash
python -m venv .venv-vl-prepare
source .venv-vl-prepare/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-vl-prepare.txt
```

Môi trường GPU để train yêu cầu checkout ERNIEKit `release/v1.5` tại đúng commit đã được pin trong `requirements-vl-erniekit.txt`.

Các bước cài đặt chuẩn cho ERNIEKit runtime:

```bash
# 1. Clone ERNIEKit và checkout đúng revision đã xác minh (branch release/v1.5)
git clone https://github.com/PaddlePaddle/ERNIE.git /path/to/erniekit
cd /path/to/erniekit
git checkout 790a50b045d1aca2753d5395d8bec0806b2e6925

# 2. Tạo virtualenv chuyên dụng cho GPU (Python 3.10 - 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 3. Cài đặt các thư viện theo phiên bản ghim cố định
# Đối với CUDA 12.9 (RTX 50 / SM120):
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
# (Hoặc thay bằng cu126 nếu hệ thống dùng CUDA 12.6)

pip install paddleformers==0.4.0 safetensors==0.7.0 transformers==4.55.4 ml_dtypes==0.5.4
pip install -e .
```

Script fail-fast nếu Git revision lệch `790a50b045d1aca2753d5395d8bec0806b2e6925` hoặc phiên bản Paddle/PaddleFormers/Transformers không khớp danh sách trên. Script tự động ưu tiên nạp Python tại `<erniekit-dir>/.venv/bin/python`, rồi `<erniekit-dir>/venv/bin/python`.

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
`--work-dir` và mọi merged output bắt buộc nằm ngoài snapshot model; script
fail-closed nếu đường dẫn có thể ghi artifact vào thư mục base model.
Trên máy hiện tại snapshot đã có tại:

```bash
MODEL=/home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6
```

## Chuẩn bị dữ liệu

Input là một hoặc nhiều thư mục Hugging Face `save_to_disk()` hoặc snapshot
Parquet có cột `image` và `label`/`text`. Nếu `label` rỗng, `text` được dùng làm
fallback. Có thể thêm cột `task` với giá trị `ocr|table|formula|chart` để trộn
nhiều loại trong một run. Với một nguồn thiếu cột này, dùng `--task` (mặc định
`ocr`). Với nhiều nguồn có nguồn thiếu `task`, bắt buộc truyền một
`--dataset-task` cho mỗi `--dataset-dir`; script không âm thầm gán tất cả nguồn
thành OCR. Target giữ nguyên newline cho cả OCR và layout. Ảnh phải giải mã
thật, có hai chiều lớn hơn 1 pixel, nằm dưới `--max-image-pixels`, được
materialize thành PNG RGB và phải fit token budget. Sample không fit bị loại ở
bước prepare và ghi vào `rejected.jsonl`, không bị âm thầm bỏ qua trong lúc train.

```bash
python finetune_vl.py \
  --dataset-dir /data/ocr_a /data/ocr_b \
  --dataset-task ocr ocr \
  --model "$MODEL" \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --work-dir runs/vl16_vi_prepare \
  --prepare-only
```

Mỗi nguồn giữ validation riêng. Nếu nguồn chưa có `validation`, `valid` hoặc
`dev`, script tách holdout với seed riêng cho nguồn. ERNIEKit nhận một JSONL cho
mỗi nguồn và xác suất trộn chuẩn hóa theo `sqrt(số sample)`. `--prepare-only`
không load model weights hoặc chạy train, nhưng vẫn load tokenizer từ `--model`
để tính token budget; nên truyền snapshot model local nếu muốn prepare offline.

### Trộn table, formula và chart

Dùng chung `finetune_vl.py` và để target nằm sẵn trong dataset. Trộn nhiều
layout trong một prepare/train để giảm nguy cơ LoRA làm hỏng các output layout
khác. Mỗi sample cần cột `task`; JSONL ERNIEKit vẫn đặt ảnh khớp
`text_info[0]`, prompt `mask`, target `no_mask`.

```bash
python finetune_vl.py \
  --dataset-dir /data/layout_mixed \
  --model "$MODEL" \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --work-dir runs/vl16_layout_mixed_prepare \
  --prepare-only

python finetune_vl.py \
  --prepared-from runs/vl16_layout_mixed_prepare \
  --erniekit-dir /opt/ERNIE \
  --model "$MODEL" \
  --work-dir runs/vl16_layout_mixed_train
```

Nếu một dataset duy nhất chỉ có một loại và không có cột `task`, đặt
`--task table|formula|chart`. Nếu có nhiều dataset đơn-task, dùng mapping vị trí:

```bash
python finetune_vl.py --prepare-only \
  --dataset-dir /data/ocr /data/formula /data/table \
  --dataset-task ocr formula table \
  --model "$MODEL" \
  --work-dir runs/vl16_mixed_prepare
```

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
layout label/task/text/bbox rồi bấm `Complete`. `Export HF` split theo trang
trước khi flatten block crop, tạo `DatasetDict` `train`/`validation` với schema
`image,text,task,source_page_id` và loại layout-only. Mọi block của cùng trang
luôn ở cùng split. `Export Layout` tạo `images/`, hai COCO
JSON train/validation và manifest; ảnh toàn trang được sao chép byte-for-byte.
`Export All` tạo nguyên tử `<root>/vl/` và `<root>/layout/`.

Editor output có hai tab đồng bộ: `Trực quan` và `Raw`. OCR được tách theo dòng;
table được convert OTSL sang bảng HTML thật với `rowspan`/`colspan` để sửa và
merge/split ô, rồi convert ngược về OTSL canonical; formula giữ nguyên LaTeX với
preview; chart là bảng Markdown có căn lề. Sửa raw chỉ cập nhật
visualize khi parse thành công; raw chưa hợp lệ vẫn được giữ nguyên để người dùng
tiếp tục sửa. Cả frontend, bước `Complete` và export đều kiểm tra cùng contract.

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

Có thể truyền nhiều run qua cùng option. Khi đó phải truyền đúng một relative
weight dương, hữu hạn cho mỗi run; script normalize tự động, giữ probability nội
bộ của từng run, rồi scale source probability cho cả train và validation. Source
được flatten theo thứ tự run rồi thứ tự source; task được union. Mọi run phải ghi
cùng base model trong metadata.

```bash
python -u finetune_vl.py \
  --prepared-from runs/vl16_vi_prepare runs/vl16_labeler_prepare \
  --prepared-weight 95 5 \
  --erniekit-dir /opt/ERNIE \
  --model "$MODEL" \
  --work-dir runs/vl16_vi_labeler_95_5 \
  --devices 0
```

Ví dụ trên tạo probability run `0.95/0.05`. Run train mới chỉ ghi metadata,
config, checkpoint và metric; JSONL/ảnh vẫn nằm tại hai prepared run nguồn.

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

### Kiến trúc planning và phạm vi thay đổi

Có ba lớp cần phân biệt để tránh nhầm giữa giao diện sử dụng và implementation:

1. **CLI và contract người dùng:** `--prepared-from`/`--prepared-weight` nằm
   trong `finetune_vl.py`. Single-run vẫn tương thích; multi-run yêu cầu weight
   dương, hữu hạn, đúng số lượng và tự normalize.
2. **Validation adapter:** `finetune_vl.py` vẫn validate summary, JSONL,
   image reference, prompt mask và target schema. Planner không bỏ qua các gate
   này và không tự suy luận lại target contract.
3. **Planning module:** `prepared_run_planning.py` tạo `PreparedRunPlan` bất
   biến và `PreparedRunPlanner` để hợp nhất các summary đã validate. Module này
   chỉ lập kế hoạch GPU-independent; nó không chạy ERNIEKit, không load model,
   không train và không đánh giá quality.

`PreparedRunPlan` giữ model identity, task/prompt union, source order,
provenance, sample counts, train/validation probabilities, normalized weights
và rejection metadata. Các source path vẫn trỏ về prepared run gốc; planner
không tạo bản sao `prepared/`. Metadata mở rộng trong summary legacy cũng được
giữ lại khi serialize.

`aggregate_prepared_runs()` và `load_prepared_runs()` trong `finetune_vl.py`
vẫn được giữ làm compatibility wrappers. Vì vậy đây là refactor nội bộ, không
phải một workflow CLI mới. Config cuối vẫn được tạo bởi
`create_resolved_config()` và đi vào backend ERNIEKit như trước.

Verification của phần planning:

```text
.venv-vl-eval/bin/python -m pytest -q tests/test_finetune_vl.py
67 passed, 1 skipped
```

Kết quả trên chỉ xác nhận contract/planning và regression tests. Nó không phải
là bằng chứng đã chạy full training GPU, native evaluation hoặc chứng minh
quality của model.

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
FlashAttention, full recompute, micro-batch 1, packing 1, accumulation 32,
sequence 2048, `50,176–451,584` pixel, LR `1e-4`, cosine, warmup 3%, weight
decay `0.01`, 3 epoch, hai worker và prefetch 2. Dùng
`--no-flash-attention` nếu build/hardware không tương thích.

OCR-VL tạo `IterableDataset` không có độ dài, nên ERNIEKit bắt buộc `max_steps`
dương. Script tự tính
`ceil(samples × epochs / (micro_batch × packing × accumulation × số GPU))`;
với micro-batch và packing bằng 1. `--smoke-steps` override giá trị này cho
smoke test.

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

Run tạo từ raw dataset có thêm `prepared/` và `rejected.jsonl`; run tạo bằng
`--prepared-from` chỉ tham chiếu các file của prepared run nguồn và không copy
chúng vào run mới:

```text
<work-dir>/
├── prepared/                         # chỉ có khi run này dùng --dataset-dir
│   ├── images/source-*/
│   ├── train-source-*.jsonl
│   └── validation-source-*.jsonl
├── rejected.jsonl                    # chỉ có khi prepare raw dataset
├── summary.json
├── resolved.yaml                     # resolved-resume*.yaml khi resume
├── adapter/
│   ├── checkpoint-*/                 # checkpoint ERNIEKit
│   └── export/                       # merged Hugging Face model
│       ├── merge_verification.json
│       └── logits_verification.json
├── export.yaml
├── export_manifest.json
├── logs/
├── metrics/                           # runtime, selection, CER, predictions
└── tensorboard_logs/
```

PaddleFormers 0.4.0 `MergeKit` chưa nhận diện model type custom `paddleocr_vl` và
có đường làm tròn BF16 khác runtime adapter. Pipeline vì vậy dùng API đã pin
`LoRAModel.merge()`, lọc LoRA tensors khỏi state dict rồi lưu lại ở định dạng
Hugging Face. Nó copy config, tokenizer, processor, custom model code,
`chat_template.jinja`, `inference.yml` và generation config từ base local.

Trước export cuối, pipeline merge/evaluate tuần tự adapter cuối và tối đa
`--eval-max-checkpoints` checkpoint gần nhất. Mỗi bản merge tạm được xác minh,
chấm native rồi xóa; base prediction chỉ chạy một lần và được tái sử dụng cho
mọi candidate. Checkpoint thắng được chọn theo CER, exact match và normalized
edit distance, sau đó mới tạo `adapter/export` chính thức. Export mới được build
và xác minh trong thư mục tạm; bản export cũ chỉ được thay sau khi toàn bộ gate
thành công và được khôi phục nếu bước promote lỗi.
`merge_verification.json` xác nhận công thức weight; `logits_verification.json`
xác nhận model reload khớp bản merge trong RAM và drift BF16 nằm trong numeric
tolerance. Native evaluator báo CER, exact match và normalized edit distance
tổng thể, theo nguồn và theo task. Mặc định merged model phải có NED >= 0,5,
CER <= 1,0, không regression so với base ở từng metric, và không prediction nào
chạm token limit khi chưa sinh EOS. Có thể chỉnh bằng
`--min-normalized-edit-distance`, `--max-cer`, `--eval-max-new-tokens` và
`--eval-task-max-new-tokens task=count`. Full run fail-fast nếu không checkpoint
nào đạt; smoke run vẫn chọn bản tốt nhất nhưng manifest giữ trạng thái failed.

Token budget prepare dùng đúng `smart_resize` của PaddleOCR-VL, grid ảnh
`(H/14)*(W/14)/2^2`, token prompt/response không special token và một EOS. Sample
vượt `--max-seq-len` bị reject trước khi JSONL được tạo; target không bao giờ bị
truncate. Nếu muốn đổi `--min-pixels`, `--max-pixels` hoặc `--max-seq-len`, phải
prepare lại từ raw dataset; đổi các cờ này chỉ ở bước train không khôi phục sample
đã bị loại trong prepared run.

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

## Recipe labeler-only: prepare → inspect → smoke → pilot

Ví dụ dưới đây chỉ dùng dataset do VL Layout Labeler tạo, không trộn các bộ OCR
lớn. Dataset export cần có `train`/`validation` và row-level task metadata.

```bash
export REPO=/home/tieubaoca/AI/ocr/paddle-ocr
export OUTPUT_ROOT=/media/tieubaoca/HDD1/F/finetune-output
export VL_MODEL=/home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6
export ERNIEKIT_DIR=$OUTPUT_ROOT/vl16_vi_experiment/runtime/erniekit
export LABELER_EXPORT=$OUTPUT_ROOT/vl_layout_experiment/export/vl
export PREPARED=$OUTPUT_ROOT/vl_layout_experiment/vl_labeler_prepare
cd "$REPO"
```

### 1. Prepare lại sau mỗi lần Export All thay đổi

```bash
python finetune_vl.py \
  --dataset-dir "$LABELER_EXPORT" \
  --model "$VL_MODEL" \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --work-dir "$PREPARED" \
  --prepare-only
```

Không cần prepare lại nếu export/manifest không đổi. Không xóa hoặc di chuyển
`$PREPARED` khi run train vẫn dùng `--prepared-from`.

### 2. Inspect decoder-only LoRA

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$PREPARED" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_inspect" \
  --inspect-model \
  --devices 0
```

Kiểm tra `metrics/trainable_parameters.json`; adapter không được có tensor
vision và vision encoder phải frozen.

*Lưu ý về cảnh báo khi inspect:* ERNIEKit v1.5 ở chế độ dry-run (`do_train=false`) sẽ ném log `AttributeError: 'FinetuningArguments' object has no attribute 'is_train_mm'`. Script đã bắt ngoại lệ này, ghi nhận cảnh báo và lưu đầy đủ thông số LoRA vào `metrics/trainable_parameters.json`.

### 3. Smoke ba bước

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$PREPARED" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_smoke" \
  --smoke-steps 3 \
  --gradient-accumulation-steps 32 \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --devices 0
```

Nếu FlashAttention lỗi, thêm `--no-flash-attention`. Không thay LoRA scope hoặc
mở vision training để chữa OOM.

### 4. Pilot 30 epoch

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$PREPARED" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_pilot" \
  --epochs 30 \
  --learning-rate 1e-4 \
  --lora-rank 32 \
  --gradient-accumulation-steps 32 \
  --save-steps 10 \
  --eval-samples-per-dataset 4 \
  --eval-max-checkpoints 3 \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --devices 0
```

Đây là overfit/feasibility pilot. Nếu validation chỉ có một table crop và không
có OCR crop, evaluator chỉ chứng minh pipeline hoạt động; không đủ để kết luận
model tốt hơn base.

### 5. Mẹo huấn luyện hiệu quả và xử lý các tình huống thực tế

#### Tránh mất sample vì token budget
- `finetune_vl.py` tính token budget ngay trong bước prepare từ prompt, target và visual tokens sau smart-resize. Sample vượt `--max-seq-len` bị reject với reason `token_budget_exceeded` và ghi chi tiết vào `rejected.jsonl`; ERNIEKit không truncate target.
- Nếu dataset có bảng hoặc OCR nhiều dòng, đặt `--max-pixels` và `--max-seq-len` ngay ở lệnh `--prepare-only`. Đổi hai cờ này chỉ ở lệnh train không làm sống lại sample đã bị reject; cần prepare lại từ raw dataset.
- Ví dụ profile nhiều layout: `--max-pixels 250880 --max-seq-len 4096`. Chỉ tăng `--max-seq-len` khi VRAM và độ dài target cho phép; tăng `--max-pixels` cũng làm tăng visual tokens và memory.

#### Xử lý khi Quality Gate so sánh với Base Model báo lỗi
- Pipeline mặc định so sánh CER/NED của model sau train với model gốc. Nếu tập validation có quá ít mẫu (ví dụ chỉ 1 crop bảng duy nhất), bất kỳ sự thay đổi nhỏ nào khiến CER tăng 0.04% cũng sẽ làm fail quality gate (`RuntimeError: No adapter checkpoint passed the native OCR quality gate`).
- Để xuất model trong các lượt chạy thử nghiệm / pilot:
  1. Thêm `--skip-evaluation` trực tiếp vào lệnh train.
  2. Hoặc nếu training đã hoàn tất lưu adapter tại `$WORK_DIR/adapter`, xuất model bằng script merge thủ công:
     ```bash
     python merge_paddleocr_vl_lora.py \
       --base-model "$VL_MODEL" \
       --adapter-dir "$WORK_DIR/adapter" \
       --output-dir "$WORK_DIR/adapter/export" \
       --fixture-jsonl "$FIXTURE_JSONL" \
       --min-pixels 50176 \
       --max-pixels 451584
     ```

## Tham số `finetune_vl.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--task {ocr,table,formula,chart}` | `ocr` | Task mặc định cho row thiếu cột `task`. Row-level task luôn được ưu tiên. |
| `--dataset-dir PATH [PATH ...]` | Một trong hai với `--prepared-from` | Raw dataset source; hỗ trợ nhiều source. |
| `--dataset-task TASK [TASK ...]` | Không có | Task mặc định cho từng source, cùng số lượng/thứ tự `--dataset-dir`. |
| `--prepared-from PATH [PATH ...]` | Một trong hai với `--dataset-dir` | Dùng lại JSONL/ảnh của một hoặc nhiều run prepare; cấm với prepare-only/resume. |
| `--prepared-weight WEIGHT [WEIGHT ...]` | `1.0` cho một run; bắt buộc khi nhiều run | Relative weight dương, hữu hạn, cùng số lượng/thứ tự prepared run; tự normalize. |
| `--erniekit-dir PATH` | Bắt buộc khi không prepare-only | ERNIEKit release/v1.5 checkout đã pin và có runtime Python. |
| `--model PATH_OR_ID` | `PaddlePaddle/PaddleOCR-VL-1.6` | Base model; inspect/train yêu cầu local snapshot hợp lệ. |
| `--work-dir PATH` | Timestamp run | Run output mới; resume dùng đúng run cũ. |
| `--prepare-only` | Tắt | Chỉ validate/stage; không cần ERNIEKit/GPU train nhưng vẫn load tokenizer để tính token budget. |
| `--smoke-steps INT` | Không có | Override `max_steps`; phải dương. |
| `--resume-from PATH` | Không có | Resume checkpoint thuộc `--work-dir`; không dùng với prepared-from. |
| `--inspect-model` | Tắt | Inspect trainable LoRA rồi dừng. |
| `--epochs FLOAT` | `3.0` | Số epoch logic dùng tính `max_steps`. |
| `--learning-rate FLOAT` | `1e-4` | Learning rate LoRA. |
| `--lora-rank INT` | `32` | Rank LoRA; alpha bằng `2 * rank`. |
| `--min-pixels INT` | `50176` | Smart-resize lower bound (`64 * 28 * 28`). |
| `--max-pixels INT` | `451584` | Smart-resize upper bound (`576 * 28 * 28`). |
| `--max-image-pixels INT` | `50_000_000` | Giới hạn ảnh nguồn. |
| `--max-seq-len INT` | `2048` | Tổng token budget; sample vượt bị reject, target không truncate. |
| `--gradient-accumulation-steps INT` | `32` | Micro-batch accumulation; batch/packing mặc định đều bằng 1. |
| `--validation-ratio FLOAT` | `0.02` | Tách validation cho raw source thiếu split, trong `(0, 0.5)`. |
| `--num-workers INT` | `2` | Dataloader workers. |
| `--prefetch-factor INT` | `2` | Prefetch mỗi worker. |
| `--seed INT` | `2026` | Seed prepare/sampling/training. |
| `--eval-samples-per-dataset INT` | `32` | Validation row tối đa mỗi source. |
| `--eval-max-new-tokens INT` | `1024` | Generation limit chung. |
| `--eval-task-max-new-tokens TASK=INT` | Không có, lặp được | Generation limit theo task. |
| `--eval-max-checkpoints INT` | `3` | Số checkpoint gần nhất cộng final adapter được chấm. |
| `--min-normalized-edit-distance FLOAT` | `0.5` | NED threshold trong `[0, 1]`. |
| `--max-cer FLOAT` | `1.0` | CER threshold không âm. |
| `--save-steps INT` | `100` | Chu kỳ save theo optimizer step. |
| `--skip-evaluation` | Tắt | Bỏ native evaluation/checkpoint selection; không dùng để claim quality. |
| `--devices VALUE` | Env hoặc `0` | CUDA device IDs comma-separated. |
| `--no-flash-attention` | Tắt | Tắt FlashAttention. |

Các tổ hợp bị cấm:

- `--dataset-dir` cùng `--prepared-from`;
- `--prepared-from` cùng `--prepare-only`;
- `--prepared-from` cùng `--resume-from`;
- `--prepared-weight` không có `--prepared-from`;
- nhiều prepared run không có weight, weight bằng `0`, âm, NaN hoặc sai số lượng;
- prepared run dùng base model khác nhau;
- `--dataset-task` không có `--dataset-dir` hoặc số task khác số source.

## Chạy evaluator độc lập

```bash
.venv-vl-eval/bin/python evaluate_paddleocr_vl.py \
  --base-model "$VL_MODEL" \
  --merged-model "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_pilot/adapter/export" \
  --validation-jsonl "$PREPARED/prepared/validation-source-000.jsonl" \
  --output-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_pilot/manual_eval" \
  --samples-per-dataset 8 \
  --max-new-tokens 1024 \
  --task-max-new-tokens table=2048 \
  --min-normalized-edit-distance 0.5 \
  --max-cer 1.0
```

### Tham số `evaluate_paddleocr_vl.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--base-model PATH` | Bắt buộc | Base snapshot. |
| `--merged-model PATH` | Bắt buộc | Merged HF model. |
| `--validation-jsonl PATH [PATH ...]` | Bắt buộc | Validation JSONL sources. |
| `--output-dir PATH` | Bắt buộc | Metrics output. |
| `--samples-per-dataset INT` | `32` | Row tối đa mỗi source. |
| `--max-new-tokens INT` | `1024` | Generation limit chung. |
| `--task-max-new-tokens TASK=INT` | Không có, lặp được | Limit riêng task. |
| `--min-normalized-edit-distance FLOAT` | `0.5` | NED quality threshold. |
| `--max-cer FLOAT` | `1.0` | CER quality threshold. |
| `--base-predictions-jsonl PATH` | Không có | Tái sử dụng base predictions. |
| `--report-only` | Tắt | Ghi report nhưng không fail exit code; chỉ dùng smoke/screening. |

## Merge adapter độc lập

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python merge_paddleocr_vl_lora.py \
  --base-model "$VL_MODEL" \
  --adapter-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_pilot/adapter" \
  --output-dir "$OUTPUT_ROOT/vl16_vi_experiment/labeler_only_pilot/manual_merge" \
  --fixture-jsonl "$PREPARED/prepared/validation-source-000.jsonl" \
  --min-pixels 50176 \
  --max-pixels 250880
```

### Tham số `merge_paddleocr_vl_lora.py`

| Argument | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `--base-model PATH` | Có | Base snapshot có `model_type=paddleocr_vl`. |
| `--adapter-dir PATH` | Có | LoRA adapter directory. |
| `--output-dir PATH` | Có | Output mới ngoài base; không chứa safetensors cũ. |
| `--fixture-jsonl PATH` | Có | Fixture cho weight/logit verification. |
| `--min-pixels INT` | Có | Min resize giống train. |
| `--max-pixels INT` | Có | Max resize giống train. |

Merge thành công phải có `model.safetensors`, `merge_verification.json` và
`logits_verification.json`, cả hai report có status `passed`.
