# Fine-tune PaddleOCR tiếng Việt

Pipeline LoRA cho **PaddleOCR-VL-1.6** được tài liệu riêng tại
[docs/finetune-paddleocr-vl-1.6.md](docs/finetune-paddleocr-vl-1.6.md). Pipeline
này dùng ERNIEKit và môi trường tách biệt với PP-OCRv6 bên dưới. Dataset VL đã
prepare có thể dùng lại bằng `finetune_vl.py --prepared-from <prepare-run>` mà
không sao chép hoặc xử lý lại ảnh. Có thể hợp nhất nhiều prepared run trực tiếp
bằng cùng option và relative weight, ví dụ `--prepared-weight 95 5`.
OCR/table/formula/chart dùng chung
`finetune_vl.py`; trộn nhiều layout trong một run qua cột `task` trong dataset
để hạn chế phá các khả năng layout khác.

### Phân biệt CLI và kiến trúc prepared-run

`--prepared-from` và `--prepared-weight` là CLI dành cho người dùng. Phần
planning nội bộ tương ứng được đặt trong `prepared_run_planning.py`: module này
đọc các prepared summary đã được validate, tạo `PreparedRunPlan` bất biến và
ghi `summary.json` cho run mới. `finetune_vl.py` vẫn giữ các hàm tương thích và
tiếp tục sở hữu validation target/prompt; refactor này không tạo thêm CLI và
không thay đổi backend ERNIEKit.

Luồng thực tế là:

1. `finetune_vl.py` validate summary, JSONL, prompt/target contract và ảnh.
2. `PreparedRunPlanner` hợp nhất một hoặc nhiều run, kiểm tra model, union
   task, normalize weight và giữ provenance/source path.
3. `create_resolved_config()` chuyển summary đã lập kế hoạch thành config
   ERNIEKit; JSONL và ảnh vẫn được tham chiếu tại run nguồn, không được copy.

Đã kiểm chứng bằng `.venv-vl-eval`: `67 passed, 1 skipped` trong
`tests/test_finetune_vl.py`. Test còn lại bị skip là kiểm thử phụ thuộc môi
trường; không dùng kết quả này để khẳng định training GPU hoặc quality model.

Với nhiều dataset VL thiếu cột `task`, phải khai báo `--dataset-task` theo đúng
thứ tự source. Table target phải là OTSL, formula là LaTeX, chart là Markdown;
OCR giữ nguyên newline. Pipeline tính image token theo `smart_resize` thật và
không truncate ground truth.

Script này fine-tune **text recognition** (ảnh crop chứa một từ/dòng chữ), không phải text detection trên ảnh trang đầy đủ. Nó nhận nhiều Hugging Face dataset đã `save_to_disk()` hoặc snapshot tải từ Hub có `data/*.parquet`, mỗi sample có:

- `image`: `datasets.Image`, PIL image, bytes/path dictionary, hoặc đường dẫn ảnh;
- `label` hoặc `text`: chuỗi ground truth.

Các dataset được lọc riêng, chia validation theo từng nguồn nếu chưa có split `validation`/`valid`/`dev`, sau đó trộn có seed. Ảnh hợp lệ được materialize thành PNG lossless để training không phụ thuộc cache gốc.

Tên file Parquet chuẩn như `train-00000-of-00003.parquet`, `test-00000-of-00001-<hash>.parquet` được tự động gom thành split tương ứng. Split `test` không được dùng làm validation; nếu dataset không có `validation`/`valid`/`dev`, script tách validation từ `train` theo `--validation-ratio`.

## Cài đặt

Python 3.10-3.12 được khuyến nghị. Với RTX 5060 Ti, dùng wheel PaddlePaddle CUDA 12.6/12.9 phù hợp với driver đang cài; ví dụ CUDA 12.9:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install paddlepaddle-gpu==3.3.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
python -m pip install -r PaddleOCR/requirements.txt
python -m pip install -r requirements.txt
python -c "import paddle; paddle.utils.run_check()"
```

Nếu driver chỉ phù hợp CUDA 12.6, đổi index thành `.../cu126/`. Không cài đồng thời `paddlepaddle` CPU và `paddlepaddle-gpu`.

Tải thủ công training checkpoint chính thức cho cả recognition và detection:

```bash
./download_pretrained_models.sh all
```

Có thể chỉ tải một model bằng `./download_pretrained_models.sh rec` hoặc
`./download_pretrained_models.sh det`. Model được lưu mặc định trong `models/`;
script bỏ qua file đã tồn tại và chỉ đổi tên file `.part` sau khi tải thành công.

## Chạy

```bash
python finetune.py \
  --dataset-dir /data/ocr_a /data/ocr_b /data/ocr_c \
  --paddleocr-dir ./PaddleOCR \
  --work-dir runs/vietnamese_v1
```

Chỉ chuẩn bị/lọc dữ liệu và sinh config, chưa tải weight/chạy GPU:

```bash
python finetune.py \
  --dataset-dir /data/ocr_a /data/ocr_b \
  --paddleocr-dir ./PaddleOCR \
  --work-dir runs/check_data \
  --prepare-only
```

Mặc định được cân bằng cho chất lượng và RTX 5060 Ti 16 GB:

- backbone `PP-OCRv6_medium_rec` pretrained và dictionary NFC riêng có đầy đủ
  chữ hoa/thường tiếng Việt, chữ Latin, số và dấu câu thường dùng;
- ảnh train đa tỉ lệ cao 32/48/64, rộng 640 để giữ chi tiết dòng dài;
- batch 32, AMP dynamic loss scaling, 6 data workers;
- cosine learning rate `3e-4`, 50 epoch, validation mỗi 1000 step;
- tối đa 80 ký tự/label và 50 triệu pixel/ảnh.

Nếu CUDA OOM, giảm theo thứ tự:

```bash
python finetune.py ... --batch-size 24
python finetune.py ... --batch-size 16 --image-width 480
```

Không giảm `--image-width` nếu phần lớn label dài; giảm batch trước. Nếu label dài hơn 80 ký tự là dữ liệu dòng dài hợp lệ, tăng đồng thời `--max-text-length` và cân nhắc `--image-width 768` với batch nhỏ hơn.

PaddlePaddle 3.3.0 dùng loss scale dạng tensor một phần tử. NumPy 2.4 trở lên
không còn cho phép chuyển mảng một chiều này trực tiếp thành scalar, khiến
dynamic GradScaler có thể crash ngay khi gặp overflow. Project pin
`numpy<2.4`; chạy lại `python -m pip install -r requirements.txt` nếu môi
trường bị nâng lên NumPy 2.4+.

## Output và kiểm tra chất lượng dữ liệu

Mỗi run tạo mới, không ghi đè thư mục đã có dữ liệu:

```text
runs/vietnamese_v1/
├── prepared/
│   ├── images/
│   ├── train.txt
│   ├── validation.txt
│   ├── rejected.jsonl
│   └── summary.json
├── pretrained/
├── resolved_config.yml
└── output/
```

`rejected.jsonl` ghi dataset, split, row và lý do bị loại: text rỗng/quá dài, ký tự ngoài dictionary, ảnh hỏng/kích thước sai, hoặc lỗi đọc row. Script không truncate label vì làm vậy sẽ tạo ground truth sai. Có thể truyền dictionary riêng bằng `--character-dict /path/to/dict.txt` (mỗi dòng một ký tự).

PP-OCRv6 gốc không có đầy đủ các code point tiếng Việt NFC trong dictionary.
Vì dictionary của run này thay đổi số lớp output, PaddleOCR sẽ nạp các layer
pretrained có shape tương thích (backbone/neck) và khởi tạo lại head CTC/NRTR.
Do đó nên có ít nhất vài nghìn ảnh crop chất lượng tốt và giữ validation riêng.

Chạy test:

```bash
python -m unittest discover -s tests -v
```

## Dịch vụ OCR và gán nhãn trên trình duyệt

Dịch vụ local dùng `PP-OCRv6_medium_det` để tìm vùng chữ và model recognition
đã fine-tune để nhận dạng tiếng Việt. Giao diện cho phép sửa text, polygon bốn
điểm, thứ tự đọc, undo/redo, OCR tuần tự cả folder và xuất JSONL.

Cài dependency web vào đúng môi trường hiện có rồi khởi chạy:

```bash
source .venv/bin/activate
python -m pip install -r requirements-labeler.txt
python run_labeler.py \
  --images /home/tieubaoca/Documents/ocr-md/images \
  --device gpu:0
```

Mở `http://127.0.0.1:8010`. Mặc định dịch vụ dùng detector
`PP-OCRv6_medium_det` từ cache chính thức của PaddleX và recognition model tại
`runs/vi_rec_3datasets_v1/inference/best_accuracy`. Chỉ các file ảnh nằm trực
tiếp trong folder được quét; thư mục con không được quét đệ quy và ảnh nguồn
không bị thay đổi.

Annotation được autosave nguyên tử theo từng ảnh vào:

```text
<folder ảnh>/.paddleocr-labeler/annotations/
```

Lệnh `Xuất JSONL` tạo
`<folder ảnh>/.paddleocr-labeler/manifest.jsonl`. OCR toàn folder chạy tuần tự,
bỏ qua ảnh đã có sidecar hợp lệ; vì vậy sau khi dừng hoặc khởi động lại service,
chạy batch lần nữa sẽ tiếp tục từ các ảnh chưa lưu.

Có thể thay model hoặc chạy CPU rõ ràng:

```bash
python run_labeler.py \
  --images /path/to/images \
  --det-model-dir /path/to/PP-OCRv6_medium_det \
  --rec-model-dir /path/to/recognition/inference \
  --device cpu
```

## Dịch vụ gán nhãn OCR detection

Hướng dẫn train detector từ output của tool: [docs/finetune-ppocrv6-det.md](docs/finetune-ppocrv6-det.md).

Chế độ này chỉ nạp `PP-OCRv6_medium_det`, không nạp recognition model, nên nhẹ
hơn service OCR đầy đủ. Khởi chạy trên RTX 5060 Ti:

```bash
source .venv/bin/activate
python run_det_labeler.py \
  --images /home/tieubaoca/Documents/ocr-md/images \
  --device gpu:0
```

Mở `http://127.0.0.1:8011`. Model mặc định là `PP-OCRv6_medium_det`; nếu model
đã nằm trong cache PaddleX thì không tải lại. Cũng có thể chỉ định model local:

```bash
python run_det_labeler.py \
  --images /path/to/images \
  --det-model-dir /home/tieubaoca/.paddlex/official_models/PP-OCRv6_medium_det \
  --device gpu:0
```

Quy trình gán nhãn:

1. Chọn ảnh rồi bấm `Detect ảnh này`, hoặc chạy `Detect toàn folder` để prelabel.
2. Bấm `Vẽ vùng` hoặc phím `A`, rồi kéo chuột để thêm bbox.
3. Chọn bbox để kéo cả vùng; kéo một trong bốn tay nắm để chỉnh polygon.
4. Xóa bằng nút `Xóa vùng đã chọn` hoặc phím `Delete`. Đánh dấu `Bỏ qua vùng này`
   để xuất transcription `###` cho vùng không dùng khi train.
5. Bấm `Đánh dấu hoàn tất` sau khi kiểm tra và `Xuất nhãn detection`.

Dữ liệu detection được tách khỏi annotation recognition:

```text
<folder ảnh>/.paddleocr-det-labeler/annotations/*.json
<folder ảnh>/.paddleocr-det-labeler/det_labels.txt
<folder ảnh>/.paddleocr-det-labeler/manifest.jsonl
```

`det_labels.txt` dùng trực tiếp cho `label_file_list` của PaddleOCR detection;
mỗi dòng có dạng `relative_image_path<TAB>[{"transcription":"text","points":[...]}]`.
`manifest.jsonl` giữ thêm score, source, revision và metadata để audit/tiếp tục sửa.

Mặc định `--det-limit-side-len 1600` ưu tiên bắt chữ nhỏ trên tài liệu scan và
phù hợp GPU 16 GB. Nếu muốn nhanh/nhẹ hơn, dùng `1280` hoặc `960`. Có thể giảm
`--det-box-thresh` từ `0.6` xuống `0.5` để tăng recall, nhưng sẽ phải xóa nhiều
false positive hơn.

Không tăng số Uvicorn worker: launcher luôn dùng đúng một worker để chỉ nạp một
pipeline detection/recognition lên GPU và để hàng đợi OCR tuần tự không bị nhân
bản. Nếu model, device hoặc folder startup không hợp lệ, service dừng với lỗi
thay vì âm thầm chuyển sang CPU.

Vì đây là công cụ local không có cơ chế đăng nhập, `--host` chỉ chấp nhận
`localhost` hoặc địa chỉ loopback IPv4/IPv6 (ví dụ `127.0.0.1`, `::1`);
`0.0.0.0` và địa chỉ LAN/public bị từ chối.

## Dịch vụ gán nhãn layout cho PaddleOCR-VL 1.6

`run_vl_layout_labeler.py` là service riêng cho pipeline train-oriented:
`PP-DocLayoutV3` local phát hiện layout, người dùng kiểm tra/sửa bbox và task,
`llama-server` prelabel text theo crop, rồi export Hugging Face `DatasetDict`
`train`/`validation` có `image`, `text`, `task`, `source_page_id`. Tool split
theo trang trước khi tạo crop để không rò cùng một page qua hai split, không
sửa ảnh nguồn và không dùng sidecar của hai labeler phía trên.

```bash
source .venv/bin/activate
python -m pip install -r requirements-labeler.txt
python run_vl_layout_labeler.py \
  --images /path/to/images \
  --layout-model-dir /home/tieubaoca/.paddlex/official_models/PP-DocLayoutV3 \
  --vl-base-url http://127.0.0.1:8000/v1 \
  --vl-model paddleocr-vl \
  --device gpu:0
```

Mở `http://127.0.0.1:8012`. Service fail-fast nếu layout model hoặc endpoint
VL không sẵn sàng, chỉ dùng một GPU queue và chỉ bind loopback. Sidecar riêng
được lưu tại `<folder ảnh>/.paddleocr-vl-labeler/annotations/`. Sidecar version
2 giữ đủ 25 class theo đúng thứ tự PP-DocLayoutV3. `table`, `chart`,
`display_formula`/`inline_formula` được map lần lượt sang `table`, `chart`,
`formula`; các class văn bản map sang `ocr`, còn class layout-only giữ bbox với
`task=null` và không gửi tới VL.

Quy trình: `Detect ảnh` hoặc `Detect folder` → chọn block → `Prelabel chọn`/
`Prelabel ảnh` → sửa layout label/task/text/bbox → `Complete`. `Export HF` chỉ
lấy block completed, không skip, có task VL và target đúng schema chính thức.
Output prelabel mở mặc định bằng editor trực quan: OCR theo dòng, table bằng
bảng HTML có merge/split ô, formula bằng LaTeX/preview và chart bằng bảng Markdown. Tab `Raw`
luôn cho phép sửa trực tiếp; raw lỗi không bị ghi đè, còn `Complete` bị chặn cho
đến khi raw và editor ánh xạ được sang cùng một target hợp lệ.
`Export Layout` tạo COCO instance segmentation từ toàn trang; `Export All` tạo nguyên tử hai
nhánh `<output>/vl/` và `<output>/layout/`. Export VL cần tối thiểu hai trang
hợp lệ để tạo train/validation không leakage.

```bash
python finetune_vl.py --prepare-only \
  --dataset-dir /path/to/export/vl \
  --model /path/to/PaddleOCR-VL-1.6 \
  --work-dir runs/vl16_layout_prepare
```

Kiểm tra và train nhánh layout bằng cấu hình PaddleX cài cùng môi trường:

```bash
PADDLEX_CONFIG=.venv/lib/python3.12/site-packages/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml

.venv/bin/python -c 'from paddlex.engine import Engine; Engine().run()' \
  -c "$PADDLEX_CONFIG" \
  -o Global.mode=check_dataset \
  -o Global.dataset_dir=/path/to/export/layout

.venv/bin/python -c 'from paddlex.engine import Engine; Engine().run()' \
  -c "$PADDLEX_CONFIG" \
  -o Global.mode=train \
  -o Global.dataset_dir=/path/to/export/layout \
  -o Train.num_classes=25
```

## Hướng dẫn đầy đủ các script fine-tune

Repository có ba pipeline huấn luyện độc lập. Không dùng lẫn môi trường, model
weight hoặc format dataset giữa các pipeline:

| Script | Bài toán | Dataset đầu vào | Backend | Artifact chính |
| --- | --- | --- | --- | --- |
| `finetune.py` | PP-OCRv6 text recognition trên ảnh crop | Hugging Face `save_to_disk()` hoặc Parquet local | `PaddleOCR/tools/train.py` | checkpoint trong `output/` |
| `finetune_det.py` | PP-OCRv6 text detection trên ảnh trang | detection labeler workspace hoặc `det_labels.txt` | `PaddleOCR/tools/train.py` | `output/`, tùy chọn `inference/best_accuracy/` |
| `finetune_vl.py` | PaddleOCR-VL-1.6 OCR/table/formula/chart | export VL của layout labeler hoặc prepared run | ERNIEKit OCR-VL-SFT + LoRA | adapter và HF merged model trong `adapter/export/` |

Hai utility của pipeline VL được trainer gọi tự động nhưng cũng có thể chạy độc lập:

| Script | Mục đích |
| --- | --- |
| `evaluate_paddleocr_vl.py` | So sánh deterministic base/merged, tính CER, exact match, normalized edit distance và quality gate |
| `merge_paddleocr_vl_lora.py` | Merge adapter ERNIEKit vào snapshot HF, kiểm tra weight merge và reload logits |

`finetune_vl_layout.py` không tồn tại trong checkout này. Nhánh layout toàn trang
được train bằng PaddleX theo lệnh ở phần
[Dịch vụ gán nhãn layout](#dịch-vụ-gán-nhãn-layout-cho-paddleocr-vl-16).

### Quy tắc chung

1. Chạy `--prepare-only` trước hoặc kiểm tra `summary.json`/`rejected.jsonl` của
   prepared run hiện có.
2. Dùng một `--work-dir` mới cho mỗi thí nghiệm. Không đặt output trong thư mục
   snapshot base model.
3. Không dùng inference directory làm training checkpoint `.pdparams`.
4. Ghi lại command, git SHA, model path/hash, dataset manifest/hash, peak VRAM,
   checkpoint/export path và metric.
5. Dataset nhỏ chỉ chứng minh pipeline có thể chạy/overfit; không suy diễn chất
   lượng tổng quát từ train loss hoặc một validation sample.

Các biến đường dẫn mẫu:

```bash
export REPO=/home/tieubaoca/AI/ocr/paddle-ocr
export PADDLEOCR_DIR=$REPO/PaddleOCR
export OUTPUT_ROOT=/media/tieubaoca/HDD1/F/finetune-output
export VL_MODEL=/home/tieubaoca/AI/models/paddleocr-cache/official_models/PaddleOCR-VL-1.6
export ERNIEKIT_DIR=$OUTPUT_ROOT/vl16_vi_experiment/runtime/erniekit
cd "$REPO"
```

### A. PP-OCRv6 recognition — `finetune.py`

#### Dataset contract

`--dataset-dir` nhận một hoặc nhiều dataset Hugging Face đã `save_to_disk()`
hoặc snapshot local chứa `data/*.parquet`. Mỗi row cần:

- `image`: `datasets.Image`, PIL image, bytes/path dictionary hoặc path ảnh;
- `label` hoặc `text`: ground truth string.

Nếu source có split `validation`, `valid` hoặc `dev`, script dùng split đó. Nếu
không có, script tách validation từ train theo từng source bằng
`--validation-ratio`. Split `test` không tự động được dùng làm validation.
Ảnh hợp lệ được materialize thành PNG lossless; sample lỗi được ghi vào
`prepared/rejected.jsonl`, không truncate hoặc sửa target âm thầm.

#### Bước 1: prepare-only

```bash
python finetune.py \
  --dataset-dir /data/ocr_a /data/ocr_b \
  --paddleocr-dir "$PADDLEOCR_DIR" \
  --work-dir "$OUTPUT_ROOT/rec_prepare" \
  --validation-ratio 0.02 \
  --seed 2026 \
  --prepare-only
```

Kiểm tra:

```text
$OUTPUT_ROOT/rec_prepare/prepared/summary.json
$OUTPUT_ROOT/rec_prepare/prepared/rejected.jsonl
$OUTPUT_ROOT/rec_prepare/prepared/train.txt
$OUTPUT_ROOT/rec_prepare/prepared/validation.txt
$OUTPUT_ROOT/rec_prepare/resolved_config.yml
```

#### Bước 2: train

```bash
python finetune.py \
  --dataset-dir /data/ocr_a /data/ocr_b \
  --paddleocr-dir "$PADDLEOCR_DIR" \
  --work-dir "$OUTPUT_ROOT/rec_v1" \
  --pretrained-model "$REPO/models/PP-OCRv6_medium_rec_pretrained.pdparams" \
  --epochs 50 \
  --learning-rate 3e-4 \
  --batch-size 32 \
  --num-workers 6
```

Nếu `--pretrained-model` là URL, script tải vào `<work-dir>/pretrained/`. Nếu là
path local, file phải tồn tại. Nếu OOM, giảm `--batch-size` trước; chỉ giảm
`--image-width` khi dữ liệu không có nhiều dòng dài.

#### Toàn bộ args của `finetune.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--dataset-dir PATH [PATH ...]` | Bắt buộc | Một hoặc nhiều dataset recognition để prepare và trộn. |
| `--paddleocr-dir PATH` | Bắt buộc | Checkout PaddleOCR chứa `tools/train.py`. |
| `--work-dir PATH` | `runs/vi_rec_YYYYmmdd_HHMMSS` | Thư mục run mới. |
| `--config VALUE` | `configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml` | Config tương đối với PaddleOCR checkout hoặc absolute path. |
| `--pretrained-model VALUE` | URL PP-OCRv6 medium rec | URL hoặc file training weight local. |
| `--validation-ratio FLOAT` | `0.02` | Tỷ lệ tách validation cho source thiếu validation; phải trong `(0, 0.5)`. |
| `--seed INT` | `2026` | Seed split và shuffle. |
| `--epochs INT` | `50` | Số epoch. |
| `--learning-rate FLOAT` | `3e-4` | Learning rate recognition. |
| `--batch-size INT` | `32` | Batch train trên mỗi card; eval batch được đặt khoảng hai lần giá trị này. |
| `--num-workers INT` | `6` | Data loader workers; eval dùng ít nhất một worker. |
| `--image-width INT` | `640` | Chiều rộng crop sau resize; tăng cho dòng dài. |
| `--max-text-length INT` | `80` | Độ dài target tối đa; row dài hơn bị reject. |
| `--max-image-pixels INT` | `50_000_000` | Giới hạn pixel ảnh nguồn trước decode/staging. |
| `--character-dict PATH` | `vietnamese_dict.txt` | Dictionary một ký tự mỗi dòng. |
| `--prepare-only` | Tắt | Chỉ prepare/filter/config; không tải weight và không train. |

Output:

```text
<work-dir>/
├── prepared/images/
├── prepared/train.txt
├── prepared/validation.txt
├── prepared/rejected.jsonl
├── prepared/summary.json
├── pretrained/
├── resolved_config.yml
└── output/
```

### B. PP-OCRv6 detection — `finetune_det.py`

Tài liệu chuyên sâu: [docs/finetune-ppocrv6-det.md](docs/finetune-ppocrv6-det.md).

#### Dataset contract

`--dataset-dir` nhận một hoặc nhiều:

- workspace ảnh có `.paddleocr-det-labeler/det_labels.txt`;
- chính thư mục `.paddleocr-det-labeler`;
- path trực tiếp tới `det_labels.txt`.

Mỗi dòng label:

```text
relative/image.png<TAB>[{"transcription":"text","points":[[x1,y1],...]}]
```

Script kiểm tra path, ảnh, JSON, polygon, bounds và diện tích. Ảnh trùng SHA-256
được loại trước khi split để tránh leakage.

#### Bước 1: prepare-only

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir "$PADDLEOCR_DIR" \
  --work-dir "$OUTPUT_ROOT/det_prepare" \
  --validation-ratio 0.10 \
  --seed 2026 \
  --prepare-only
```

#### Bước 2: train và export

```bash
python finetune_det.py \
  --dataset-dir /data/pages_a /data/pages_b \
  --paddleocr-dir "$PADDLEOCR_DIR" \
  --work-dir "$OUTPUT_ROOT/det_v1" \
  --pretrained-model "$REPO/models/PP-OCRv6_medium_det_pretrained.pdparams" \
  --epochs 100 \
  --batch-size 4 \
  --export-after-train
```

Script từ chối inference directory và so shape toàn bộ pretrained tensor với
PP-OCRv6 detector trước khi train.

#### Toàn bộ args của `finetune_det.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--dataset-dir PATH [PATH ...]` | Bắt buộc | Workspace, labeler dir hoặc `det_labels.txt`; hỗ trợ nhiều source. |
| `--paddleocr-dir PATH` | `./PaddleOCR` | Checkout PaddleOCR. |
| `--work-dir PATH` | `runs/vi_det_YYYYmmdd_HHMMSS` | Thư mục run mới. |
| `--config VALUE` | `configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml` | Config detection gốc; script giữ architecture/loss/transform chain. |
| `--pretrained-model VALUE` | URL PP-OCRv6 medium det | URL hoặc file `.pdparams`; không nhận inference directory. |
| `--validation-ratio FLOAT` | `0.10` | Tỷ lệ validation; phải trong `(0, 0.5)`. |
| `--seed INT` | `2026` | Seed split và shuffle. |
| `--epochs INT` | `100` | Số epoch. |
| `--learning-rate FLOAT` | `1e-4` | Learning rate detection. |
| `--batch-size INT` | `4` | Batch mỗi card; giảm xuống `3`, rồi `2` khi OOM. |
| `--num-workers INT` | `4` | Data loader workers. |
| `--eval-batch-step INT` | `200` | Chu kỳ evaluation theo step. |
| `--save-epoch-step INT` | `5` | Chu kỳ lưu checkpoint theo epoch. |
| `--max-image-pixels INT` | `50_000_000` | Giới hạn pixel ảnh nguồn. |
| `--min-polygon-area FLOAT` | `4.0` | Diện tích polygon tối thiểu; box nhỏ hơn bị reject. |
| `--disable-amp` | Tắt | Tắt AMP; mặc định AMP bật. |
| `--prepare-only` | Tắt | Chỉ validate/stage/config. |
| `--export-after-train` | Tắt | Export best checkpoint sang `inference/best_accuracy`. |

Output:

```text
<work-dir>/
├── prepared/images/
├── prepared/train.txt
├── prepared/validation.txt
├── prepared/rejected.jsonl
├── prepared/summary.json
├── pretrained/
├── resolved_config.yml
├── output/
└── inference/best_accuracy/    # khi có --export-after-train
```

### C. PaddleOCR-VL-1.6 LoRA — `finetune_vl.py`

Tài liệu cài đặt/runtime chuyên sâu:
[docs/finetune-paddleocr-vl-1.6.md](docs/finetune-paddleocr-vl-1.6.md).
Pipeline này dùng ERNIEKit release/v1.5, không dùng `PaddleOCR/tools/train.py`.

#### Cài đặt ERNIEKit runtime chuẩn

Pipeline kiểm tra nghiêm ngặt git commit và version runtime của ERNIEKit để đảm bảo tương thích LoRA decoder-only. Cài đặt vào thư mục riêng:

```bash
# 1. Clone ERNIEKit và checkout đúng commit đã pin (branch release/v1.5)
git clone https://github.com/PaddlePaddle/ERNIE.git erniekit
cd erniekit
git checkout 790a50b045d1aca2753d5395d8bec0806b2e6925

# 2. Tạo virtualenv Python 3.10-3.12 (khuyến nghị 3.11 hoặc 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 3. Cài đặt các gói đúng phiên bản đã pin trong requirements-vl-erniekit.txt
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/  # hoặc cu126
pip install paddleformers==0.4.0 safetensors==0.7.0 transformers==4.55.4 ml_dtypes==0.5.4
pip install -e .
```

Thư mục virtualenv này sẽ là `$ERNIEKIT_DIR/.venv`.

#### Dataset và task contract

Dataset labeler chuẩn có split `train`/`validation` và các cột `image`, `text`,
`task`, `source_page_id`.

| `task` | Prompt | Target |
| --- | --- | --- |
| `ocr` | `OCR:` | Text OCR, giữ newline |
| `table` | `Table Recognition:` | OTSL canonical, không phải HTML |
| `formula` | `Formula Recognition:` | LaTeX |
| `chart` | `Chart Recognition:` | Markdown table |

Cột `task` trên row được ưu tiên. Row thiếu task dùng `--task`. Với nhiều source
thiếu task, truyền một `--dataset-task` cho mỗi `--dataset-dir` theo đúng thứ tự.
Không dùng `--dataset-dir` và `--prepared-from` cùng lúc.

#### Bước 1: prepare-only

```bash
python finetune_vl.py \
  --dataset-dir /path/to/export/vl \
  --model "$VL_MODEL" \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --work-dir "$OUTPUT_ROOT/vl_prepare" \
  --prepare-only
```

Prepare-only không cần ERNIEKit hoặc GPU train; script vẫn load tokenizer từ `--model` để tính token budget. Giữ nguyên prepared run vì
`--prepared-from` tham chiếu trực tiếp JSONL/ảnh, không copy lại. Khi cần trộn
prepared corpus cũ với export labeler mới, truyền nhiều run và weight tương ứng:

```bash
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$OUTPUT_ROOT/vl_prepare" "$OUTPUT_ROOT/vl_labeler_prepare" \
  --prepared-weight 95 5 \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl_mixed_95_5" \
  --devices 0
```

Weight được normalize thành `0.95/0.05`, áp dụng giống nhau cho train và
validation, đồng thời giữ probability nội bộ của từng prepared run. Các run phải
dùng cùng base model; task được union và source giữ đúng thứ tự input.

#### Bước 2: inspect LoRA scope

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$OUTPUT_ROOT/vl_prepare" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl_inspect" \
  --inspect-model \
  --devices 0
```

Inspect phải xác nhận LoRA chỉ ở text decoder, vision frozen và không có adapter
tensor thuộc vision encoder.

*Lưu ý:* Khi chạy `--inspect-model`, nếu thấy log `AttributeError: 'FinetuningArguments' object has no attribute 'is_train_mm'` và cảnh báo của script, đây là hành vi dry-run bình thường của ERNIEKit v1.5; script đã bắt exception và lưu `metrics/trainable_parameters.json` thành công.

#### Bước 3: smoke test

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$OUTPUT_ROOT/vl_prepare" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl_smoke" \
  --smoke-steps 3 \
  --gradient-accumulation-steps 32 \
  --max-pixels 250880 \
  --max-seq-len 4096 \
  --devices 0
```

Nếu FlashAttention không tương thích, thêm `--no-flash-attention`. Smoke phải tạo
adapter/checkpoint, merged model, `merge_verification.json`,
`logits_verification.json` và evaluator output.

#### Bước 4: pilot labeler-only

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --prepared-from "$OUTPUT_ROOT/vl_prepare" \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl_labeler_pilot" \
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

Với chỉ `35` train samples, đây là feasibility/overfit pilot. Cần validation có
nhiều OCR và table crop trước khi dùng metric để kết luận chất lượng.

#### Mẹo huấn luyện hiệu quả và xử lý lỗi thường gặp

1. **Tránh mất sample vì token budget**:
   - `finetune_vl.py` tính token budget ngay trong bước prepare. Sample vượt `--max-seq-len` bị reject với reason `token_budget_exceeded` và ghi vào `rejected.jsonl`; target không bị truncate.
   - Nếu dataset có bảng hoặc OCR nhiều dòng, truyền `--max-pixels 250880 --max-seq-len 4096` ngay ở lệnh `--prepare-only`. Đổi các cờ này sau đó ở lệnh train không khôi phục sample đã bị reject; cần prepare lại từ raw dataset.

2. **Xử lý Quality Gate khi tập validation nhỏ**:
   - Nếu tập validation chỉ có 1–2 mẫu, chỉ cần lệch 1 ký tự là CER tăng nhẹ và Quality Gate (`no_regression_vs_base: true`) sẽ chặn tạo model merge (`RuntimeError: No adapter checkpoint passed the native OCR quality gate`).
   - **Giải pháp A:** Thêm cờ `--skip-evaluation` vào lệnh `finetune_vl.py` khi train thử nghiệm/pilot.
   - **Giải pháp B:** Nếu đã train xong adapter tại `$WORK_DIR/adapter`, dùng script merge trực tiếp:
     ```bash
     python merge_paddleocr_vl_lora.py \
       --base-model "$VL_MODEL" \
       --adapter-dir "$WORK_DIR/adapter" \
       --output-dir "$WORK_DIR/export" \
       --fixture-jsonl "$FIXTURE" \
       --min-pixels 50176 \
       --max-pixels 451584
     ```

#### Bước 5: resume

Resume dùng checkpoint thuộc chính `--work-dir`; không truyền `--prepared-from`:

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python finetune_vl.py \
  --erniekit-dir "$ERNIEKIT_DIR" \
  --model "$VL_MODEL" \
  --work-dir "$OUTPUT_ROOT/vl_labeler_pilot" \
  --resume-from "$OUTPUT_ROOT/vl_labeler_pilot/adapter/checkpoint-60" \
  --devices 0
```

#### Toàn bộ args của `finetune_vl.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--task {ocr,table,formula,chart}` | `ocr` | Task mặc định cho row không có cột `task`. |
| `--dataset-dir PATH [PATH ...]` | Một trong hai với `--prepared-from` | Dataset VL raw; hỗ trợ nhiều source. |
| `--dataset-task TASK [TASK ...]` | Không có | Task mặc định từng source, cùng thứ tự `--dataset-dir`. |
| `--prepared-from PATH [PATH ...]` | Một trong hai với `--dataset-dir` | Dùng lại một hoặc nhiều run `--prepare-only`; không copy JSONL/ảnh. |
| `--prepared-weight WEIGHT [WEIGHT ...]` | Mặc định `1.0` cho một run; bắt buộc khi nhiều run | Positive relative weight theo đúng thứ tự prepared run; tự normalize. |
| `--erniekit-dir PATH` | Bắt buộc khi train/inspect | Checkout ERNIEKit release/v1.5; không cần cho prepare-only. |
| `--model PATH_OR_ID` | `PaddlePaddle/PaddleOCR-VL-1.6` | Model; train yêu cầu local snapshot hợp lệ. |
| `--work-dir PATH` | Tự sinh timestamp | Thư mục run; resume phải dùng đúng run này. |
| `--prepare-only` | Tắt | Chỉ validate/stage; cấm dùng với `--prepared-from`. |
| `--smoke-steps INT` | Không có | Override `max_steps` cho smoke; phải dương. |
| `--resume-from PATH` | Không có | Resume adapter checkpoint; cấm dùng với `--prepared-from`. |
| `--inspect-model` | Tắt | Inspect trainable LoRA rồi dừng trước train. |
| `--epochs FLOAT` | `3.0` | Số epoch logic; script tự tính optimizer `max_steps`. |
| `--learning-rate FLOAT` | `1e-4` | Learning rate LoRA. |
| `--lora-rank INT` | `32` | LoRA rank; alpha bằng `2 * rank`. |
| `--min-pixels INT` | `50176` | Pixel tối thiểu khi smart-resize (`64 * 28 * 28`). |
| `--max-pixels INT` | `451584` | Pixel tối đa khi smart-resize (`576 * 28 * 28`). |
| `--max-image-pixels INT` | `50_000_000` | Giới hạn ảnh nguồn. |
| `--max-seq-len INT` | `2048` | Giới hạn prompt/image/target token; sample vượt bị reject, không truncate target. |
| `--gradient-accumulation-steps INT` | `32` | Số micro-batch trước optimizer step; micro-batch mặc định là 1. |
| `--validation-ratio FLOAT` | `0.02` | Tách validation cho source raw thiếu validation; trong `(0, 0.5)`. |
| `--num-workers INT` | `2` | Data loader workers. |
| `--prefetch-factor INT` | `2` | Batch prefetch mỗi worker. |
| `--seed INT` | `2026` | Seed prepare/sampling/runtime. |
| `--eval-samples-per-dataset INT` | `32` | Số validation row tối đa mỗi source. |
| `--eval-max-new-tokens INT` | `1024` | Generation limit chung. |
| `--eval-task-max-new-tokens TASK=INT` | Không có, lặp được | Override generation limit theo task, ví dụ `table=2048`. |
| `--eval-max-checkpoints INT` | `3` | Số checkpoint gần nhất cộng adapter cuối được chấm. |
| `--min-normalized-edit-distance FLOAT` | `0.5` | NED tối thiểu; trong `[0, 1]`. |
| `--max-cer FLOAT` | `1.0` | CER tối đa; không âm. |
| `--save-steps INT` | `100` | Chu kỳ lưu checkpoint theo optimizer step. |
| `--skip-evaluation` | Tắt | Bỏ native evaluator/checkpoint selection; vẫn merge/verify, không dùng cho quality claim. |
| `--devices VALUE` | `CUDA_VISIBLE_DEVICES` hoặc `0` | GPU ID comma-separated; ví dụ `0` hoặc `0,1`. |
| `--no-flash-attention` | Tắt | Tắt FlashAttention khi runtime/hardware không tương thích. |

Resolved config cố định decoder-only LoRA, vision frozen, BF16/O2, full
recompute, cosine LR, warmup `0.03`, weight decay `0.01`, micro-batch 1 và
`do_eval: false`. Validation được chấm bằng native evaluator, không phải Trainer
validation loss.

Output:

```text
<work-dir>/
├── prepared/                 # chỉ có khi prepare từ raw source trong run này
├── rejected.jsonl
├── summary.json
├── resolved.yaml
├── adapter/
│   ├── checkpoint-*/
│   ├── lora_config.json
│   └── export/
│       ├── model.safetensors
│       ├── merge_verification.json
│       └── logits_verification.json
├── export.yaml
├── export_manifest.json
├── logs/
├── metrics/
└── tensorboard_logs/
```

### D. Native evaluator — `evaluate_paddleocr_vl.py`

Trainer VL gọi utility này tự động. Chạy độc lập để chấm lại merged model:

```bash
.venv-vl-eval/bin/python evaluate_paddleocr_vl.py \
  --base-model "$VL_MODEL" \
  --merged-model "$OUTPUT_ROOT/vl_labeler_pilot/adapter/export" \
  --validation-jsonl "$OUTPUT_ROOT/vl_prepare/prepared/validation-source-000.jsonl" \
  --output-dir "$OUTPUT_ROOT/vl_labeler_pilot/manual_eval" \
  --samples-per-dataset 8 \
  --max-new-tokens 1024 \
  --task-max-new-tokens table=2048 \
  --min-normalized-edit-distance 0.5 \
  --max-cer 1.0
```

#### Toàn bộ args của `evaluate_paddleocr_vl.py`

| Argument | Bắt buộc/default | Ý nghĩa |
| --- | --- | --- |
| `--base-model PATH` | Bắt buộc | Snapshot base để tạo baseline. |
| `--merged-model PATH` | Bắt buộc | HF merged model cần đánh giá. |
| `--validation-jsonl PATH [PATH ...]` | Bắt buộc | Một hoặc nhiều validation JSONL. |
| `--output-dir PATH` | Bắt buộc | Nơi ghi metrics/predictions. |
| `--samples-per-dataset INT` | `32` | Số row tối đa mỗi JSONL; phải dương. |
| `--max-new-tokens INT` | `1024` | Generation limit chung. |
| `--task-max-new-tokens TASK=INT` | Không có, lặp được | Generation limit riêng task. |
| `--min-normalized-edit-distance FLOAT` | `0.5` | Ngưỡng NED quality gate. |
| `--max-cer FLOAT` | `1.0` | Ngưỡng CER quality gate. |
| `--base-predictions-jsonl PATH` | Không có | Tái sử dụng base predictions khi chấm nhiều checkpoint. |
| `--report-only` | Tắt | Ghi failed report nhưng không trả non-zero; chỉ dùng smoke/screening. |

Output chính: `ocr_metrics.json`, `ocr_predictions.jsonl`, metric overall/theo
source/theo task và trạng thái EOS/token limit.

### E. Merge utility — `merge_paddleocr_vl_lora.py`

```bash
CUDA_VISIBLE_DEVICES=0 \
$ERNIEKIT_DIR/.venv/bin/python merge_paddleocr_vl_lora.py \
  --base-model "$VL_MODEL" \
  --adapter-dir "$OUTPUT_ROOT/vl_labeler_pilot/adapter" \
  --output-dir "$OUTPUT_ROOT/vl_labeler_pilot/manual_merge" \
  --fixture-jsonl "$OUTPUT_ROOT/vl_prepare/prepared/validation-source-000.jsonl" \
  --min-pixels 50176 \
  --max-pixels 250880
```

#### Toàn bộ args của `merge_paddleocr_vl_lora.py`

| Argument | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `--base-model PATH` | Có | Base HF snapshot có `model_type=paddleocr_vl`. |
| `--adapter-dir PATH` | Có | Adapter chứa `lora_config.json` và weights. |
| `--output-dir PATH` | Có | Output mới; không nằm trong base và không chứa safetensors cũ. |
| `--fixture-jsonl PATH` | Có | JSONL dùng kiểm tra logits trước/sau merge. |
| `--min-pixels INT` | Có | Min resize giống train/eval. |
| `--max-pixels INT` | Có | Max resize giống train/eval. |

Merge phải tạo `model.safetensors`, `merge_verification.json` và
`logits_verification.json`; cả hai verification phải có status `passed`.

### F. Quality gate và checkpoint selection VL

Full run evaluate adapter cuối và tối đa `--eval-max-checkpoints` checkpoint gần
nhất. Checkpoint được chọn theo CER, exact match và normalized edit distance,
không theo train loss đơn thuần. Chỉ gọi model `passed` khi:

- LoRA scope không có tensor vision;
- base model không bị sửa;
- merge verification pass;
- logit reload verification pass;
- evaluator không chạm token limit bất thường;
- merged không regression rõ ràng so với base;
- metric vượt thresholds đã khai báo.

Kiểm tra đồng thời:

```text
metrics/ocr_metrics.json
metrics/ocr_predictions.jsonl
metrics/checkpoint_selection.json
adapter/export/merge_verification.json
adapter/export/logits_verification.json
export_manifest.json
```

### G. Lỗi thường gặp

| Triệu chứng | Xử lý |
| --- | --- |
| Dùng cùng `--dataset-dir` và `--prepared-from` | Chọn raw mode hoặc prepared mode, không chọn cả hai. |
| Nhiều prepared run thiếu/sai `--prepared-weight` | Truyền đúng một số dương hữu hạn cho mỗi run, ví dụ `95 5`. |
| VL thiếu task | Thêm cột `task`, hoặc dùng `--task`/`--dataset-task` đúng thứ tự source. |
| Table bị reject | Dùng OTSL canonical, không dùng HTML. |
| CUDA OOM VL | Giảm `--max-pixels`, giữ micro-batch 1, thử `--no-flash-attention`; không mở LoRA vision. |
| CUDA OOM rec/det | Giảm `--batch-size` trước; recognition mới cân nhắc giảm `--image-width`. |
| Detection pretrained mismatch | Dùng training `.pdparams` đúng model, không dùng inference directory. |
| AMP/GradScaler lỗi NumPy | Cài dependency đã pin, đặc biệt `numpy<2.4`. |
| Validation quá ít | Bổ sung validation theo page/task; không hạ gate chỉ để tuyên bố pass. |
| Merge output tồn tại | Chọn output mới; utility cố ý không ghi đè safetensors. |

### H. Kiểm tra trước bàn giao

```bash
PYTHONPATH=. pytest -q \
  tests/test_finetune.py \
  tests/test_finetune_det.py \
  tests/test_finetune_vl.py \
  tests/test_finetune_vl_layout.py

bash -n download_pretrained_models.sh
python finetune.py --help >/tmp/finetune-rec-help.txt
python finetune_det.py --help >/tmp/finetune-det-help.txt
python finetune_vl.py --help >/tmp/finetune-vl-help.txt
python evaluate_paddleocr_vl.py --help >/tmp/evaluate-vl-help.txt
python merge_paddleocr_vl_lora.py --help >/tmp/merge-vl-help.txt
```
