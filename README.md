# Fine-tune PaddleOCR tiếng Việt

Pipeline LoRA cho **PaddleOCR-VL-1.6** được tài liệu riêng tại
[docs/finetune-paddleocr-vl-1.6.md](docs/finetune-paddleocr-vl-1.6.md). Pipeline
này dùng ERNIEKit và môi trường tách biệt với PP-OCRv6 bên dưới. Dataset VL đã
prepare có thể dùng lại bằng `finetune_vl.py --prepared-from <prepare-run>` mà
không sao chép hoặc xử lý lại ảnh. OCR/table/formula/chart dùng chung
`finetune_vl.py`; trộn nhiều layout trong một run qua cột `task` trong dataset
để hạn chế phá các khả năng layout khác.

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
