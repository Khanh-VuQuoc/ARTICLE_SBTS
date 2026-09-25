# Hướng dẫn hiệu chỉnh luận văn

Phạm vi làm việc là `thesis/UEH_SBTS_THESIS/`. Trước khi sửa, đọc:

1. `UEH_SBTS_THESIS/QUY_TAC_VIET_LUAN_VAN.md`;
2. `REVIEW_NOTES.md`;
3. năm tệp trong `UEH_SBTS_THESIS/chapters/`.

## Nguồn kết quả chính thức

Ba notebook sau là nguồn mã và kết quả của luận văn:

- `../notebooks/25_GBM.ipynb`;
- `../notebooks/25_heston_SBTS.ipynb`;
- `../notebooks/25_merge.ipynb`.

Không thay đổi notebook. Chỉ đối chiếu đầu ra để hiệu chỉnh nội dung luận văn.
Các notebook `article_*` không phải nguồn kết quả cuối của luận văn.

## Giới hạn nội dung

- Không bổ sung Doob $h$-transform hoặc dẫn xuất drift tối ưu.
- Không thay đổi số liệu nếu chưa đối chiếu notebook `25_*`.
- Không che giấu trường hợp seed 10 thay seed 3 của cấu hình
  SBTS--Asian worst-of put--$\kappa=0{,}95$.
- Không khẳng định cơ chế ngoại suy của kernel đã được kiểm định trực tiếp.
- Sau mỗi đợt sửa, biên dịch bằng XeLaTeX và kiểm tra toàn bộ tham chiếu.
