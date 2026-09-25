# UEH SBTS Thesis

## Cấu trúc

- `main.tex`: file biên dịch chính theo lớp UEH.
- `TKT-UEH.cls`: lớp định dạng do UEH cung cấp.
- `WholeBook.tex`: nạp lần lượt năm chương.
- `chapters/`: mỗi chương là một file độc lập để duyệt và sửa theo phiên bản.
- `trangbia/`: tóm tắt tiếng Việt, tiếng Anh và danh mục từ viết tắt.
- `figures/`: hình và đồ thị của luận văn.
- `phuluc/`: phụ lục kỹ thuật và bảng kết quả bổ sung.
- `references.bib`: cơ sở dữ liệu tài liệu tham khảo dùng chung.

## Trạng thái

- Chương 1: bản thảo đầy đủ, sẵn sàng để phản hồi.
- Chương 2: bản ráp đầy đủ, sẵn sàng để phản hồi.
- Chương 3: bản ráp đầy đủ, sẵn sàng để phản hồi.
- Chương 4: bản ráp đầy đủ từ các bảng kết quả đã khóa, sẵn sàng để phản hồi.
- Chương 5: bản ráp đầy đủ, sẵn sàng để phản hồi.

## Nguồn nội dung chính thức

1. Các notebook `25_GBM.ipynb`, `25_heston_SBTS.ipynb` và `25_merge.ipynb`
   trong repository `ARTICLE_SBTS` là nguồn quy trình và kết quả thực nghiệm.
2. `Main GBM_HESTON_SBTS_final (1).tex` là nguồn bản thảo học thuật để chuyển hóa sang cấu trúc UEH bằng tiếng Việt.
3. `technical_note (1).tex` và `hedging_note.tex` là nguồn giải thích kỹ thuật.
4. Hai bài báo Schrödinger Bridge được dùng cho Chương 2.
5. Các notebook `article_*` chỉ dùng để đối chiếu và không phải nguồn kết quả
   cuối của luận văn.

## Trình tự duyệt

1. Duyệt và sửa song song Chương 2--5.
2. Khóa thuật ngữ, phương trình và bảng số liệu của từng chương.
3. Đồng bộ lại Chương 1, tóm tắt tiếng Việt và tiếng Anh theo bản đã khóa.
4. Biên tập toàn văn và kiểm tra định dạng trước khi nộp.

## Biên dịch

Chọn `XeLaTeX` và backend tài liệu tham khảo `Biber`. Trình tự chuẩn:

```text
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Hệ thống cần có font Times New Roman và các gói `babel-vietnamese`, `polyglossia`, `biblatex`, `biber` theo yêu cầu của `TKT-UEH.cls`.
