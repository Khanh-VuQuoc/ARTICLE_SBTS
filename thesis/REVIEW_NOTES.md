# Các điểm cần xử lý trước khi khóa luận văn

## Mức ưu tiên 1

1. Chương 3 đang ghi sai nguồn notebook. Thay `article_*` bằng
   `25_GBM.ipynb`, `25_heston_SBTS.ipynb` và `25_merge.ipynb`.
2. Chương 2 đang biểu diễn SBTS bằng các ràng buộc phân phối biên
   `Q_{t_j}=mu_{t_j}`. Cần trình bày nhất quán với phân phối chung của chuỗi
   trên không gian quỹ đạo, hoặc giới hạn lại kết luận về phụ thuộc thời gian.
3. Chương 4 và Chương 5 phải ghi chú rằng kiểm định của cấu hình
   SBTS--Asian worst-of put--$\kappa=0{,}95$ không có cấu trúc ghép cặp hoàn
   toàn do seed 3 được thay bằng seed 10.
4. Kiểm tra lại cách gọi $V_0^*$. Với hàm mất mát MSE, ưu tiên “mức vốn ban
   đầu tối ưu theo tiêu chuẩn bình phương sai số” hoặc “mức vốn ban đầu tối ưu
   trong bài toán phòng ngừa theo phương sai”. Không gọi là giá thị trường.

## Mức ưu tiên 2

1. Đối chiếu notebook để thống nhất khoảng giảm 8--38\% hay 8--39\%.
2. Ghi rõ cặp mô hình của từng cột $p_{\mathrm{BH}}$ trong bảng Chương 4.
3. Đổi tiêu đề cột “Mức giảm” thành “Thay đổi” nếu cột có giá trị âm.
4. Kiểm tra biến thời gian trong trạng thái mạng là $T-t$ hay
   $\tau_t=(T-t)/T$ theo đúng mã nguồn.
5. Phân biệt seed chia tập 42 với mười seed huấn luyện.
6. Giải thích vì sao 3.774 quan sát tạo 3.522 cửa sổ tham chiếu.
7. Chương 1 đang dùng $\Phi$ cho hàm phân phối chuẩn, trong khi các chương sau
   dùng $\Phi$ cho khoản thanh toán; cần đổi ký hiệu hoặc rút gọn phần công
   thức Black--Scholes. Ký hiệu $K$ cũng trùng với bậc nhớ của SBTS.

## Thuật ngữ cần thay thống nhất

- `bộ sinh` → `mô hình sinh dữ liệu`;
- `chế độ thị trường` → `giai đoạn thị trường`;
- `strike` → `mức giá thực hiện`;
- `hedging rủi ro` → `phòng ngừa rủi ro`;
- `sai số hedging` → `sai số phòng ngừa`;
- `danh mục hedging` → `danh mục phòng ngừa`;
- `cầu Schrödinger` → `\textit{Schrödinger Bridge}`;
- `deep hedging` → `\textit{Deep Hedging}`.
