# Quy tắc viết luận văn

## 1. Văn phong

- Viết theo văn phong khoa học, trực tiếp và trung tính.
- Mỗi đoạn trình bày một nội dung chính.
- Mỗi câu chỉ chứa một quan hệ hoặc một kết quả cần diễn đạt.
- Ưu tiên cấu trúc: đối tượng, phương pháp, kết quả, điều kiện áp dụng.
- Không dùng câu dẫn mang tính giới thiệu, quảng bá hoặc nhận xét chủ quan.
- Không dùng các cụm như “đáng chú ý”, “rất quan trọng”, “ưu thế rõ rệt”,
  “kết quả ấn tượng”, “có thể thấy rằng” nếu số liệu đã thể hiện trực tiếp.
- Không khẳng định quan hệ nhân quả khi thiết kế chỉ cho phép xác định mối liên
  hệ hoặc thứ hạng.

## 2. Thuật ngữ thống nhất

| Sử dụng | Không sử dụng |
| --- | --- |
| mô hình sinh dữ liệu | bộ sinh |
| \textit{Deep Hedging} | deep hedging viết thường |
| \textit{Schrödinger Bridge} | cầu Schrödinger |
| quy trình nghiên cứu/huấn luyện | khuôn khổ, framework |
| giai đoạn thị trường | chế độ thị trường |
| mức giá thực hiện | strike |
| khoản thanh toán | payoff trong phần diễn giải tiếng Việt |
| sai số phòng ngừa | residual trong phần diễn giải tiếng Việt |
| hàm mất mát | loss trong phần diễn giải tiếng Việt |
| tập xác thực | validation set trong phần diễn giải tiếng Việt |
| rổ tài sản | danh mục tài sản khi nói về tài sản cơ sở |
| danh mục phòng ngừa | rổ tài sản khi nói về vị thế giao dịch |

Các tên hợp đồng được giữ bằng tiếng Anh và viết nhất quán: basket Asian call
và Asian worst-of put. Các ký hiệu MSE, CVaR, MCS, GBM, Heston và SBTS được giữ
nguyên.

## 3. Trình bày phương pháp

- Nêu dữ liệu, biến số, công thức và tham số trước phần giải thích.
- Phân biệt dữ liệu dùng để hiệu chỉnh mô hình sinh dữ liệu, dữ liệu tổng hợp
  dùng để huấn luyện và dữ liệu lịch sử dùng để đánh giá.
- Chỉ mô tả quy trình và cấu hình đã được sử dụng trong thực nghiệm.
- Không trình bày Doob $h$-transform hoặc dẫn xuất riêng của drift tối ưu.
- Phần \textit{Schrödinger Bridge} đi theo thứ tự: bài toán entropy, HJB,
  Cole--Hopf, phương trình nhiệt và ước lượng Nadaraya--Watson.

## 4. Trình bày kết quả

- Mỗi kết luận định lượng phải gắn với chỉ tiêu, giai đoạn thị trường, hợp đồng
  và mức giá thực hiện.
- Dùng “có khác biệt thống kê” khi $p_{\mathrm{BH}}<0,05$; các trường hợp còn
  lại ghi “chưa có khác biệt thống kê”.
- Phân biệt kết quả thực nghiệm và cơ chế giải thích.
- Không dùng kết quả phù hợp phân phối để thay cho kết quả phòng ngừa rủi ro.
- Hiệu quả phòng ngừa được mô tả bằng P\&L trung bình, độ lệch chuẩn sai số
  phòng ngừa và CVaR sau chi phí giao dịch.
- $V_0^{*}$ được gọi là mức vốn ban đầu tối ưu theo tiêu chuẩn bình phương sai
  số; không tự động đồng nhất với giá bàng quan nếu chưa xây dựng một bài toán
  định giá bàng quan riêng.
- $V_0^{*}$ không được diễn giải như giá thị trường hoặc giá phi arbitrage.

## 5. Cấu trúc đoạn và bảng

- Đoạn phương pháp: định nghĩa, công thức, tham số, cách áp dụng.
- Đoạn kết quả: số liệu, kiểm định, phạm vi kết luận.
- Đoạn hạn chế: nội dung không được kiểm định hoặc phạm vi không được bao phủ.
- Tiêu đề bảng phải nêu chỉ tiêu, giai đoạn và đơn vị khi cần.
- Chú thích bảng không lặp lại nội dung đã trình bày trong thân bài.

## 6. Kiểm tra trước khi hoàn tất

- Không còn các từ: “bộ sinh”, “chế độ”, “strike”, “hạ nguồn”.
- Tên \textit{Deep Hedging} và \textit{Schrödinger Bridge} được viết thống
  nhất.
- Mỗi tỷ lệ phần trăm có đối tượng so sánh rõ ràng.
- Mỗi kết luận về giả thuyết phù hợp với đúng ngưỡng kiểm định đã báo cáo.
- Các môi trường LaTeX được đóng đầy đủ; không chèn dấu Markdown vào tệp
  `.tex`.
