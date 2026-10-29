# Dự đoán khả năng vượt qua (Pass) của test case bằng 3 mô hình học máy

Đọc danh sách test case từ file Excel, huấn luyện Logistic Regression, Random Forest và Gradient Boosting trên các test case
có kết quả Pass/Fail, rồi dự đoán xác suất Pass cho các test case Block / Chưa test và xếp hạng rủi ro.

## Cấu trúc
- `config/settings.py` – hằng số: tên cột, nhãn, nhóm đặc trưng, ánh xạ tên cột.
- `src/data/data_loader.py` – đọc Excel, tự phát hiện dòng tiêu đề, chuẩn hóa, tách tập có nhãn / cần dự đoán.
- `src/models/pipeline_builder.py` – ColumnTransformer (TF-IDF + One-Hot) và mô hình phân loại.
- `src/models/model_trainer.py` – chia dữ liệu, huấn luyện, đánh giá (hold-out, 200 lần chia, Leave-One-Out), dự đoán, xếp hạng.
- `src/models/predictor.py`, `main.py` – chạy từ dòng lệnh. `app.py`, `src/ui/dashboard.py` – giao diện Streamlit.

## Chạy
    pip install pandas scikit-learn matplotlib seaborn openpyxl streamlit
    python main.py          # kết quả ghi ra ket_qua_du_doan.xlsx
    streamlit run app.py
