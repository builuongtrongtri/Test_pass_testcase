# config/settings.py - Cấu hình trung tâm cho dự án ML Dự đoán Trạng thái Test Case

DEFAULT_DATA_PATH = 'Danh sách Test Case.xlsx'

TARGET_COL = 'trạng thái'
HISTORICAL_TARGETS = ['Pass', 'Fail']
PENDING_TARGETS = ['Block', 'Chưa test', 'Chưa thực hiện', 'Pending']

# Các cột không dùng làm đặc trưng:
#  - 'stt', 'testcase id': chỉ là mã định danh.
#  - 'kết quả thực tế': chỉ có giá trị ở test case Fail (1/1), trống ở toàn bộ test case Pass và cần dự đoán -> làm lộ nhãn.
#  - 'ngày thực hiện', 'người thực hiện': chỉ có giá trị ở test case đã chạy, trống ở toàn bộ 152 test case cần dự đoán,
#    nên không có sẵn tại thời điểm dự đoán (và làm lộ nhãn nếu dùng để huấn luyện).
#  - 'module': 5 module chỉ có Block (không có nhãn Pass/Fail) nên mô hình không học được gì từ cột này.
DROP_COLS = ['stt', 'testcase id', 'kết quả thực tế', 'ngày thực hiện', 'người thực hiện', 'module']
TEXT_COLS = ['tên testcase', 'điều kiện tiền đề', 'bước thực hiện', 'dữ liệu test', 'kết quả mong đợi']
CAT_COLS = ['mức độ ưu tiên']

# Mapping cột linh hoạt để xử lý các file Excel có tên cột khác nhau
COLUMN_PATTERNS = {
    'tên testcase': ['tên testcase', 'tên test case'],
    'điều kiện tiền đề': ['điều kiện tiền đề'],
    'bước thực hiện': ['bước thực hiện'],
    'dữ liệu test': ['dữ liệu test', 'dữ liệu'],
    'kết quả mong đợi': ['kết quả mong đợi'],
    'module': ['module'],
    'mức độ ưu tiên': ['mức độ ưu tiên'],
    'người thực hiện': ['người thực hiện', 'người test'],
    'ngày thực hiện': ['ngày thực hiện', 'ngày test'],
    'trạng thái': ['trạng thái', 'status'],
    'kết quả thực tế': ['kết quả thực tế'],
    'testcase id': ['testcase id', 'test case id'],
    'stt': ['stt']
}
