# main.py - CLI Terminal Entry Point

from config.settings import DEFAULT_DATA_PATH
from src.data.data_loader import DataLoader
from src.models.model_trainer import TestCaseModelTrainer
from sklearn.metrics import classification_report


class TerminalApp:
    """Class ứng dụng chạy dự đoán từ Terminal theo kiến trúc OOP."""

    def __init__(self, data_path=DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.loader = DataLoader(data_path)
        self.trainer = TestCaseModelTrainer()

    def run(self):
        print("=" * 74)
        print(" DỰ ÁN ML: DỰ ĐOÁN KHẢ NĂNG PASS CHO CÁC TEST CASE BLOCK / CHƯA TEST")
        print("=" * 74)

        # 1. Nạp và phân tách dữ liệu (Lịch sử vs Tương lai)
        data = self.loader.load_and_clean_data()
        X_hist, y_hist, df_hist = data['X_history'], data['y_history'], data['df_history']
        X_future, df_future = data['X_future'], data['df_future']
        text_cols, cat_cols = data['text_cols'], data['cat_cols']

        print(f"-> Tổng số test case trong file: {len(data['df_all'])}")
        print(f"-> Dữ liệu lịch sử có nhãn (Pass/Fail): {len(df_hist)} mẫu -> {y_hist.value_counts().to_dict()}")
        print(f"-> Dữ liệu cần dự đoán (Block/Chưa test): {len(df_future)} mẫu")
        print(f"-> Trường Text: {text_cols} | Trường Categorical: {cat_cols}")

        # 2. Chia train/test rồi huấn luyện, đánh giá hold-out
        print("\n--- 2. Đánh giá hold-out (chia trước, không nhân bản mẫu) ---")
        results = self.trainer.train_and_evaluate_all(X_hist, y_hist, text_cols, cat_cols)
        X_train, X_test, y_train, y_test = results['split_data']
        labels = results['labels']
        print(f"-> Train: {len(X_train)} mẫu ({y_train.value_counts().to_dict()}) | "
              f"Test: {len(X_test)} mẫu ({y_test.value_counts().to_dict()})")
        for m_name, msg in results['skipped'].items():
            print(f"   [!] {m_name}: không huấn luyện được trên tập train ({msg})")
        for m_name, pipe in results['pipelines'].items():
            y_pred = pipe.predict(X_test)
            print(f"\n== {m_name} ==")
            print(classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0))
        if results['confusion_matrices']:
            print("-> Biểu đồ ma trận nhầm lẫn:", self.trainer.plot_confusion_matrices(results['confusion_matrices'], labels=labels))

        # 3. Độ ổn định khi chia dữ liệu và Leave-One-Out
        rob = self.trainer.evaluate_split_robustness(y_hist, n_seeds=200)
        print(f"\n--- 3. Độ ổn định: chia ngẫu nhiên {rob['n_seeds']} lần ---")
        print(f"-> Mẫu Fail rơi vào tập test (train không có Fail): {rob['fail_in_test']} lần")
        print(f"-> Mẫu Fail rơi vào tập train (test không có Fail): {rob['fail_in_train']} lần")
        loo = self.trainer.leave_one_out(X_hist, y_hist, text_cols, cat_cols)
        print("-> Leave-One-Out: số test case Pass được nhận đúng (P(Pass) >= 50%) / tổng Pass hợp lệ:")
        for m in loo.columns:
            v = loo[m].dropna()
            print(f"   {m}: {(v >= 0.5).sum()}/{len(v)} | bỏ riêng mẫu Fail: "
                  f"{int(loo[m].isna().sum())} mẫu không đánh giá được")

        # 4. Dự đoán tương lai bằng mô hình huấn luyện trên toàn bộ dữ liệu có nhãn
        print("\n--- 4. Xếp hạng rủi ro cho các test case Block / Chưa test ---")
        ranked = self.trainer.rank_future_risks(results['final_pipelines'], X_future, df_future)
        cols = [c for c in ranked.columns if c.startswith('P(Pass)')]
        print(ranked[cols].describe().loc[['min', 'mean', 'max']].round(1))
        print("\nTOP 10 TEST CASE CÓ HẠNG RỦI RO CAO NHẤT (CHỈ MANG TÍNH THAM KHẢO):")
        for _, row in ranked.head(10).iterrows():
            print(f"  • [{row.get('module', 'N/A')}] {row.get('tên testcase', 'N/A')} | hạng TB: "
                  f"{row['Hạng rủi ro TB (1 = rủi ro cao nhất)']}")
        ranked.to_excel('ket_qua_du_doan.xlsx', index=False)
        print("\n-> Đã lưu: ket_qua_du_doan.xlsx")
        print("\n[LƯU Ý] Dữ liệu chỉ có 1 mẫu Fail: không đo được khả năng phát hiện Fail; "
              "xác suất và thứ hạng chỉ mang tính tham khảo.")


def main():
    TerminalApp().run()


if __name__ == '__main__':
    main()
