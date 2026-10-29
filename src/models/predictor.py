# src/models/predictor.py - Standalone Predictor OOP

import pandas as pd
import numpy as np

from config.settings import DEFAULT_DATA_PATH
from src.data.data_loader import DataLoader
from src.models.model_trainer import TestCaseModelTrainer

class StandalonePredictor:
    """Class độc lập chạy pipeline dự đoán tương lai từ tập tin dữ liệu Excel."""

    def __init__(self, file_path=DEFAULT_DATA_PATH):
        self.file_path = file_path
        self.loader = DataLoader(file_path)
        self.trainer = TestCaseModelTrainer()

    def run(self):
        print("==========================================================================")
        print(" 🚀 HỆ THỐNG ML DỰ ĐOÁN TƯƠNG LAI CHO CÁC TEST CASE BLOCK & CHƯA TEST")
        print("==========================================================================")
        
        data = self.loader.load_and_clean_data()
        X_hist = data['X_history']
        y_hist = data['y_history']
        df_hist = data['df_history']
        X_future = data['X_future']
        df_future = data['df_future']
        text_cols = data['text_cols']
        cat_cols = data['cat_cols']
        
        print(f"-> Tổng số mẫu testcase: {len(data['df_all'])}")
        print(f"-> Tập Lịch sử (Pass/Fail đã test): {len(df_hist)} mẫu")
        print(f"-> Tập Tương lai (Block/Chưa test): {len(df_future)} mẫu cần dự đoán")
        
        results = self.trainer.train_and_evaluate_all(X_hist, y_hist, text_cols, cat_cols)
        
        df_pred_future = self.trainer.rank_future_risks(results['final_pipelines'], X_future, df_future)
        
        print("\n--- TỔNG HỢP DỰ ĐOÁN TƯƠNG LAI KHI UNBLOCK / TEST ---")
        if len(df_pred_future) > 0:
            display_cols = [c for c in df_pred_future.columns if c in ('tên testcase', 'module', 'trạng thái') or c.startswith('P(Pass)') or c.startswith('Hạng')]
            print(df_pred_future[display_cols].head(10))
            
        if results['confusion_matrices']:
            self.trainer.plot_confusion_matrices(results['confusion_matrices'], labels=results['labels'])
        print("\n✅ Hoàn tất pipeline dự đoán tương lai!")

def run_prediction_pipeline(file_path=DEFAULT_DATA_PATH):
    predictor = StandalonePredictor(file_path)
    predictor.run()
