# src/data/data_loader.py - Class nạp và làm sạch dữ liệu OOP

import pandas as pd
import re
from config.settings import COLUMN_PATTERNS, TARGET_COL, HISTORICAL_TARGETS, PENDING_TARGETS, DROP_COLS, TEXT_COLS, CAT_COLS

class DataLoader:
    """
    Class quản lý quy trình nạp dữ liệu từ Excel, chuẩn hóa tên cột,
    làm sạch văn bản tiếng Việt và phân tách tập có nhãn (lịch sử) với tập cần dự đoán (tương lai).
    Lớp này không nhân bản mẫu; mất cân bằng nhãn được xử lý ở bước huấn luyện (class_weight / sample_weight).
    """

    def __init__(self, file_source, target_col=TARGET_COL):
        self.file_source = file_source
        self.target_col = target_col

    @staticmethod
    def clean_text(text):
        """Làm sạch văn bản tiếng Việt: chuyển chữ thường và xóa khoảng trắng thừa."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def detect_header_row(self):
        """Tự động quét và tìm dòng tiêu đề (Header Row) trong tệp Excel."""
        df_raw = pd.read_excel(self.file_source, header=None)
        for i in range(min(15, len(df_raw))):
            row_vals = [str(x).lower() for x in df_raw.iloc[i].values]
            if any('trạng thái' in v or 'status' in v for v in row_vals):
                return i
        return 0

    @staticmethod
    def ensure_model_features(df_input, text_cols, cat_cols):
        """Đảm bảo DataFrame luôn chứa đầy đủ các cột Text và Categorical yêu cầu."""
        df_out = df_input.copy()
        for col in text_cols:
            if col not in df_out.columns:
                df_out[col] = ""
            else:
                df_out[col] = df_out[col].fillna("").astype(str)
                
        for col in cat_cols:
            if col not in df_out.columns:
                df_out[col] = "Unknown"
            else:
                df_out[col] = df_out[col].fillna("Unknown").astype(str)
                
        return df_out

    def load_and_clean_data(self):
        """
        Nạp dữ liệu từ Excel, chuẩn hóa tên cột và phân tách:
        - Tập Lịch sử (Pass, Fail) dùng để huấn luyện mô hình ML.
        - Tập Tương lai (Block, Chưa test) dùng để dự đoán khả năng Pass khi test trong tương lai.
        """
        header_idx = self.detect_header_row()
        df = pd.read_excel(self.file_source, header=header_idx)
        
        column_mapping = {}
        for col in df.columns:
            c_str = str(col).strip().lower()
            for std_col, patterns in COLUMN_PATTERNS.items():
                if any(p in c_str for p in patterns):
                    column_mapping[col] = std_col
                    break
                    
        df = df.rename(columns=column_mapping)
        
        if self.target_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột nhãn target '{self.target_col}' trong tệp Excel.")
            
        df[self.target_col] = df[self.target_col].fillna("Chưa test").astype(str).str.strip().str.capitalize()
        df[self.target_col] = df[self.target_col].replace({
            'Chưa thực hiện': 'Chưa test',
            'Untested': 'Chưa test',
            'Pending': 'Block'
        })
        
        df_history = df[df[self.target_col].isin(HISTORICAL_TARGETS)].copy()
        df_future = df[~df[self.target_col].isin(HISTORICAL_TARGETS)].copy()
        
        if len(df_history) == 0:
            df_history = df.copy()
            
        
        existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
        
        X_history = df_history.drop(columns=[self.target_col] + existing_drop_cols, errors='ignore')
        y_history = df_history[self.target_col]
        
        X_future = df_future.drop(columns=[self.target_col] + existing_drop_cols, errors='ignore') if len(df_future) > 0 else pd.DataFrame()
        
        text_cols = [c for c in TEXT_COLS if c in X_history.columns]
        cat_cols = [c for c in CAT_COLS if c in X_history.columns]
        
        X_history = self.ensure_model_features(X_history, text_cols, cat_cols)
        if len(X_future) > 0:
            X_future = self.ensure_model_features(X_future, text_cols, cat_cols)
                
        return {
            'X_history': X_history,
            'y_history': y_history,
            'df_history': df_history,
            'X_future': X_future,
            'df_future': df_future,
            'text_cols': text_cols,
            'cat_cols': cat_cols,
            'df_all': df
        }

# Helpers
def load_and_clean_data(file_source):
    loader = DataLoader(file_source)
    return loader.load_and_clean_data()

def clean_text(text):
    return DataLoader.clean_text(text)

def ensure_model_features(df_input, text_cols, cat_cols):
    return DataLoader.ensure_model_features(df_input, text_cols, cat_cols)
