# src/models/pipeline_builder.py - Class đóng gói Pipeline tiền xử lý và mô hình Scikit-Learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.data.data_loader import DataLoader

class PipelineBuilder:
    """
    Class chịu trách nhiệm xây dựng bộ tiền xử lý ColumnTransformer
    và đóng gói Pipeline Học máy cho các thuật toán phân loại.
    """

    def __init__(self, text_cols, cat_cols):
        self.text_cols = text_cols
        self.cat_cols = cat_cols

    def build_preprocessor(self):
        """Đóng gói TfidfVectorizer cho trường văn bản và OneHotEncoder cho danh mục."""
        transformers = []
        for text_col in self.text_cols:
            transformers.append((
                f'tfidf_{text_col}',
                TfidfVectorizer(
                    preprocessor=DataLoader.clean_text,
                    ngram_range=(1, 2),
                    max_features=1000
                ),
                text_col
            ))
            
        if self.cat_cols:
            transformers.append((
                'cat_ohe',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                self.cat_cols
            ))
            
        return ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

    @staticmethod
    def build_model_instance(model_name, model_params=None):
        """Khởi tạo mô hình học máy với siêu tham số (Hyperparameters) linh hoạt."""
        params = model_params or {}
        
        if model_name == 'Logistic Regression':
            c_val = params.get('C', 1.0)
            max_iter_val = params.get('max_iter', 1000)
            return LogisticRegression(
                C=c_val, 
                class_weight='balanced', 
                solver='lbfgs', 
                max_iter=max_iter_val, 
                random_state=42
            )
                
        elif model_name == 'Random Forest':
            n_est = params.get('n_estimators', 100)
            max_d = params.get('max_depth', None)
            return RandomForestClassifier(
                n_estimators=n_est, 
                max_depth=max_d, 
                class_weight='balanced', 
                random_state=42
            )
            
        elif model_name == 'Gradient Boosting':
            n_est = params.get('n_estimators', 100)
            lr = params.get('learning_rate', 0.1)
            max_d = params.get('max_depth', 3)
            return GradientBoostingClassifier(
                n_estimators=n_est, 
                learning_rate=lr, 
                max_depth=max_d, 
                random_state=42
            )
        else:
            raise ValueError(f"Mô hình không hợp lệ: {model_name}")

    def build_pipeline(self, model_name, model_params=None):
        """Tạo Pipeline khép kín chống rò rỉ dữ liệu (Data Leakage)."""
        preprocessor = self.build_preprocessor()
        model = self.build_model_instance(model_name, model_params=model_params)
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

def build_pipeline(model_name, text_cols, cat_cols, model_params=None):
    builder = PipelineBuilder(text_cols, cat_cols)
    return builder.build_pipeline(model_name, model_params=model_params)
