# src/models/model_trainer.py - Class Huấn luyện, Đánh giá & Dự đoán Tương lai OOP
# Nguyên tắc thiết kế:
#   1. Chia train/test trước mọi bước xử lý dữ liệu, không nhân bản mẫu (tránh rò rỉ dữ liệu).
#   2. Mất cân bằng nhãn được xử lý bằng class_weight (LR, RF) và sample_weight (GB).
#   3. Đánh giá bằng hold-out, chia ngẫu nhiên nhiều lần và Leave-One-Out (do chỉ có 1 mẫu Fail).
#   4. Mô hình dự đoán cuối cùng huấn luyện trên toàn bộ dữ liệu có nhãn; xếp hạng rủi ro theo trung bình hạng của 3 mô hình.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.models.pipeline_builder import PipelineBuilder

MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
LABELS = ['Fail', 'Pass']


def fit_pipeline(pipe, model_name, X, y):
    """Huấn luyện pipeline. Gradient Boosting không có class_weight nên dùng sample_weight."""
    if y.nunique() < 2:
        raise ValueError("Tập huấn luyện chỉ có 1 lớp, không thể huấn luyện bộ phân loại.")
    if model_name == 'Gradient Boosting':
        n_fail, n_pass = (y == 'Fail').sum(), (y != 'Fail').sum()
        w = np.where(y == 'Fail', n_pass / max(n_fail, 1), 1.0)
        pipe.fit(X, y, classifier__sample_weight=w)
    else:
        pipe.fit(X, y)
    return pipe


def pass_probability(pipe, X):
    classes = list(pipe.classes_)
    return pipe.predict_proba(X)[:, classes.index('Pass')]


class TestCaseModelTrainer:
    """
    Class quản lý quy trình huấn luyện 3 mô hình trên tập lịch sử,
    tính toán các thước đo đánh giá và dự đoán kết quả tương lai cho các testcase đang Block/Chưa test.
    """

    def __init__(self, model_names=None, labels=None):
        self.model_names = model_names or MODEL_NAMES
        self.labels = labels or LABELS

    # ------------------------------------------------------------------ đánh giá hold-out
    def train_and_evaluate_all(self, X_hist, y_hist, text_cols, cat_cols, test_size=0.20, random_state=42, hyperparams=None):
        """Chia train/test TRƯỚC, huấn luyện và đánh giá các mô hình trên tập dữ liệu lịch sử (Pass, Fail)."""
        hyperparams = hyperparams or {}

        class_counts = y_hist.value_counts()
        can_stratify = not (class_counts < 2).any()
        stratify_param = y_hist if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_hist, y_hist,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        unique_labels = sorted(list(y_hist.unique()))
        if len(unique_labels) < 2:
            unique_labels = self.labels

        builder = PipelineBuilder(text_cols, cat_cols)
        pipelines, final_pipelines, reports, confusion_matrices, skipped = {}, {}, {}, {}, {}

        for m_name in self.model_names:
            m_params = hyperparams.get(m_name, {})
            # (a) mô hình dùng để ĐÁNH GIÁ: chỉ thấy tập train
            try:
                pipe = fit_pipeline(builder.build_pipeline(m_name, model_params=m_params), m_name, X_train, y_train)
                y_pred = pipe.predict(X_test)
                pipelines[m_name] = pipe
                reports[m_name] = classification_report(
                    y_test, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
                confusion_matrices[m_name] = confusion_matrix(y_test, y_pred, labels=unique_labels)
            except ValueError as e:
                skipped[m_name] = str(e)
            # (b) mô hình dùng để DỰ ĐOÁN tương lai: huấn luyện trên toàn bộ dữ liệu có nhãn
            final_pipelines[m_name] = fit_pipeline(
                builder.build_pipeline(m_name, model_params=m_params), m_name, X_hist, y_hist)

        return {
            'pipelines': pipelines,
            'final_pipelines': final_pipelines,
            'reports': reports,
            'confusion_matrices': confusion_matrices,
            'skipped': skipped,
            'split_data': (X_train, X_test, y_train, y_test),
            'labels': unique_labels
        }

    # ------------------------------------------------------------------ độ ổn định khi chia dữ liệu
    @staticmethod
    def evaluate_split_robustness(y_hist, n_seeds=200, test_size=0.20):
        """Đếm số lần mẫu Fail rơi vào tập test / tập train khi chia ngẫu nhiên nhiều lần."""
        idx = np.arange(len(y_hist))
        in_test = 0
        for seed in range(n_seeds):
            _, te = train_test_split(idx, test_size=test_size, random_state=seed)
            if (y_hist.iloc[te] == 'Fail').any():
                in_test += 1
        return {'n_seeds': n_seeds, 'fail_in_test': in_test, 'fail_in_train': n_seeds - in_test}

    # ------------------------------------------------------------------ Leave-One-Out
    def leave_one_out(self, X_hist, y_hist, text_cols, cat_cols):
        """Mỗi lần bỏ 1 mẫu, huấn luyện trên phần còn lại, dự đoán xác suất Pass của mẫu bị bỏ.
        Mẫu mà việc bỏ đi làm tập huấn luyện còn 1 lớp (TC Fail duy nhất) sẽ có giá trị NaN."""
        builder = PipelineBuilder(text_cols, cat_cols)
        out = pd.DataFrame(np.nan, index=X_hist.index, columns=self.model_names)
        for pos in range(len(X_hist)):
            tr = [j for j in range(len(X_hist)) if j != pos]
            for m in self.model_names:
                try:
                    p = fit_pipeline(builder.build_pipeline(m), m, X_hist.iloc[tr], y_hist.iloc[tr])
                except ValueError:
                    continue
                out.iloc[pos, out.columns.get_loc(m)] = pass_probability(p, X_hist.iloc[[pos]])[0]
        return out

    # ------------------------------------------------------------------ dự đoán tương lai
    @staticmethod
    def predict_future_outcomes(pipeline, X_future, df_future):
        """Dự đoán kết quả tương lai (Pass / Fail) và Xác suất Pass (%) cho các Test Case Block / Chưa test."""
        if len(df_future) == 0 or len(X_future) == 0:
            return df_future.copy()

        df_res = df_future.copy()
        preds = pipeline.predict(X_future)

        classes = list(pipeline.classes_)
        pass_idx = classes.index('Pass') if 'Pass' in classes else 0

        try:
            probs = pipeline.predict_proba(X_future)
            pass_probs = probs[:, pass_idx] * 100
        except Exception:
            pass_probs = np.where(preds == 'Pass', 100.0, 0.0)

        df_res['Dự đoán tương lai (ML)'] = preds
        df_res['Xác suất Pass (%)'] = np.round(pass_probs, 1)

        conditions = [
            df_res['Xác suất Pass (%)'] < 50.0,
            (df_res['Xác suất Pass (%)'] >= 50.0) & (df_res['Xác suất Pass (%)'] < 80.0),
            df_res['Xác suất Pass (%)'] >= 80.0
        ]
        choices = ['🔴 Cao (Nguy cơ Fail)', '🟡 Trung bình', '🟢 Thấp (Dễ Pass)']
        df_res['Mức độ rủi ro'] = np.select(conditions, choices, default='🟡 Trung bình')
        return df_res

    def rank_future_risks(self, final_pipelines, X_future, df_future):
        """Xác suất Pass của từng mô hình và hạng rủi ro trung bình (hạng 1 = xác suất Pass thấp nhất)."""
        df_res = df_future.copy()
        ranks = []
        for m, pipe in final_pipelines.items():
            p = pass_probability(pipe, X_future)
            df_res[f'P(Pass) {m} (%)'] = np.round(p * 100, 2)
            ranks.append(rankdata(p, method='average'))
        df_res['Hạng rủi ro TB (1 = rủi ro cao nhất)'] = np.round(np.mean(ranks, axis=0), 1)
        return df_res.sort_values('Hạng rủi ro TB (1 = rủi ro cao nhất)')

    # ------------------------------------------------------------------ biểu đồ
    def plot_confusion_matrices(self, confusion_matrices, labels=None, output_image='confusion_matrices_comparison.png'):
        """Vẽ biểu đồ Ma trận nhầm lẫn bằng Seaborn."""
        labels = labels or self.labels
        n = max(len(confusion_matrices), 1)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        fig.suptitle('MA TRẬN NHẦM LẪN TRÊN TẬP KIỂM TRA (PASS / FAIL)', fontsize=16, fontweight='bold', y=1.05)

        for ax, (m_name, cm) in zip(axes, confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                        ax=ax, cbar=False, annot_kws={"size": 14, "weight": "bold"})
            ax.set_title(m_name, fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Nhãn Dự Đoán (Predicted)', fontsize=11, fontweight='bold')
            if ax == axes[0]:
                ax.set_ylabel('Nhãn Thực Tế (True)', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        plt.close()
        return output_image


def train_and_evaluate_all(X_hist, y_hist, text_cols, cat_cols, test_size=0.20, random_state=42, hyperparams=None):
    trainer = TestCaseModelTrainer()
    return trainer.train_and_evaluate_all(X_hist, y_hist, text_cols, cat_cols, test_size=test_size, random_state=random_state, hyperparams=hyperparams)

def predict_future_outcomes(pipeline, X_future, df_future):
    return TestCaseModelTrainer.predict_future_outcomes(pipeline, X_future, df_future)

def plot_confusion_matrices(confusion_matrices, labels=LABELS, output_image='confusion_matrices_comparison.png'):
    trainer = TestCaseModelTrainer(labels=labels)
    return trainer.plot_confusion_matrices(confusion_matrices, labels=labels, output_image=output_image)
