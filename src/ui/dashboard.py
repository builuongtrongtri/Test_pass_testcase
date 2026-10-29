# src/ui/dashboard.py - Class Streamlit Dashboard OOP

import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from config.settings import DEFAULT_DATA_PATH, TARGET_COL
from src.data.data_loader import DataLoader
from src.models.model_trainer import TestCaseModelTrainer, MODEL_NAMES

class TestCaseDashboardApp:
    """Class quản lý toàn bộ giao diện và luồng ứng dụng Streamlit Dashboard theo kiến trúc OOP."""

    def __init__(self, default_data_path=DEFAULT_DATA_PATH):
        self.default_data_path = default_data_path
        self.trainer = TestCaseModelTrainer()

    @staticmethod
    def render_custom_styles():
        """Cấu hình trang và theme CSS."""
        st.set_page_config(
            page_title="QA TestCase Future Predictor",
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'Inter', -apple-system, sans-serif !important;
            }
            .app-header {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: white;
                padding: 1.5rem 2rem;
                border-radius: 12px;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            }
            .app-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
            .app-subtitle { font-size: 0.95rem; color: #94a3b8; margin-top: 4px; }
            .metric-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 1.2rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
                transition: transform 0.2s ease;
            }
            .metric-card:hover { transform: translateY(-2px); }
            .metric-label { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
            .metric-val { font-size: 1.8rem; font-weight: 700; color: #0f172a; margin: 0.4rem 0 0.2rem; }
            .metric-sub { font-size: 0.8rem; color: #475569; font-weight: 500; }
            .card-wrap {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                margin-bottom: 1.5rem;
            }
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        """Hiển thị Banner tiêu đề ứng dụng."""
        st.markdown("""
        <div class="app-header">
            <div class="app-title">🔮 QA TestCase ML Future Predictor</div>
            <div class="app-subtitle">Hệ thống Học máy Dự đoán Khả năng PASS / FAIL tương lai cho các Test Case đang bị BLOCK & CHƯA TEST từ Dữ liệu Lịch sử</div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Hiển thị bộ điều khiển bên Sidebar."""
        with st.sidebar:
            st.header("⚙️ Nguồn Dữ Liệu & Cấu Hình")
            uploaded_file = st.file_uploader("Tải lên file Excel (.xlsx)", type=["xlsx"])
            data_source = uploaded_file if uploaded_file is not None else self.default_data_path
            
            if uploaded_file is not None:
                st.success("Đã tải lên tệp tùy chỉnh thành công!")
            else:
                st.info(f"Đang dùng tệp mặc định `{self.default_data_path}`")
                
            st.divider()
            st.subheader("🎯 Thử Nghiệm Mô Hình")
            tuning_mode = st.radio(
                "Chế độ áp dụng:",
                options=[
                    "1. Tùy chỉnh Tỷ lệ tập Kiểm tra (Test Ratio)",
                    "2. Tùy chỉnh Siêu tham số Mô hình"
                ],
                help="Khi chọn 1 chế độ, chế độ còn lại sẽ bị VÔ HIỆU HÓA để đảm bảo đánh giá minh bạch."
            )
            
            use_custom_ratio = ("1." in tuning_mode)
            use_custom_params = ("2." in tuning_mode)
            
            st.divider()
            st.subheader("🤖 Chọn Mô Hình ML Dự Đoán")
            selected_model_name = st.selectbox("Mô hình huấn luyện chính:", MODEL_NAMES)
            
            st.markdown("---")
            st.markdown("**1. Tập Kiểm Tra Lịch Sử:**")
            test_size_val = st.slider(
                "Tỷ lệ tập Test Lịch sử", 
                0.1, 0.4, 0.2, 0.05,
                disabled=not use_custom_ratio
            )
            
            st.markdown("**2. Siêu Tham Số Mô Hình:**")
            with st.expander("🛠️ Chi tiết Siêu tham số", expanded=use_custom_params):
                if not use_custom_params:
                    st.caption("🔒 Đang bị VÔ HIỆU HÓA do bạn chọn Chế độ 1. (Chuyển sang Chế độ 2 để mở khóa).")
                    
                lr_c = st.slider("Logistic Regression - Hệ số C:", 0.01, 10.0, 1.0, 0.1, disabled=not use_custom_params)
                rf_nest = st.slider("Random Forest - Số cây:", 50, 300, 100, 25, disabled=not use_custom_params)
                rf_depth = st.slider("Random Forest - Độ sâu max_depth:", 3, 30, 15, disabled=not use_custom_params)
                gb_nest = st.slider("Gradient Boosting - Số cây:", 50, 300, 100, 25, disabled=not use_custom_params)
                gb_lr = st.slider("Gradient Boosting - Tốc độ học lr:", 0.01, 0.5, 0.1, 0.01, disabled=not use_custom_params)
                
            final_test_size = test_size_val if use_custom_ratio else 0.20
            hyperparams = {
                'Logistic Regression': {'C': lr_c},
                'Random Forest': {'n_estimators': rf_nest, 'max_depth': rf_depth},
                'Gradient Boosting': {'n_estimators': gb_nest, 'learning_rate': gb_lr}
            } if use_custom_params else {}
            
            st.button("🔄 Huấn Luyện Lại & Dự Đoán Tương Lai", use_container_width=True)
            
        return data_source, selected_model_name, final_test_size, hyperparams, use_custom_ratio, use_custom_params

    @staticmethod
    def render_overview_tab(df_all, df_hist, df_future, df_future_pred, active_report, selected_model_name, use_custom_ratio, final_test_size):
        """Hiển thị Tab 1: Tổng quan chỉ số & các biểu đồ dự đoán tương lai."""
        if use_custom_ratio:
            st.info(f"🟢 **Chế độ 1 Đang Hoạt Động**: Tỷ lệ tập kiểm tra lịch sử **(Test Ratio = {final_test_size*100:.0f}%)**.")
        else:
            st.success("🟢 **Chế độ 2 Đang Hoạt Động**: Đang áp dụng **Tùy chỉnh Siêu tham số mô hình**.")
            
        total_cases = len(df_all)
        hist_cnt = len(df_hist)
        future_cnt = len(df_future)
        
        hist_pass = (df_hist[TARGET_COL] == 'Pass').sum()
        hist_fail = (df_hist[TARGET_COL] == 'Fail').sum()
        
        pred_pass = (df_future_pred['Dự đoán tương lai (ML)'] == 'Pass').sum() if len(df_future_pred) > 0 else 0
        pred_fail = (df_future_pred['Dự đoán tương lai (ML)'] == 'Fail').sum() if len(df_future_pred) > 0 else 0
        
        acc = active_report.get('accuracy', 0) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng Số Testcase</div><div class="metric-val">{total_cases}</div><div class="metric-sub">Dữ liệu từ Excel</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Dữ Liệu Lịch Sử (Đã test)</div><div class="metric-val" style="color: #2563eb;">{hist_cnt}</div><div class="metric-sub">{hist_pass} Pass | {hist_fail} Fail</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Tương Lai (Block/Chưa test)</div><div class="metric-val" style="color: #d97706;">{future_cnt}</div><div class="metric-sub">Cần dự đoán</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Độ Chính Xác Model</div><div class="metric-val" style="color: #16a34a;">{acc:.1f}%</div><div class="metric-sub">{selected_model_name}</div></div>', unsafe_allow_html=True)
        with col5:
            pred_pass_pct = (pred_pass / future_cnt * 100) if future_cnt > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Dự Đoán Tỷ Lệ Pass</div><div class="metric-val" style="color: #059669;">{pred_pass_pct:.1f}%</div><div class="metric-sub">{pred_pass}/{future_cnt} bài test sẽ Pass</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown('<div class="card-wrap"><b>Phân Phối Dự Đoán Khả Năng Vượt Qua Bài Test (Unblocked Future)</b>', unsafe_allow_html=True)
            if len(df_future_pred) > 0:
                fig_pie = px.pie(
                    values=[pred_pass, pred_fail],
                    names=['Dự đoán PASS khi test', 'Dự đoán FAIL (Cần xem xét)'],
                    color=['Dự đoán PASS khi test', 'Dự đoán FAIL (Cần xem xét)'],
                    color_discrete_map={'Dự đoán PASS khi test': '#22c55e', 'Dự đoán FAIL (Cần xem xét)': '#ef4444'},
                    hole=0.45
                )
                fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
                st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with ch_col2:
            st.markdown('<div class="card-wrap"><b>Dự Đoán Kết Quả Theo Module (Testcases Block/Chưa test)</b>', unsafe_allow_html=True)
            if len(df_future_pred) > 0 and 'module' in df_future_pred.columns:
                mod_df = df_future_pred.groupby(['module', 'Dự đoán tương lai (ML)']).size().reset_index(name='Count')
                fig_mod = px.bar(
                    mod_df, x='module', y='Count', color='Dự đoán tương lai (ML)', barmode='group',
                    color_discrete_map={'Pass': '#22c55e', 'Fail': '#ef4444'}
                )
                fig_mod.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, xaxis_title="Module", yaxis_title="Số lượng Testcase")
                st.plotly_chart(fig_mod, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="card-wrap"><b>Phân Phối Mức Độ Rủi Ro Của Các Test Case Đang Block / Chưa Test</b>', unsafe_allow_html=True)
        if len(df_future_pred) > 0 and 'Mức độ rủi ro' in df_future_pred.columns:
            risk_counts = df_future_pred['Mức độ rủi ro'].value_counts().reset_index()
            risk_counts.columns = ['Mức độ rủi ro', 'Số lượng']
            fig_risk = px.bar(
                risk_counts, x='Mức độ rủi ro', y='Số lượng', color='Mức độ rủi ro',
                color_discrete_map={
                    '🟢 Thấp (Dễ Pass)': '#22c55e',
                    '🟡 Trung bình': '#f59e0b',
                    '🔴 Cao (Nguy cơ Fail)': '#ef4444'
                },
                text_auto=True
            )
            fig_risk.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def render_evaluation_tab(active_report, selected_model_name, train_results, confusion_matrices):
        """Hiển thị Tab 2: Báo cáo đánh giá mô hình và ma trận nhầm lẫn."""
        st.subheader(f"📈 Báo Cáo Đánh Giá Mô Hình Trên Dữ Liệu Lịch Sử: {selected_model_name}")
        rep_df = pd.DataFrame(active_report).transpose()
        st.dataframe(rep_df.style.format(precision=4).background_gradient(cmap="Blues", subset=['precision', 'recall', 'f1-score']), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Ma Trận Nhầm Lẫn (Confusion Matrices) 3 Mô Hình")
        cm_col1, cm_col2, cm_col3 = st.columns(3)
        labels = train_results['labels']
        for idx, (m_name, cm_val) in enumerate(confusion_matrices.items()):
            col_target = [cm_col1, cm_col2, cm_col3][idx]
            with col_target:
                fig_cm = px.imshow(
                    cm_val, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Dự đoán (Predicted)", y="Thực tế (True)", color="Số lượng")
                )
                fig_cm.update_layout(title=dict(text=m_name, font=dict(size=14)), height=300, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_cm, use_container_width=True)

    @staticmethod
    def render_predictions_table_tab(df_future_pred):
        """Hiển thị Tab 3: Bảng danh sách dự đoán tương lai và nút xuất Excel."""
        st.subheader("🔮 Danh Sách Dự Đoán Tương Lai Cho Các Test Case Block & Chưa Test")
        
        if len(df_future_pred) > 0:
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                mod_options = list(df_future_pred['module'].unique()) if 'module' in df_future_pred.columns else []
                mod_filter = st.multiselect("Lọc theo Module:", options=mod_options, default=mod_options) if mod_options else None
            with f_col2:
                risk_options = list(df_future_pred['Mức độ rủi ro'].unique()) if 'Mức độ rủi ro' in df_future_pred.columns else []
                risk_filter = st.multiselect("Lọc theo Mức độ rủi ro:", options=risk_options, default=risk_options) if risk_options else None
            with f_col3:
                pred_options = list(df_future_pred['Dự đoán tương lai (ML)'].unique()) if 'Dự đoán tương lai (ML)' in df_future_pred.columns else []
                pred_filter = st.multiselect("Lọc theo Dự đoán ML:", options=pred_options, default=pred_options) if pred_options else None
                
            filtered_df = df_future_pred.copy()
            if mod_filter and 'module' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['module'].isin(mod_filter)]
            if risk_filter and 'Mức độ rủi ro' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Mức độ rủi ro'].isin(risk_filter)]
            if pred_filter and 'Dự đoán tương lai (ML)' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Dự đoán tương lai (ML)'].isin(pred_filter)]
                
            st.dataframe(filtered_df, use_container_width=True, height=450)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Future_Testcase_Predictions')
                
            st.download_button(
                label="📥 Tải File Excel Kết Quả Dự Đoán Tương Lai (.xlsx)",
                data=output.getvalue(),
                file_name="Future_Blocked_Testcases_Predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.info("Không có test case nào ở trạng thái Block hoặc Chưa test trong dữ liệu.")

    @staticmethod
    def render_single_testcase_sandbox_tab(df_all, active_pipe, text_cols, cat_cols):
        """Hiển thị Tab 4: Form thử nghiệm dự đoán cho 1 test case tùy chỉnh."""
        st.subheader("🎯 Dự Đoán Khả Năng Pass Cho 1 Test Case Block / Mới")
        with st.form("single_testcase_form"):
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                inp_module = st.selectbox("Module:", list(df_all['module'].unique()) if 'module' in df_all.columns else ['Người dùng'])
            with s_col2:
                inp_prio = st.selectbox("Mức độ ưu tiên:", list(df_all['mức độ ưu tiên'].unique()) if 'mức độ ưu tiên' in df_all.columns else ['High'])
            with s_col3:
                inp_user = st.selectbox("Người thực hiện/test:", list(df_all['người thực hiện'].unique()) if 'người thực hiện' in df_all.columns else ['Tester 1'])
                
            inp_name = st.text_input("Tên Test Case:", "Đăng nhập lại trong thời gian đang bị tạm khoá do sai mật khẩu")
            inp_precond = st.text_area("Điều kiện tiền đề:", "Tài khoản khách hàng đang bị khóa tạm thời 15 phút do nhập sai mật khẩu 5 lần")
            inp_steps = st.text_area("Bước thực hiện:", "1. Mở màn hình đăng nhập\n2. Nhập username & mật khẩu đúng\n3. Bấm Đăng nhập")
            inp_testdata = st.text_area("Dữ liệu test (nếu có):", "")
            inp_expected = st.text_area("Kết quả mong đợi (nếu có):", "")
            
            submit_pred = st.form_submit_button("🔮 Dự Đoán Khả Năng Pass", use_container_width=True)
            
        if submit_pred:
            single_raw = pd.DataFrame({
                'module': [inp_module],
                'mức độ ưu tiên': [inp_prio],
                'người thực hiện': [inp_user],
                'tên testcase': [inp_name],
                'điều kiện tiền đề': [inp_precond],
                'bước thực hiện': [inp_steps],
                'dữ liệu test': [inp_testdata],
                'kết quả mong đợi': [inp_expected]
            })
            
            single_df = DataLoader.ensure_model_features(single_raw, text_cols, cat_cols)
            
            pred_label = active_pipe.predict(single_df)[0]
            try:
                pred_probs = active_pipe.predict_proba(single_df)[0]
                prob_dict = dict(zip(active_pipe.classes_, pred_probs))
                pass_prob = prob_dict.get('Pass', 0.0) * 100
            except Exception:
                pass_prob = 100.0 if pred_label == 'Pass' else 0.0
                
            st.divider()
            st.markdown("### Kết Quả Dự Đoán Tương Lai:")
            if pred_label == 'Pass':
                st.markdown(f'<div style="font-size: 1.4rem; color: #15803d; background: #dcfce7; padding: 1.2rem; border-radius: 8px; font-weight: bold; text-align: center;">✅ Dự đoán: PASS khi test trong tương lai (Xác suất Pass: {pass_prob:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size: 1.4rem; color: #b91c1c; background: #fee2e2; padding: 1.2rem; border-radius: 8px; font-weight: bold; text-align: center;">❌ Dự đoán: FAIL (Nguy cơ cao bị lỗi khi unblock test - Xác suất Pass chỉ {pass_prob:.1f}%)</div>', unsafe_allow_html=True)

    def run(self):
        """Hàm thực thi chính chạy ứng dụng Dashboard Streamlit."""
        self.render_custom_styles()
        self.render_header()
        
        data_source, selected_model_name, final_test_size, hyperparams, use_custom_ratio, use_custom_params = self.render_sidebar()
        
        try:
            loader = DataLoader(data_source)
            data = loader.load_and_clean_data()
            X_hist = data['X_history']
            y_hist = data['y_history']
            df_hist = data['df_history']
            X_future = data['X_future']
            df_future = data['df_future']
            df_all = data['df_all']
            text_cols = data['text_cols']
            cat_cols = data['cat_cols']
        except Exception as e:
            st.error(f"Lỗi đọc dữ liệu: {e}")
            st.stop()

        train_results = self.trainer.train_and_evaluate_all(
            X_hist, y_hist, text_cols, cat_cols, 
            test_size=final_test_size, 
            hyperparams=hyperparams
        )

        pipelines = train_results.get('final_pipelines', train_results['pipelines'])
        reports = train_results['reports']
        confusion_matrices = train_results['confusion_matrices']

        active_pipe = pipelines[selected_model_name]
        active_report = reports.get(selected_model_name) or {}

        df_future_pred = self.trainer.predict_future_outcomes(active_pipe, X_future, df_future)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Tổng Quan & Dự Đoán Tương Lai", 
            "📈 Đánh Giá Mô Hình Trên Dữ Liệu Cũ", 
            "🔮 Danh Sách Block / Chưa Test Kèm Dự Đoán ML", 
            "🎯 Dự Đoán 1 Testcase Mới"
        ])

        with tab1:
            self.render_overview_tab(df_all, df_hist, df_future, df_future_pred, active_report, selected_model_name, use_custom_ratio, final_test_size)
        with tab2:
            self.render_evaluation_tab(active_report, selected_model_name, train_results, confusion_matrices)
        with tab3:
            self.render_predictions_table_tab(df_future_pred)
        with tab4:
            self.render_single_testcase_sandbox_tab(df_all, active_pipe, text_cols, cat_cols)
