# app.py - Streamlit Dashboard Entry Point

from src.ui.dashboard import TestCaseDashboardApp

def main():
    app = TestCaseDashboardApp()
    app.run()

if __name__ == '__main__':
    main()
