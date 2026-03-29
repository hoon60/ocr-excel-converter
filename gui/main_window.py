"""PySide6 메인 윈도우 — 드래그앤드롭 + 진행률 + QThread."""

import json
import os
import subprocess
import sys

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.pipeline import run_pipeline


class OcrWorker(QThread):
    """백그라운드에서 OCR 파이프라인을 실행하는 워커 스레드."""

    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        file_path: str,
        output_path: str | None = None,
        use_ai: bool = False,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
        vision_engine: str = "auto",
    ):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.use_ai = use_ai
        self.groq_api_key = groq_api_key
        self.gemini_api_key = gemini_api_key
        self.vision_engine = vision_engine

    def run(self):
        try:
            result = run_pipeline(
                self.file_path,
                output_path=self.output_path,
                progress_cb=self.progress.emit,
                use_ai=self.use_ai,
                groq_api_key=self.groq_api_key,
                gemini_api_key=self.gemini_api_key,
                vision_engine=self.vision_engine,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DropZone(QLabel):
    """드래그앤드롭 영역."""

    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_idle_style()

    def _set_idle_style(self):
        self.setText(
            "PDF 또는 이미지 파일을\n여기에 드래그하세요\n\n"
            "지원 형식: PDF, PNG, JPG, BMP, TIFF"
        )
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #aaaaaa;
                border-radius: 16px;
                background-color: #f8f9fa;
                color: #555555;
                font-size: 16px;
                padding: 40px;
                min-height: 180px;
            }
            """
        )

    def _set_hover_style(self):
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #4472C4;
                border-radius: 16px;
                background-color: #e8f0fe;
                color: #4472C4;
                font-size: 16px;
                padding: 40px;
                min-height: 180px;
            }
            """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover_style()

    def dragLeaveEvent(self, event):
        self._set_idle_style()

    def dropEvent(self, event: QDropEvent):
        self._set_idle_style()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)


INPUT_STYLE = """
    QLineEdit {
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 12px;
    }
    QLineEdit:focus { border-color: #4472C4; }
"""


class MainWindow(QMainWindow):
    """메인 윈도우."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR → Excel 변환기")
        self.setMinimumSize(560, 560)
        self.worker: OcrWorker | None = None
        self._result_path: str | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 제목
        title = QLabel("OCR → Excel 변환기")
        title.setFont(QFont("맑은 고딕", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 드롭존
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_zone, stretch=1)

        # ── AI Vision 설정 그룹 ──
        ai_group = QGroupBox("AI Vision OCR 설정")
        ai_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px;
                border: 1px solid #ddd; border-radius: 8px;
                margin-top: 8px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px; padding: 0 6px;
            }
        """)
        ai_layout = QGridLayout(ai_group)
        ai_layout.setSpacing(8)

        # AI 사용 체크박스
        self.chk_ai = QCheckBox("AI Vision OCR 사용")
        self.chk_ai.setToolTip("Gemini/Groq Vision AI로 정확한 표 인식")
        self.chk_ai.setStyleSheet("font-size: 13px; font-weight: normal;")
        self.chk_ai.toggled.connect(self._on_ai_toggled)
        ai_layout.addWidget(self.chk_ai, 0, 0, 1, 2)

        # OCR 엔진 선택
        lbl_engine = QLabel("엔진:")
        lbl_engine.setStyleSheet("font-size: 12px; font-weight: normal;")
        self.cmb_engine = QComboBox()
        self.cmb_engine.addItems(["auto (Gemini→Groq)", "gemini", "groq"])
        self.cmb_engine.setStyleSheet("font-size: 12px; padding: 3px;")
        self.cmb_engine.setVisible(False)
        lbl_engine.setVisible(False)
        self._lbl_engine = lbl_engine
        ai_layout.addWidget(lbl_engine, 0, 2)
        ai_layout.addWidget(self.cmb_engine, 0, 3)

        # Gemini API Key
        lbl_gemini = QLabel("Gemini Key:")
        lbl_gemini.setStyleSheet("font-size: 12px; font-weight: normal;")
        self.txt_gemini_key = QLineEdit()
        self.txt_gemini_key.setPlaceholderText("Gemini API Key")
        self.txt_gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_gemini_key.setStyleSheet(INPUT_STYLE)
        self.txt_gemini_key.setVisible(False)
        lbl_gemini.setVisible(False)
        self._lbl_gemini = lbl_gemini
        ai_layout.addWidget(lbl_gemini, 1, 0)
        ai_layout.addWidget(self.txt_gemini_key, 1, 1, 1, 3)

        # Groq API Key
        lbl_groq = QLabel("Groq Key:")
        lbl_groq.setStyleSheet("font-size: 12px; font-weight: normal;")
        self.txt_groq_key = QLineEdit()
        self.txt_groq_key.setPlaceholderText("Groq API Key")
        self.txt_groq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_groq_key.setStyleSheet(INPUT_STYLE)
        self.txt_groq_key.setVisible(False)
        lbl_groq.setVisible(False)
        self._lbl_groq = lbl_groq
        ai_layout.addWidget(lbl_groq, 2, 0)
        ai_layout.addWidget(self.txt_groq_key, 2, 1, 1, 3)

        layout.addWidget(ai_group)

        # 파일 선택 버튼
        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("파일 선택")
        self.btn_browse.setMinimumHeight(40)
        self.btn_browse.setStyleSheet(
            """
            QPushButton {
                background-color: #4472C4;
                color: white;
                border-radius: 8px;
                font-size: 14px;
                padding: 8px 24px;
            }
            QPushButton:hover { background-color: #3561b3; }
            QPushButton:disabled { background-color: #cccccc; }
            """
        )
        self.btn_browse.clicked.connect(self._browse_file)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_browse)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 상태 라벨
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #333; font-size: 13px;")
        layout.addWidget(self.status_label)

        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 6px;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #4472C4;
                border-radius: 5px;
            }
            """
        )
        layout.addWidget(self.progress_bar)

        # 결과 열기 버튼
        self.btn_open = QPushButton("결과 파일 열기")
        self.btn_open.setMinimumHeight(36)
        self.btn_open.setVisible(False)
        self.btn_open.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 8px;
                font-size: 13px;
                padding: 6px 20px;
            }
            QPushButton:hover { background-color: #218838; }
            """
        )
        self.btn_open.clicked.connect(self._open_result)
        layout.addWidget(self.btn_open, alignment=Qt.AlignmentFlag.AlignCenter)

        # config.json에서 API 키 로드
        self._load_config()

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        groq_key = ""
        gemini_key = ""
        use_ai = False
        engine = "auto"

        if os.path.isfile(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                groq_key = cfg.get("groq_api_key", "")
                gemini_key = cfg.get("gemini_api_key", "")
                use_ai = cfg.get("use_ai_by_default", False)
                engine = cfg.get("vision_engine", "auto")
            except (json.JSONDecodeError, OSError):
                pass

        # 환경변수 오버라이드
        groq_key = os.environ.get("GROQ_API_KEY", groq_key)
        gemini_key = os.environ.get("GEMINI_API_KEY", gemini_key)

        if groq_key:
            self.txt_groq_key.setText(groq_key)
        if gemini_key:
            self.txt_gemini_key.setText(gemini_key)

        # 엔진 선택
        engine_map = {"auto": 0, "gemini": 1, "groq": 2}
        self.cmb_engine.setCurrentIndex(engine_map.get(engine, 0))

        if use_ai and (groq_key or gemini_key):
            self.chk_ai.setChecked(True)

    def _save_config(self):
        """현재 설정을 config.json에 저장한다."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        engine_map = {0: "auto", 1: "gemini", 2: "groq"}
        cfg = {
            "groq_api_key": self.txt_groq_key.text().strip(),
            "gemini_api_key": self.txt_gemini_key.text().strip(),
            "use_ai_by_default": self.chk_ai.isChecked(),
            "vision_engine": engine_map.get(self.cmb_engine.currentIndex(), "auto"),
        }
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
        except OSError:
            pass

    def _on_ai_toggled(self, checked: bool):
        for w in (self._lbl_engine, self.cmb_engine,
                  self._lbl_gemini, self.txt_gemini_key,
                  self._lbl_groq, self.txt_groq_key):
            w.setVisible(checked)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "파일 선택",
            "",
            "지원 파일 (*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
            "PDF (*.pdf);;"
            "이미지 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
            "모든 파일 (*)",
        )
        if path:
            self._on_file_dropped(path)

    def _on_file_dropped(self, file_path: str):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "처리 중", "이미 파일을 처리하고 있습니다.")
            return

        self.btn_browse.setEnabled(False)
        self.btn_open.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        filename = os.path.basename(file_path)
        self.status_label.setText(f"처리 중: {filename}")
        self.drop_zone.setText(f"처리 중...\n{filename}")

        use_ai = self.chk_ai.isChecked()
        groq_key = self.txt_groq_key.text().strip() or None
        gemini_key = self.txt_gemini_key.text().strip() or None

        engine_map = {0: "auto", 1: "gemini", 2: "groq"}
        engine = engine_map.get(self.cmb_engine.currentIndex(), "auto")

        if use_ai and not groq_key and not gemini_key:
            QMessageBox.warning(
                self, "API 키 필요",
                "AI Vision OCR을 사용하려면 Gemini 또는 Groq API 키를 입력하세요."
            )
            self.btn_browse.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.drop_zone._set_idle_style()
            return

        # 설정 저장
        self._save_config()

        self.worker = OcrWorker(
            file_path,
            use_ai=use_ai,
            groq_api_key=groq_key,
            gemini_api_key=gemini_key,
            vision_engine=engine,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _on_finished(self, result_path: str):
        self._result_path = result_path
        self.progress_bar.setValue(100)
        self.status_label.setText(f"완료! → {os.path.basename(result_path)}")
        self.btn_open.setVisible(True)
        self.btn_browse.setEnabled(True)
        self.drop_zone._set_idle_style()

        QMessageBox.information(
            self,
            "변환 완료",
            f"Excel 파일이 저장되었습니다:\n{result_path}",
        )

    def _on_error(self, message: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText("오류 발생")
        self.btn_browse.setEnabled(True)
        self.drop_zone._set_idle_style()

        QMessageBox.critical(self, "오류", f"처리 중 오류가 발생했습니다:\n\n{message}")

    def _open_result(self):
        if self._result_path and os.path.isfile(self._result_path):
            if sys.platform == "win32":
                os.startfile(self._result_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", self._result_path])
            else:
                subprocess.run(["xdg-open", self._result_path])
