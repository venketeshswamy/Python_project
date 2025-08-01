#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-Powered Data Snipping Tool
A robust, asynchronous desktop application for extracting structured data from
PDF and image documents using multiple OCR engines.

Author: Gemini
Version: 1.0.0
Python Requirement: 3.9+
Key Libraries: PyQt6, PyMuPDF, Pillow, pytesseract, pywinrt, pandas, openpyxl
"""

import sys
import os
import json
import shutil
import zipfile
import platform
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# --- Core GUI Framework ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QTableWidget, QGraphicsView, QGraphicsScene, QMenuBar,
    QToolBar, QStatusBar, QFileDialog, QMessageBox, QProgressBar, QLineEdit,
    QPushButton, QComboBox, QLabel, QGraphicsRectItem, QGraphicsItem,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QMenu, QRubberBand,
    QDialog, QCheckBox, QDialogButtonBox, QListWidgetItem, QInputDialog, QStyleFactory, QGraphicsPixmapItem
)
from PyQt6.QtGui import (
    QAction, QPixmap, QImage, QPainter, QColor, QPen, QBrush, QKeySequence,
    QIcon, QFont, QPainterPath, QTransform
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QRectF, QPointF, QSize, QDir, QFileInfo,
    QPoint, QSettings
)

# --- Document & Image Handling ---
import fitz  # PyMuPDF
from PIL import Image, ImageQt

# --- OCR Engines ---
import pytesseract
if platform.system() == "Windows":
    try:
        # pywinrt is required for Windows OCR
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
        import asyncio
        WINDOWS_OCR_AVAILABLE = True
    except ImportError:
        WINDOWS_OCR_AVAILABLE = False
else:
    WINDOWS_OCR_AVAILABLE = False

# --- Data Handling ---
import pandas as pd

# --- Constants ---
APP_NAME = "AI Data Snipping Tool"
APP_VERSION = "1.0.0"
CONFIG_FILE = "config.json"
SESSION_EXTENSION = ".ocr-session"
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
SUPPORTED_FORMATS = ['.pdf'] + SUPPORTED_IMAGE_FORMATS

DEFAULT_CONFIG = {
    "poppler_path": "C:\\path\\to\\poppler\\bin" if platform.system() == "Windows" else "/usr/bin",
    "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" if platform.system() == "Windows" else "tesseract",
    "workspace_dir": "ocr_tool_workspace",
    "default_ocr_engine": "windows" if WINDOWS_OCR_AVAILABLE else "tesseract",
    "clear_cache_on_startup": False,
    "ocr_dpi": 300,
    "pdf_preview_dpi": 150,
    "windows_ocr_lang": "en-US",
    "tesseract_lang": "eng",
    "tesseract_psm": "3",
    "cache_dir": "ocr_tool_workspace/cache",
    "imports_dir": "ocr_tool_workspace/imports",
    "exports_dir": "ocr_tool_workspace/exports"
}

# --- Helper Classes & Functions ---


class ConfigManager:
    """Handles loading, creating, and managing the application's configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}

    def load_config(self) -> Dict[str, Any]:
        """Loads config from JSON file, creating it if it doesn't exist."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(
                f"'{self.config_path}' not found or invalid. Creating default config.")
            self.config = DEFAULT_CONFIG
            self.save_config()

        self._validate_and_create_dirs()
        self._apply_ocr_config()
        return self.config

    def save_config(self):
        """Saves the current configuration to the JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            print(f"Error saving config file: {e}")

    def _validate_and_create_dirs(self):
        """Ensures all necessary workspace directories exist."""
        for key in ["workspace_dir", "cache_dir", "imports_dir", "exports_dir"]:
            dir_path = self.config.get(key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    def _apply_ocr_config(self):
        """Applies configuration settings to OCR engines."""
        if platform.system() == "Windows":
            # Set Poppler path for PDF to image conversion if needed by Tesseract
            poppler_path = self.config.get("poppler_path")
            if poppler_path and os.path.isdir(poppler_path):
                os.environ["PATH"] += os.pathsep + poppler_path

        # Set Tesseract command
        tesseract_cmd = self.config.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key."""
        return self.config.get(key, default)


class LRUCache:
    """A simple Least Recently Used (LRU) Cache implementation."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.cache: Dict[Any, Any] = {}
        self.order: List[Any] = []

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def set(self, key: Any, value: Any):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.order.append(key)

    def clear(self):
        self.cache.clear()
        self.order.clear()

# --- Worker Threads for Asynchronous Operations ---


class WorkerSignals(QObject):
    """Defines signals available from a running worker thread."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class FileImportWorker(QThread):
    """Worker thread for importing files asynchronously."""
    file_imported = pyqtSignal(
        str, str, int)  # original_path, imported_path, num_pages

    def __init__(self, file_paths: List[str], import_dir: str):
        super().__init__()
        self.file_paths = file_paths
        self.import_dir = import_dir
        self.is_interrupted = False

    def run(self):
        for i, file_path in enumerate(self.file_paths):
            if self.is_interrupted:
                break
            try:
                if not QFileInfo(file_path).exists():
                    continue

                # Copy file to the workspace to make sessions self-contained
                file_name = os.path.basename(file_path)
                imported_path = os.path.join(self.import_dir, file_name)

                # Handle potential name collisions
                counter = 1
                while os.path.exists(imported_path):
                    name, ext = os.path.splitext(file_name)
                    imported_path = os.path.join(
                        self.import_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy(file_path, imported_path)

                # Determine page count while still in the worker thread
                num_pages = 0
                try:
                    ext = os.path.splitext(imported_path)[1].lower()
                    if ext == '.pdf':
                        doc = fitz.open(imported_path)
                        num_pages = len(doc)
                        doc.close()
                    elif ext in SUPPORTED_IMAGE_FORMATS:
                        num_pages = 1
                except Exception as e:
                    print(f"Error counting pages in {imported_path}: {e}")

                self.file_imported.emit(file_path, imported_path, num_pages)
            except Exception as e:
                print(f"Error importing {file_path}: {e}")
        self.finished.emit()

    def stop(self):
        self.is_interrupted = True


class PageRenderWorker(QThread):
    """Worker thread for rendering document pages."""
    page_rendered = pyqtSignal(QPixmap)
    error = pyqtSignal(str)

    def __init__(self, file_path: str, page_num: int, cache_dir: str, memory_cache: LRUCache, dpi: int):
        super().__init__()
        self.file_path = file_path
        self.page_num = page_num
        self.cache_dir = cache_dir
        self.memory_cache = memory_cache
        self.dpi = dpi
        self.cache_key = (file_path, page_num, dpi)

    def run(self):
        try:
            # 1. Check memory cache
            pixmap = self.memory_cache.get(self.cache_key)
            if pixmap:
                self.page_rendered.emit(pixmap)
                return

            # 2. Check disk cache
            cached_image_path = self._get_cached_image_path()
            if os.path.exists(cached_image_path):
                pixmap = QPixmap(cached_image_path)
                if not pixmap.isNull():
                    self.memory_cache.set(self.cache_key, pixmap)
                    self.page_rendered.emit(pixmap)
                    return

            # 3. Render from source
            pixmap = self._render_from_source()
            if pixmap and not pixmap.isNull():
                self.memory_cache.set(self.cache_key, pixmap)
                pixmap.save(cached_image_path, "PNG")
                self.page_rendered.emit(pixmap)
            else:
                self.error.emit(
                    f"Failed to render page {self.page_num + 1} from {os.path.basename(self.file_path)}.")

        except Exception as e:
            self.error.emit(f"Error rendering page: {e}")

    def _get_cached_image_path(self) -> str:
        """Generates a unique path for the cached page image."""
        file_hash = hash(self.file_path)
        return os.path.join(self.cache_dir, f"{file_hash}_{self.page_num}_{self.dpi}.png")

    def _render_from_source(self) -> Optional[QPixmap]:
        """Renders a page from the original PDF or image file."""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.pdf':
            doc = fitz.open(self.file_path)
            if 0 <= self.page_num < len(doc):
                page = doc.load_page(self.page_num)
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                image = QImage(pix.samples, pix.width, pix.height,
                               pix.stride, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(image)
            doc.close()
        elif ext in SUPPORTED_IMAGE_FORMATS:
            if self.page_num == 0:  # Images have only one page
                image = Image.open(self.file_path)
                # Ensure image has 3 channels (RGB) for consistency
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                qimage = ImageQt.ImageQt(image)
                return QPixmap.fromImage(qimage)
        return None


class OcrWorker(QThread):
    """Worker thread for performing OCR on specified regions."""
    result_ready = pyqtSignal(dict)  # {file_path, page_num, snip_name, text}
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, tasks: List[Dict], config: Dict[str, Any]):
        super().__init__()
        self.tasks = tasks
        self.config = config
        self.is_interrupted = False

        if WINDOWS_OCR_AVAILABLE:
            try:
                self.win_ocr_engine = OcrEngine.try_create_from_language(
                    Language(config.get("windows_ocr_lang", "en-US")))
            except Exception as e:
                self.win_ocr_engine = None
                print(f"Could not initialize Windows OCR Engine: {e}")
        else:
            self.win_ocr_engine = None

    def run(self):
        total_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            if self.is_interrupted:
                break

            self.progress.emit(int((i / total_tasks) * 100),
                               f"Processing snip '{task['snip_name']}' on page {task['page_num']+1}...")

            try:
                file_path = task['file_path']
                page_num = task['page_num']
                snip_rect = task['snip_rect']  # QRectF
                snip_name = task['snip_name']
                ocr_engine = task['ocr_engine']

                text = ""
                if ocr_engine == 'pdf_native':
                    text = self._run_pdf_native(file_path, page_num, snip_rect)
                    # Fallback to Tesseract if native extraction fails
                    if not text or text.isspace():
                        text = self._run_tesseract(
                            file_path, page_num, snip_rect)
                elif ocr_engine == 'tesseract':
                    text = self._run_tesseract(file_path, page_num, snip_rect)
                elif ocr_engine == 'windows' and self.win_ocr_engine:
                    text = self._run_windows_ocr(
                        file_path, page_num, snip_rect)
                else:
                    self.error.emit(
                        f"Unsupported or unavailable OCR engine: {ocr_engine}")
                    continue

                self.result_ready.emit({
                    'file_path': file_path,
                    'page_num': page_num,
                    'snip_name': snip_name,
                    'text': text.strip()
                })

            except Exception as e:
                self.error.emit(
                    f"OCR failed for snip '{task.get('snip_name', 'Unknown')}': {e}")

        self.progress.emit(100, "OCR complete.")
        self.finished.emit()

    def _get_cropped_image(self, file_path: str, page_num: int, snip_rect: QRectF) -> Optional[Image.Image]:
        """Extracts and returns a cropped PIL image for a snip region."""
        dpi = self.config.get('ocr_dpi', 300)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                doc = fitz.open(file_path)
                if not (0 <= page_num < len(doc)):
                    return None
                page = doc.load_page(page_num)

                # Convert Qt rect to fitz rect
                page_width, page_height = page.rect.width, page.rect.height
                fitz_rect = fitz.Rect(
                    snip_rect.x() * page_width,
                    snip_rect.y() * page_height,
                    snip_rect.right() * page_width,
                    snip_rect.bottom() * page_height
                )

                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, clip=fitz_rect, alpha=False)
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                return img

            elif ext in SUPPORTED_IMAGE_FORMATS:
                if page_num != 0:
                    return None
                img = Image.open(file_path)

                # Convert normalized rect to pixel coordinates
                width, height = img.size
                crop_box = (
                    int(snip_rect.x() * width),
                    int(snip_rect.y() * height),
                    int(snip_rect.right() * width),
                    int(snip_rect.bottom() * height)
                )
                return img.crop(crop_box)

        except Exception as e:
            self.error.emit(f"Failed to crop image for OCR: {e}")
            return None
        return None

    def _run_pdf_native(self, file_path: str, page_num: int, snip_rect: QRectF) -> str:
        """Extracts text directly from a PDF's text layer."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext != '.pdf':
            return ""

        try:
            doc = fitz.open(file_path)
            if not (0 <= page_num < len(doc)):
                return ""
            page = doc.load_page(page_num)

            page_width, page_height = page.rect.width, page.rect.height
            fitz_rect = fitz.Rect(
                snip_rect.x() * page_width,
                snip_rect.y() * page_height,
                snip_rect.right() * page_width,
                snip_rect.bottom() * page_height
            )

            text = page.get_text("text", clip=fitz_rect, sort=True)
            doc.close()
            return text
        except Exception as e:
            self.error.emit(f"PDF native text extraction failed: {e}")
            return ""

    def _run_tesseract(self, file_path: str, page_num: int, snip_rect: QRectF) -> str:
        """Runs Tesseract OCR on a cropped image region."""
        img = self._get_cropped_image(file_path, page_num, snip_rect)
        if not img:
            return ""

        try:
            lang = self.config.get('tesseract_lang', 'eng')
            psm = self.config.get('tesseract_psm', '3')
            custom_config = f'--oem 3 --psm {psm}'
            text = pytesseract.image_to_string(
                img, lang=lang, config=custom_config)
            return text
        except pytesseract.TesseractNotFoundError:
            self.error.emit(
                "Tesseract not found. Check 'tesseract_cmd' in config.json.")
            return ""
        except Exception as e:
            self.error.emit(f"Tesseract OCR failed: {e}")
            return ""

    def _run_windows_ocr(self, file_path: str, page_num: int, snip_rect: QRectF) -> str:
        """Runs native Windows OCR on a cropped image region."""
        if not WINDOWS_OCR_AVAILABLE or not self.win_ocr_engine:
            self.error.emit("Windows OCR is not available on this system.")
            return ""

        img = self._get_cropped_image(file_path, page_num, snip_rect)
        if not img:
            return ""

        try:
            # Convert PIL image to a format Windows OCR can understand
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            width, height = img.size
            img_bytes = img.tobytes()

            # This must be run in an async context
            return asyncio.run(self._perform_win_ocr(img_bytes, width, height))

        except Exception as e:
            self.error.emit(f"Windows OCR failed: {e}")
            return ""

    async def _perform_win_ocr(self, img_bytes: bytes, width: int, height: int) -> str:
        """Helper async function to call the Windows OCR API."""
        try:
            software_bitmap = SoftwareBitmap(
                BitmapPixelFormat.RGBA8, width, height)
            software_bitmap.copy_from_buffer(img_bytes)

            ocr_result = await self.win_ocr_engine.recognize_async(software_bitmap)
            return ocr_result.text
        except Exception as e:
            # Cannot emit signal from here, so we print
            print(f"Async Windows OCR call failed: {e}")
            return ""

    def stop(self):
        self.is_interrupted = True

# --- Custom GUI Widgets ---


class SnipItem(QGraphicsRectItem):
    """A movable, resizable rectangle representing a snip on the canvas."""
    handle_size = 8.0
    handle_space = 4.0

    # Signals
    geometry_changed = pyqtSignal()

    def __init__(self, rect: QRectF, color: QColor, name: str):
        super().__init__(rect)
        self.name = name
        self.color = color
        self.handles = {}
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.update_handles()

    def set_color(self, color: QColor):
        self.color = color
        self.update()

    def handle_at(self, point: QPointF) -> Optional[int]:
        """Returns the handle index at a given point."""
        for i, handle_rect in self.handles.items():
            if handle_rect.contains(point):
                return i
        return None

    def hoverMoveEvent(self, event):
        """Adjusts cursor shape when hovering over handles."""
        handle = self.handle_at(event.pos())
        if handle is not None:
            if handle in [0, 5]:  # Top-left, Bottom-right
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif handle in [2, 7]:  # Top-right, Bottom-left
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif handle in [1, 6]:  # Top, Bottom
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:  # Left, Right
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """Selects a handle for resizing."""
        self.handle_selected = self.handle_at(event.pos())
        if self.handle_selected is not None:
            self.mouse_press_pos = event.pos()
            self.mouse_press_rect = self.rect()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Resizes the rectangle based on the selected handle's movement."""
        if self.handle_selected is not None:
            self.interactive_resize(event.pos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finalizes resizing and emits a signal."""
        super().mouseReleaseEvent(event)
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None
        self.update()
        self.geometry_changed.emit()

    def interactive_resize(self, mouse_pos: QPointF):
        """Calculates new rectangle geometry during resizing."""
        rect = self.rect()
        if self.mouse_press_rect is None:
            return

        diff = mouse_pos - self.mouse_press_pos
        self.prepareGeometryChange()

        if self.handle_selected == 0:  # Top-left
            rect.setTopLeft(self.mouse_press_rect.topLeft() + diff)
        elif self.handle_selected == 1:  # Top
            rect.setTop(self.mouse_press_rect.top() + diff.y())
        elif self.handle_selected == 2:  # Top-right
            rect.setTopRight(self.mouse_press_rect.topRight() + diff)
        elif self.handle_selected == 3:  # Right
            rect.setRight(self.mouse_press_rect.right() + diff.x())
        elif self.handle_selected == 4:  # Bottom-right
            rect.setBottomRight(self.mouse_press_rect.bottomRight() + diff)
        elif self.handle_selected == 5:  # Bottom
            rect.setBottom(self.mouse_press_rect.bottom() + diff.y())
        elif self.handle_selected == 6:  # Bottom-left
            rect.setBottomLeft(self.mouse_press_rect.bottomLeft() + diff)
        elif self.handle_selected == 7:  # Left
            rect.setLeft(self.mouse_press_rect.left() + diff.x())

        self.setRect(rect.normalized())
        self.update_handles()

    def update_handles(self):
        """Updates the positions of the resize handles."""
        s = self.handle_size
        r = self.rect()
        self.handles[0] = QRectF(r.left(), r.top(), s, s)  # Top-left
        self.handles[1] = QRectF(r.center().x() - s/2, r.top(), s, s)  # Top
        self.handles[2] = QRectF(r.right() - s, r.top(), s, s)  # Top-right
        self.handles[3] = QRectF(
            r.right() - s, r.center().y() - s/2, s, s)  # Right
        self.handles[4] = QRectF(
            r.right() - s, r.bottom() - s, s, s)  # Bottom-right
        self.handles[5] = QRectF(r.center().x() - s/2,
                                 r.bottom() - s, s, s)  # Bottom
        self.handles[6] = QRectF(r.left(), r.bottom() - s, s, s)  # Bottom-left
        self.handles[7] = QRectF(r.left(), r.center().y() - s/2, s, s)  # Left

    def itemChange(self, change, value):
        """Emits signal when item is moved."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.geometry_changed.emit()
        return super().itemChange(change, value)

    def boundingRect(self) -> QRectF:
        """Returns the bounding rect including handles."""
        o = self.handle_size + self.handle_space
        return self.rect().adjusted(-o, -o, o, o)

    def paint(self, painter: QPainter, option, widget=None):
        """Paints the snip rectangle and its handles."""
        pen = QPen(self.color, 2, Qt.PenStyle.SolidLine)
        if self.isSelected():
            pen.setStyle(Qt.PenStyle.DashLine)

        painter.setPen(pen)
        painter.setBrush(
            QBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 50)))
        painter.drawRect(self.rect())

        # Draw handles if selected
        if self.isSelected():
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor("white")))
            painter.setPen(QPen(QColor("black"), 1.0, Qt.PenStyle.SolidLine))
            for i, rect in self.handles.items():
                painter.drawEllipse(rect)


class InteractiveGraphicsView(QGraphicsView):
    """A QGraphicsView that supports zooming, panning, and drawing snips."""
    new_snip_drawn = pyqtSignal(QRectF)

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.is_drawing_snip = False

    def set_pixmap(self, pixmap: QPixmap):
        """Clears the scene and displays a new pixmap."""
        self.scene().clear()
        if not pixmap.isNull():
            self.scene().addPixmap(pixmap)
            # Use a QGraphicsRectItem to define the scene rect based on the image
            # This ensures snip coordinates are normalized relative to the image
            self.scene().setSceneRect(0, 0, pixmap.width(), pixmap.height())
        else:
            self.scene().setSceneRect(QRectF())

    def wheelEvent(self, event):
        """Handles zooming with the mouse wheel."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Starts drawing a snip or panning."""
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.is_drawing_snip = True
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.is_drawing_snip = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Updates the rubber band while drawing."""
        if self.is_drawing_snip:
            self.rubber_band.setGeometry(
                QRect(self.origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finishes drawing a snip."""
        if self.is_drawing_snip and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing_snip = False
            self.rubber_band.hide()

            rect_in_view = self.rubber_band.geometry()
            rect_in_scene = self.mapToScene(rect_in_view).boundingRect()

            # Normalize coordinates relative to the scene (image) size
            scene_rect = self.scene().sceneRect()
            if scene_rect.width() > 0 and scene_rect.height() > 0:
                normalized_rect = QRectF(
                    rect_in_scene.x() / scene_rect.width(),
                    rect_in_scene.y() / scene_rect.height(),
                    rect_in_scene.width() / scene_rect.width(),
                    rect_in_scene.height() / scene_rect.height()
                )
                self.new_snip_drawn.emit(normalized_rect)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mouseReleaseEvent(event)

    def fit_to_page(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_width(self):
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatioByExpanding)
            # This can be tricky, let's refine it.
            view_rect = self.viewport().rect()
            scene_rect = self.sceneRect()
            if scene_rect.width() == 0:
                return

            scale = view_rect.width() / scene_rect.width()
            self.setTransform(QTransform().scale(scale, scale))


class FilterDialog(QDialog):
    """A dialog for filtering table columns, similar to Excel."""

    def __init__(self, column_name: str, unique_values: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Filter {column_name}")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.list_widget)

        for value in sorted(unique_values):
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_widget.addItem(item)

        select_all_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self._select_all)
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(self._deselect_all)
        select_all_layout.addWidget(self.select_all_button)
        select_all_layout.addWidget(self.deselect_all_button)
        layout.addLayout(select_all_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _filter_list(self, text: str):
        """Hides/shows items in the list based on search text."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def get_selected_values(self) -> List[str]:
        """Returns a list of the checked values."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

# --- Main Application Window ---


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)

        # --- Load Config and Setup Workspace ---
        self.config_manager = ConfigManager(CONFIG_FILE)
        self.config = self.config_manager.load_config()
        self.settings = QSettings("MyCompany", APP_NAME)  # For window state

        # --- Initialize Core Components ---
        self.memory_cache = LRUCache(capacity=20)
        if self.config.get("clear_cache_on_startup", False):
            self._clear_disk_cache()

        # --- Data Models ---
        # {imported_path: {orig_path, num_pages}}
        self.imported_files: Dict[str, Dict] = {}
        # {snip_name: {rect: QRectF, color: QColor, item: SnipItem}}
        self.snips: Dict[str, Dict] = {}
        self.ocr_results = pd.DataFrame()
        self.column_filters: Dict[str, List[str]] = {}

        # --- State Variables ---
        self.current_file_path: Optional[str] = None
        self.current_page_num: int = 0
        self.is_bw_mode: bool = False

        # --- Worker Threads ---
        self.import_worker: Optional[FileImportWorker] = None
        self.render_worker: Optional[PageRenderWorker] = None
        self.ocr_worker: Optional[OcrWorker] = None

        self._setup_ui()
        self._connect_signals()

        self.resize(self.settings.value("size", QSize(1280, 800)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
        splitter_state = self.settings.value("splitter_state")
        if splitter_state:
            self.main_splitter.restoreState(splitter_state)

    def _setup_ui(self):
        """Initializes all UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._create_actions()
        self._create_menubar()
        self._create_toolbar()
        self._create_statusbar()

        # --- Main Layout Panes ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.main_splitter)

        self._create_left_pane()
        self._create_center_pane()
        self._create_right_pane()

        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(self.center_pane)
        self.main_splitter.addWidget(self.right_pane)

        self.main_splitter.setSizes([250, 600, 350])  # Initial proportions

    def _create_actions(self):
        """Creates all QAction objects for menus and toolbars."""
        # File Actions
        self.import_files_action = QAction(QIcon.fromTheme(
            "document-open"), "&Import Files...", self)
        self.import_folder_action = QAction(
            QIcon.fromTheme("folder-open"), "Import &Folder...", self)
        self.save_session_action = QAction(QIcon.fromTheme(
            "document-save-as"), "&Save Session As...", self)
        self.load_session_action = QAction(QIcon.fromTheme(
            "document-open"), "&Load Session...", self)
        self.export_csv_action = QAction(QIcon.fromTheme(
            "document-save"), "Export to &CSV...", self)
        self.export_excel_action = QAction(QIcon.fromTheme(
            "x-office-spreadsheet"), "Export to E&xcel...", self)
        self.exit_action = QAction(QIcon.fromTheme(
            "application-exit"), "E&xit", self)

        # Edit Actions
        self.clear_snips_action = QAction("Clear All &Snips", self)
        self.clear_files_action = QAction("Clear All Imported &Files", self)
        self.clear_results_action = QAction("Clear All OCR &Results", self)
        self.prefs_action = QAction(QIcon.fromTheme(
            "preferences-system"), "Application &Preferences...", self)

        # View Actions
        self.zoom_in_action = QAction(
            QIcon.fromTheme("zoom-in"), "Zoom &In", self)
        self.zoom_out_action = QAction(
            QIcon.fromTheme("zoom-out"), "Zoom &Out", self)
        self.fit_page_action = QAction(QIcon.fromTheme(
            "zoom-fit-best"), "&Fit to Page", self)
        self.fit_width_action = QAction(QIcon.fromTheme(
            "zoom-fit-width"), "Fit to &Width", self)
        self.bw_toggle_action = QAction(
            "Display Black & White", self, checkable=True)

        # Set shortcuts
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)

    def _create_menubar(self):
        menubar = self.menuBar()
        # File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.import_files_action)
        file_menu.addAction(self.import_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_session_action)
        file_menu.addAction(self.load_session_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_csv_action)
        file_menu.addAction(self.export_excel_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.clear_snips_action)
        edit_menu.addAction(self.clear_files_action)
        edit_menu.addAction(self.clear_results_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.prefs_action)

        # View Menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.fit_page_action)
        view_menu.addAction(self.fit_width_action)
        view_menu.addSeparator()
        view_menu.addAction(self.bw_toggle_action)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("OCR Engine:"))
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItem("PDF Native", "pdf_native")
        self.ocr_engine_combo.addItem("Tesseract", "tesseract")
        if WINDOWS_OCR_AVAILABLE:
            self.ocr_engine_combo.addItem("Windows OCR", "windows")
        default_engine = self.config.get("default_ocr_engine")
        index = self.ocr_engine_combo.findData(default_engine)
        if index != -1:
            self.ocr_engine_combo.setCurrentIndex(index)
        toolbar.addWidget(self.ocr_engine_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Snipping Mode:"))
        self.snip_mode_combo = QComboBox()
        self.snip_mode_combo.addItem("Single Page Snip", "single")
        self.snip_mode_combo.addItem("Template Snip", "template")
        toolbar.addWidget(self.snip_mode_combo)

        toolbar.addSeparator()

        self.run_ocr_button = QPushButton(
            QIcon.fromTheme("media-playback-start"), "Run OCR")
        self.stop_ocr_button = QPushButton(
            QIcon.fromTheme("media-playback-stop"), "Stop OCR")
        self.stop_ocr_button.setEnabled(False)
        toolbar.addWidget(self.run_ocr_button)
        toolbar.addWidget(self.stop_ocr_button)

    def _create_statusbar(self):
        self.status_bar = self.statusBar()
        self.status_file_label = QLabel("No file loaded")
        self.status_page_label = QLabel()
        self.status_zoom_label = QLabel()
        self.status_coords_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.status_bar.addPermanentWidget(self.status_file_label, 1)
        self.status_bar.addPermanentWidget(self.status_page_label)
        self.status_bar.addPermanentWidget(self.status_zoom_label)
        self.status_bar.addPermanentWidget(self.status_coords_label)
        self.status_bar.addWidget(self.progress_bar)

    def _create_left_pane(self):
        self.left_pane = QWidget()
        layout = QVBoxLayout(self.left_pane)

        # Files List
        layout.addWidget(QLabel("Imported Files:"))
        self.files_list_widget = QListWidget()
        self.files_list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.files_list_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self.files_list_widget)

        # Template Snip Page Range
        template_group = QWidget()
        template_layout = QHBoxLayout(template_group)
        template_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.addWidget(QLabel("Page Range (for Template):"))
        self.page_range_edit = QLineEdit()
        self.page_range_edit.setPlaceholderText("e.g., 1, 3-5, 10")
        template_layout.addWidget(self.page_range_edit)
        layout.addWidget(template_group)

        # Snips Table
        layout.addWidget(QLabel("Defined Snips:"))
        self.snips_table = QTableWidget()
        self.snips_table.setColumnCount(2)
        self.snips_table.setHorizontalHeaderLabels(["Snip Name", "Color"])
        self.snips_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.snips_table.verticalHeader().setVisible(False)
        self.snips_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.snips_table)

    def _create_center_pane(self):
        self.center_pane = QWidget()
        layout = QVBoxLayout(self.center_pane)

        # Navigation controls
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.first_page_button = QPushButton("<< First")
        self.prev_page_button = QPushButton("< Prev")
        self.page_nav_edit = QLineEdit()
        self.page_nav_edit.setFixedWidth(50)
        self.page_nav_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_page_button = QPushButton("Next >")
        self.last_page_button = QPushButton("Last >>")
        nav_layout.addStretch()
        nav_layout.addWidget(self.first_page_button)
        nav_layout.addWidget(self.prev_page_button)
        nav_layout.addWidget(self.page_nav_edit)
        nav_layout.addWidget(self.next_page_button)
        nav_layout.addWidget(self.last_page_button)
        nav_layout.addStretch()
        layout.addWidget(nav_widget)

        # Graphics View
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = InteractiveGraphicsView(self.graphics_scene)
        layout.addWidget(self.graphics_view)

        # Info label
        info_label = QLabel(
            "<b>Hint:</b> Hold <b>Shift</b> and drag to draw a snip. Select snips to move/resize.")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

    def _create_right_pane(self):
        self.right_pane = QWidget()
        layout = QVBoxLayout(self.right_pane)
        layout.addWidget(QLabel("OCR Results:"))

        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        self.results_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self.results_table)

    def _connect_signals(self):
        """Connects all signals to their corresponding slots."""
        # File Menu
        self.import_files_action.triggered.connect(self._prompt_import_files)
        self.import_folder_action.triggered.connect(self._prompt_import_folder)
        self.save_session_action.triggered.connect(self._save_session)
        self.load_session_action.triggered.connect(self._load_session)
        self.export_csv_action.triggered.connect(
            lambda: self._export_results('csv'))
        self.export_excel_action.triggered.connect(
            lambda: self._export_results('excel'))
        self.exit_action.triggered.connect(self.close)

        # Edit Menu
        self.clear_snips_action.triggered.connect(self._clear_all_snips)
        self.clear_files_action.triggered.connect(self._clear_all_files)
        self.clear_results_action.triggered.connect(self._clear_all_results)
        self.prefs_action.triggered.connect(self._open_preferences)

        # View Menu
        self.zoom_in_action.triggered.connect(
            lambda: self.graphics_view.scale(1.2, 1.2))
        self.zoom_out_action.triggered.connect(
            lambda: self.graphics_view.scale(1/1.2, 1/1.2))
        self.fit_page_action.triggered.connect(self.graphics_view.fit_to_page)
        self.fit_width_action.triggered.connect(
            self.graphics_view.fit_to_width)
        self.bw_toggle_action.toggled.connect(self._toggle_bw_mode)

        # Toolbar
        self.run_ocr_button.clicked.connect(self._run_ocr_task)
        self.stop_ocr_button.clicked.connect(self._stop_ocr_task)

        # Left Pane
        self.files_list_widget.currentItemChanged.connect(
            self._on_file_selected)
        self.files_list_widget.customContextMenuRequested.connect(
            self._show_files_list_context_menu)
        self.snips_table.itemDoubleClicked.connect(self._rename_snip)

        # Center Pane
        self.first_page_button.clicked.connect(
            lambda: self._navigate_to_page(0))
        self.prev_page_button.clicked.connect(
            lambda: self._navigate_to_page(self.current_page_num - 1))
        self.next_page_button.clicked.connect(
            lambda: self._navigate_to_page(self.current_page_num + 1))
        self.last_page_button.clicked.connect(self._navigate_to_last_page)
        self.page_nav_edit.returnPressed.connect(self._navigate_from_edit)
        self.graphics_view.new_snip_drawn.connect(self._add_new_snip)

        # Right Pane
        self.results_table.customContextMenuRequested.connect(
            self._show_results_table_context_menu)
        self.results_table.horizontalHeader().customContextMenuRequested.connect(
            self._show_header_context_menu)

    # --- Event Handlers ---

    def closeEvent(self, event):
        """Saves window state on close."""
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue(
            "splitter_state", self.main_splitter.saveState())

        reply = QMessageBox.question(self, "Confirm Exit",
                                     "Are you sure you want to exit? Any unsaved work will be lost.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # Clean up any running threads
            if self.ocr_worker and self.ocr_worker.isRunning():
                self.ocr_worker.stop()
                self.ocr_worker.wait()
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        """Accepts drops if they contain file paths."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handles dropped files."""
        urls = event.mimeData().urls()
        file_paths = [url.toLocalFile() for url in urls if url.isLocalFile()]

        supported_files = [
            path for path in file_paths if os.path.splitext(path)[1].lower() in SUPPORTED_FORMATS
        ]

        if supported_files:
            self._start_file_import(supported_files)
        else:
            QMessageBox.warning(
                self, "Unsupported Files", "None of the dropped files are of a supported format.")

    # --- Core Logic & Slots ---

    def _show_error_message(self, message: str):
        """Displays a non-blocking error message."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.show()  # Show non-blockingly

    def _update_status_bar(self):
        """Updates all labels in the status bar."""
        if self.current_file_path and self.current_file_path in self.imported_files:
            file_info = self.imported_files[self.current_file_path]
            self.status_file_label.setText(
                f"File: {os.path.basename(self.current_file_path)}")
            self.status_page_label.setText(
                f"Page {self.current_page_num + 1} of {file_info['num_pages']}")
        else:
            self.status_file_label.setText("No file loaded")
            self.status_page_label.setText("")

        zoom = self.graphics_view.transform().m11() * 100
        self.status_zoom_label.setText(f"Zoom: {zoom:.0f}%")

    def _clear_disk_cache(self):
        """Deletes all files in the disk cache directory."""
        cache_dir = self.config.get("cache_dir")
        if os.path.isdir(cache_dir):
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        print("Disk cache cleared.")

    def _prompt_import_files(self):
        """Opens a dialog to select files for import."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Files",
            "",
            f"All Supported Files ({' '.join(f'*{f}' for f in SUPPORTED_FORMATS)});;"
            f"PDF Documents (*.pdf);;"
            f"Images ({' '.join(f'*{f}' for f in SUPPORTED_IMAGE_FORMATS)})"
        )
        if file_paths:
            self._start_file_import(file_paths)

    def _prompt_import_folder(self):
        """Opens a dialog to select a folder for import."""
        folder_path = QFileDialog.getExistingDirectory(self, "Import Folder")
        if folder_path:
            file_paths = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in SUPPORTED_FORMATS:
                        file_paths.append(os.path.join(root, file))
            if file_paths:
                self._start_file_import(file_paths)
            else:
                QMessageBox.information(
                    self, "No Files Found", "No supported files were found in the selected folder.")

    def _start_file_import(self, file_paths: List[str]):
        """Starts the file import worker thread."""
        if self.import_worker and self.import_worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "An import operation is already in progress.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Importing files...")

        import_dir = self.config.get("imports_dir")
        self.import_worker = FileImportWorker(file_paths, import_dir)
        self.import_worker.file_imported.connect(self._handle_imported_file)
        self.import_worker.finished.connect(self._on_import_finished)
        self.import_worker.start()

    def _handle_imported_file(self, original_path: str, imported_path: str, num_pages: int):
        """Adds a newly imported file to the data model and UI list."""
        try:

            self.imported_files[imported_path] = {
                'original_path': original_path,
                'num_pages': num_pages
            }

            list_item = QListWidgetItem(os.path.basename(imported_path))
            list_item.setData(Qt.ItemDataRole.UserRole, imported_path)
            self.files_list_widget.addItem(list_item)

        except Exception as e:
            self._show_error_message(
                f"Could not open or process file: {os.path.basename(imported_path)}\n\nError: {e}")

    def _on_import_finished(self):
        """Cleans up after the import worker is done."""
        self.progress_bar.setVisible(False)
        self.status_bar.clearMessage()
        if self.files_list_widget.count() > 0 and self.files_list_widget.currentRow() == -1:
            self.files_list_widget.setCurrentRow(0)

    def _on_file_selected(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Handles changing the selected file in the list."""
        if not current_item:
            self.current_file_path = None
            self.graphics_scene.clear()
            return

        self.current_file_path = current_item.data(Qt.ItemDataRole.UserRole)
        self.current_page_num = 0
        self._navigate_to_page(0)
        self._update_snips_on_canvas()

    def _navigate_to_page(self, page_num: int):
        """Navigates to a specific page number in the current document."""
        if not self.current_file_path or self.current_file_path not in self.imported_files:
            return

        num_pages = self.imported_files[self.current_file_path]['num_pages']
        if not (0 <= page_num < num_pages):
            return  # Invalid page number

        self.current_page_num = page_num
        self.page_nav_edit.setText(str(page_num + 1))
        self._render_current_page()
        self._update_status_bar()

    def _navigate_to_last_page(self):
        if self.current_file_path and self.current_file_path in self.imported_files:
            num_pages = self.imported_files[self.current_file_path]['num_pages']
            self._navigate_to_page(num_pages - 1)

    def _navigate_from_edit(self):
        try:
            page_num = int(self.page_nav_edit.text()) - 1
            self._navigate_to_page(page_num)
        except ValueError:
            self._update_status_bar()  # Reset text to current page

    def _render_current_page(self):
        """Starts the page rendering worker thread."""
        if not self.current_file_path:
            return

        if self.render_worker and self.render_worker.isRunning():
            # Don't interrupt, just let it finish. New requests will queue.
            # A more complex implementation could use a request queue.
            return

        self.status_bar.showMessage("Rendering page...")
        dpi = self.config.get("pdf_preview_dpi", 150)
        cache_dir = self.config.get("cache_dir")

        self.render_worker = PageRenderWorker(
            self.current_file_path, self.current_page_num, cache_dir, self.memory_cache, dpi
        )
        self.render_worker.page_rendered.connect(self._display_page)
        self.render_worker.error.connect(self._show_error_message)
        self.render_worker.finished.connect(
            lambda: self.status_bar.clearMessage())
        self.render_worker.start()

    def _display_page(self, pixmap: QPixmap):
        """Displays the rendered pixmap on the graphics view."""
        self.graphics_view.set_pixmap(pixmap)
        self._update_snips_on_canvas()
        self._update_status_bar()
        if self.is_bw_mode:
            self._apply_bw_filter()

    def _toggle_bw_mode(self, checked: bool):
        self.is_bw_mode = checked
        if checked:
            self._apply_bw_filter()
        else:
            # Re-render to restore color
            self._render_current_page()

    def _apply_bw_filter(self):
        """Applies a grayscale effect to the current view."""
        pixmap_item = next((item for item in self.graphics_scene.items(
        ) if isinstance(item, QGraphicsPixmapItem)), None)
        if pixmap_item:
            pixmap = pixmap_item.pixmap()
            if not pixmap.isNull():
                image = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
                pixmap_item.setPixmap(QPixmap.fromImage(image))

    # --- Snip Management ---

    def _add_new_snip(self, rect: QRectF):
        """Adds a new snip to the data model and UI."""
        base_name = "Snip"
        i = 1
        while f"{base_name}_{i}" in self.snips:
            i += 1
        snip_name = f"{base_name}_{i}"

        # Assign a unique color
        hue = (len(self.snips) * 55) % 360  # Cycle through hues
        color = QColor.fromHsv(hue, 200, 255)

        snip_item = SnipItem(self._denormalize_rect(rect), color, snip_name)
        snip_item.geometry_changed.connect(
            lambda: self._on_snip_geometry_changed(snip_name))

        self.snips[snip_name] = {
            'rect': rect,  # Normalized
            'color': color,
            'item': snip_item
        }

        self.graphics_scene.addItem(snip_item)
        self._update_snips_table()

        # If in single snip mode, run OCR immediately
        if self.snip_mode_combo.currentData() == 'single':
            self._run_ocr_for_single_snip(snip_name)

    def _update_snips_table(self):
        """Refreshes the snips table from the data model."""
        self.snips_table.setRowCount(0)
        for name, data in self.snips.items():
            row = self.snips_table.rowCount()
            self.snips_table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~
                               Qt.ItemFlag.ItemIsEditable)  # Read-only here
            self.snips_table.setItem(row, 0, name_item)

            color_item = QTableWidgetItem()
            color_item.setBackground(data['color'])
            color_item.setFlags(color_item.flags() & ~
                                Qt.ItemFlag.ItemIsEditable)
            self.snips_table.setItem(row, 1, color_item)

    def _rename_snip(self, item: QTableWidgetItem):
        """Handles renaming a snip from the table."""
        if item.column() != 0:
            return

        old_name = item.text()
        # Simple input dialog for new name
        new_name, ok = QInputDialog.getText(
            self, "Rename Snip", "Enter new name:", text=old_name)

        if ok and new_name and new_name != old_name:
            if new_name in self.snips:
                QMessageBox.warning(self, "Name Exists",
                                    "A snip with this name already exists.")
                return

            # Update data model
            snip_data = self.snips.pop(old_name)
            snip_data['item'].name = new_name
            self.snips[new_name] = snip_data

            # Update table
            item.setText(new_name)

            # Update results DataFrame if it exists
            if not self.ocr_results.empty and old_name in self.ocr_results.columns:
                self.ocr_results.rename(
                    columns={old_name: new_name}, inplace=True)
                self._update_results_table()

    def _on_snip_geometry_changed(self, snip_name: str):
        """Updates the normalized rect when a SnipItem is moved/resized."""
        if snip_name in self.snips:
            snip_item = self.snips[snip_name]['item']
            # The item's rect is already in scene coordinates
            new_scene_rect = snip_item.rect()

            self.snips[snip_name]['rect'] = self._normalize_rect(
                new_scene_rect)

    def _update_snips_on_canvas(self):
        """Clears and redraws all snips on the canvas."""
        # Remove old snip items
        for item in self.graphics_scene.items():
            if isinstance(item, SnipItem):
                self.graphics_scene.removeItem(item)

        # Add snips from the model
        for name, data in self.snips.items():
            denormalized_rect = self._denormalize_rect(data['rect'])
            snip_item = SnipItem(denormalized_rect, data['color'], name)
            snip_item.geometry_changed.connect(
                lambda n=name: self._on_snip_geometry_changed(n))
            data['item'] = snip_item  # Update item reference
            self.graphics_scene.addItem(snip_item)

    def _normalize_rect(self, scene_rect: QRectF) -> QRectF:
        """Converts scene coordinates to normalized (0-1) coordinates."""
        page_rect = self.graphics_scene.sceneRect()
        if page_rect.width() == 0 or page_rect.height() == 0:
            return QRectF()
        return QRectF(
            scene_rect.x() / page_rect.width(),
            scene_rect.y() / page_rect.height(),
            scene_rect.width() / page_rect.width(),
            scene_rect.height() / page_rect.height()
        )

    def _denormalize_rect(self, norm_rect: QRectF) -> QRectF:
        """Converts normalized (0-1) coordinates to scene coordinates."""
        page_rect = self.graphics_scene.sceneRect()
        return QRectF(
            norm_rect.x() * page_rect.width(),
            norm_rect.y() * page_rect.height(),
            norm_rect.width() * page_rect.width(),
            norm_rect.height() * page_rect.height()
        )

    # --- OCR & Results Management ---

    def _run_ocr_task(self):
        """Prepares and starts the main OCR worker thread."""
        if self.ocr_worker and self.ocr_worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "An OCR task is already running.")
            return

        if not self.snips:
            QMessageBox.warning(
                self, "No Snips", "Please define at least one snip before running OCR.")
            return

        tasks = self._prepare_ocr_tasks()
        if not tasks:
            QMessageBox.warning(
                self, "No Tasks", "Could not generate any OCR tasks. Check your file and page range selections.")
            return

        self.run_ocr_button.setEnabled(False)
        self.stop_ocr_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.ocr_worker = OcrWorker(tasks, self.config)
        self.ocr_worker.result_ready.connect(self._handle_ocr_result)
        self.ocr_worker.finished.connect(self._on_ocr_finished)
        self.ocr_worker.error.connect(self._show_error_message)
        self.ocr_worker.progress.connect(self._update_progress)
        self.ocr_worker.start()

    def _run_ocr_for_single_snip(self, snip_name: str):
        """Runs OCR for a single, newly created snip."""
        if not self.current_file_path:
            return

        snip_data = self.snips.get(snip_name)
        if not snip_data:
            return

        task = {
            'file_path': self.current_file_path,
            'page_num': self.current_page_num,
            'snip_rect': snip_data['rect'],
            'snip_name': snip_name,
            'ocr_engine': self.ocr_engine_combo.currentData()
        }

        # Use the main worker for consistency, even for one task
        self._run_ocr_task()

    def _prepare_ocr_tasks(self) -> List[Dict]:
        """Generates a list of task dictionaries for the OCR worker."""
        tasks = []
        ocr_engine = self.ocr_engine_combo.currentData()
        snip_mode = self.snip_mode_combo.currentData()

        if snip_mode == 'single':
            # In single mode, assume we run on all snips on the current page
            if self.current_file_path:
                for name, data in self.snips.items():
                    tasks.append({
                        'file_path': self.current_file_path,
                        'page_num': self.current_page_num,
                        'snip_rect': data['rect'],
                        'snip_name': name,
                        'ocr_engine': ocr_engine
                    })
        elif snip_mode == 'template':
            selected_files = [item.data(Qt.ItemDataRole.UserRole)
                              for item in self.files_list_widget.selectedItems()]
            if not selected_files:
                QMessageBox.warning(
                    self, "No Files Selected", "Please select one or more files from the list for Template mode.")
                return []

            page_indices = self._parse_page_range(self.page_range_edit.text())
            if not page_indices:
                QMessageBox.warning(
                    self, "Invalid Page Range", "Please enter a valid page range (e.g., 1, 3-5).")
                return []

            for file_path in selected_files:
                num_pages = self.imported_files[file_path]['num_pages']
                for page_num in page_indices:
                    if 0 <= page_num < num_pages:
                        for name, data in self.snips.items():
                            tasks.append({
                                'file_path': file_path,
                                'page_num': page_num,
                                'snip_rect': data['rect'],
                                'snip_name': name,
                                'ocr_engine': ocr_engine
                            })
        return tasks

    def _parse_page_range(self, range_str: str) -> List[int]:
        """Parses a page range string (e.g., "1, 3-5") into a list of 0-based indices."""
        indices = set()
        if not range_str:
            return []

        parts = range_str.split(',')
        try:
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    for i in range(start, end + 1):
                        indices.add(i - 1)
                else:
                    indices.add(int(part) - 1)
            return sorted(list(indices))
        except ValueError:
            return []

    def _stop_ocr_task(self):
        """Signals the OCR worker to stop."""
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.stop()
            self.status_bar.showMessage("Stopping OCR task...")

    def _on_ocr_finished(self):
        """Cleans up after the OCR worker is done."""
        self.run_ocr_button.setEnabled(True)
        self.stop_ocr_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("OCR task finished.", 3000)

    def _update_progress(self, value: int, message: str):
        """Updates the progress bar and status message."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def _handle_ocr_result(self, result: Dict):
        """Adds a new OCR result to the results DataFrame."""
        new_row = pd.DataFrame({
            'Source File': [os.path.basename(result['file_path'])],
            'Page Number': [result['page_num'] + 1],
            result['snip_name']: [result['text']]
        })

        if self.ocr_results.empty:
            self.ocr_results = new_row
        else:
            # Find if a row for this file/page already exists
            match = self.ocr_results[
                (self.ocr_results['Source File'] == new_row['Source File'][0]) &
                (self.ocr_results['Page Number'] == new_row['Page Number'][0])
            ]

            if not match.empty:
                # Update existing row
                index = match.index[0]
                if result['snip_name'] not in self.ocr_results.columns:
                    # Add new column
                    self.ocr_results[result['snip_name']] = ""
                self.ocr_results.loc[index,
                                     result['snip_name']] = result['text']
            else:
                # Append new row, merging columns
                self.ocr_results = pd.concat(
                    [self.ocr_results, new_row], ignore_index=True)

        self.ocr_results.fillna("", inplace=True)
        self._update_results_table()

    def _update_results_table(self):
        """Refreshes the results table from the DataFrame, applying filters."""
        if self.ocr_results.empty:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        # Apply filters
        df_filtered = self.ocr_results.copy()
        for col, values in self.column_filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col].astype(
                    str).isin(values)]

        self.results_table.setRowCount(df_filtered.shape[0])
        self.results_table.setColumnCount(df_filtered.shape[1])
        self.results_table.setHorizontalHeaderLabels(df_filtered.columns)

        for row_idx, row_data in enumerate(df_filtered.itertuples(index=False)):
            for col_idx, cell_data in enumerate(row_data):
                self.results_table.setItem(
                    row_idx, col_idx, QTableWidgetItem(str(cell_data)))

        self.results_table.resizeColumnsToContents()

    # --- Context Menus ---

    def _show_files_list_context_menu(self, pos: QPoint):
        """Context menu for the imported files list."""
        if not self.files_list_widget.selectedItems():
            return

        menu = QMenu()
        delete_action = menu.addAction("Delete Selected File(s)")
        action = menu.exec(self.files_list_widget.mapToGlobal(pos))

        if action == delete_action:
            self._delete_selected_files()

    def _delete_selected_files(self):
        """Deletes selected files from the project."""
        items_to_delete = self.files_list_widget.selectedItems()
        if not items_to_delete:
            return

        reply = QMessageBox.question(self, "Confirm Delete",
                                     f"Are you sure you want to remove {len(items_to_delete)} file(s) from the project? This cannot be undone.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        for item in items_to_delete:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            self.files_list_widget.takeItem(self.files_list_widget.row(item))
            if file_path in self.imported_files:
                del self.imported_files[file_path]
            # Optionally, delete from results table
            if not self.ocr_results.empty:
                self.ocr_results = self.ocr_results[self.ocr_results['Source File'] != os.path.basename(
                    file_path)]
            # Optionally, delete the file from imports_dir
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    self._show_error_message(
                        f"Could not delete file from workspace: {e}")

        self._update_results_table()
        if self.files_list_widget.count() == 0:
            self.graphics_scene.clear()

    def _show_results_table_context_menu(self, pos: QPoint):
        """Context menu for the results table cells."""
        item = self.results_table.itemAt(pos)
        if not item:
            return

        menu = QMenu()
        copy_cell_action = menu.addAction("Copy Cell")
        copy_row_action = menu.addAction("Copy Row")
        menu.addSeparator()
        goto_action = menu.addAction("Go to Snip Location")

        action = menu.exec(self.results_table.mapToGlobal(pos))

        if action == copy_cell_action:
            QApplication.clipboard().setText(item.text())
        elif action == copy_row_action:
            row_text = "\t".join([self.results_table.item(
                item.row(), c).text() for c in range(self.results_table.columnCount())])
            QApplication.clipboard().setText(row_text)
        elif action == goto_action:
            self._go_to_snip_location(item.row(), item.column())

    def _go_to_snip_location(self, row, col):
        """Navigates the view to the location of a result cell's snip."""
        try:
            file_name = self.results_table.item(row, 0).text()
            page_num = int(self.results_table.item(row, 1).text()) - 1
            snip_name = self.results_table.horizontalHeaderItem(col).text()
        except (AttributeError, ValueError, IndexError):
            return

        # Find the full path for the file name
        target_path = None
        for path in self.imported_files:
            if os.path.basename(path) == file_name:
                target_path = path
                break

        if not target_path or snip_name not in self.snips:
            return

        # Select the file in the list
        for i in range(self.files_list_widget.count()):
            item = self.files_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == target_path:
                self.files_list_widget.setCurrentItem(item)
                break

        # Navigate to the page
        self._navigate_to_page(page_num)

        # Highlight the snip
        QApplication.processEvents()  # Allow view to update
        snip_item = self.snips[snip_name].get('item')
        if snip_item:
            for item in self.graphics_scene.items():
                item.setSelected(False)
            snip_item.setSelected(True)
            self.graphics_view.centerOn(snip_item)

    def _show_header_context_menu(self, pos: QPoint):
        """Context menu for the results table header for filtering."""
        header = self.results_table.horizontalHeader()
        col = header.logicalIndexAt(pos)
        if col == -1:
            return

        col_name = self.results_table.horizontalHeaderItem(col).text()

        menu = QMenu()
        filter_action = menu.addAction("Filter...")
        clear_filter_action = menu.addAction("Clear Filter")
        clear_filter_action.setEnabled(col_name in self.column_filters)

        action = menu.exec(header.mapToGlobal(pos))

        if action == filter_action:
            if not self.ocr_results.empty and col_name in self.ocr_results.columns:
                unique_values = self.ocr_results[col_name].astype(
                    str).unique().tolist()
                dialog = FilterDialog(col_name, unique_values, self)
                if dialog.exec():
                    self.column_filters[col_name] = dialog.get_selected_values(
                    )
                    self._update_results_table()
        elif action == clear_filter_action:
            if col_name in self.column_filters:
                del self.column_filters[col_name]
                self._update_results_table()

    # --- Clear Actions ---

    def _clear_all_snips(self):
        if QMessageBox.question(self, "Confirm", "Clear all defined snips?") == QMessageBox.StandardButton.Yes:
            for item in self.graphics_scene.items():
                if isinstance(item, SnipItem):
                    self.graphics_scene.removeItem(item)
            self.snips.clear()
            self._update_snips_table()

    def _clear_all_files(self):
        if QMessageBox.question(self, "Confirm", "Clear all imported files? This will also clear snips and results.") == QMessageBox.StandardButton.Yes:
            self._reset_workspace()

    def _clear_all_results(self):
        if QMessageBox.question(self, "Confirm", "Clear all OCR results?") == QMessageBox.StandardButton.Yes:
            self.ocr_results = pd.DataFrame()
            self.column_filters.clear()
            self._update_results_table()

    def _reset_workspace(self, clear_files_from_disk=True):
        """Resets the entire application state."""
        # Stop any running workers
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.stop()

        # Clear data models
        self.snips.clear()
        self.ocr_results = pd.DataFrame()
        self.column_filters.clear()

        # Clear UI
        self.graphics_scene.clear()
        self.files_list_widget.clear()
        self._update_snips_table()
        self._update_results_table()

        # Clear files
        if clear_files_from_disk:
            import_dir = self.config.get("imports_dir")
            for path in self.imported_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        self.imported_files.clear()
        self.current_file_path = None
        self.current_page_num = 0
        self._update_status_bar()

    def _open_preferences(self):
        """Opens the config.json file in the default text editor."""
        try:
            os.startfile(self.config_manager.config_path)
        except AttributeError:  # For non-Windows
            import subprocess
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, self.config_manager.config_path])
        except Exception as e:
            self._show_error_message(f"Could not open config file: {e}")

    # --- Session Management ---

    def _save_session(self):
        """Saves the current state into a .ocr-session zip file."""
        if not self.imported_files:
            QMessageBox.warning(self, "Empty Session",
                                "There are no imported files to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", f"OCR Session Files (*{SESSION_EXTENSION})"
        )
        if not file_path:
            return

        if not file_path.endswith(SESSION_EXTENSION):
            file_path += SESSION_EXTENSION

        try:
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Save session metadata (snips, results)
                session_data = {
                    'snips': {name: {'rect': [d.x(), d.y(), d.width(), d.height()], 'color': data['color'].name()}
                              for name, data in self.snips.items() for d in [data['rect']]},
                    'version': APP_VERSION
                }
                zf.writestr('session.json', json.dumps(session_data, indent=4))

                # Save results to a CSV inside the zip
                if not self.ocr_results.empty:
                    zf.writestr('results.csv',
                                self.ocr_results.to_csv(index=False))

                # 2. Save imported files
                for imported_path in self.imported_files.keys():
                    if os.path.exists(imported_path):
                        zf.write(imported_path,
                                 arcname=os.path.basename(imported_path))

            self.status_bar.showMessage(
                f"Session saved to {os.path.basename(file_path)}", 5000)

        except Exception as e:
            self._show_error_message(f"Failed to save session: {e}")

    def _load_session(self):
        """Loads a .ocr-session file."""
        if self.imported_files:
            reply = QMessageBox.question(self, "Load Session",
                                         "This will clear your current session. Continue?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", f"OCR Session Files (*{SESSION_EXTENSION})"
        )
        if not file_path:
            return

        self._reset_workspace(clear_files_from_disk=True)
        import_dir = self.config.get("imports_dir")

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                file_list = zf.namelist()

                # 1. Extract imported files into workspace
                for file_name in file_list:
                    if file_name not in ['session.json', 'results.csv']:
                        zf.extract(file_name, path=import_dir)
                        extracted_path = os.path.join(import_dir, file_name)
                        num_pages = 0
                        try:
                            ext = os.path.splitext(extracted_path)[1].lower()
                            if ext == '.pdf':
                                doc = fitz.open(extracted_path)
                                num_pages = len(doc)
                                doc.close()
                            elif ext in SUPPORTED_IMAGE_FORMATS:
                                num_pages = 1
                        except Exception as e:
                            print(
                                f"Error counting pages in {extracted_path}: {e}")

                        # This adds the file to our data models
                        self._handle_imported_file(os.path.join(
                            zf.filename, file_name), extracted_path, num_pages)

                # 2. Load session metadata
                if 'session.json' in file_list:
                    with zf.open('session.json') as f:
                        session_data = json.load(f)

                    # Recreate snips
                    for name, data in session_data.get('snips', {}).items():
                        rect_data = data['rect']
                        self.snips[name] = {
                            'rect': QRectF(rect_data[0], rect_data[1], rect_data[2], rect_data[3]),
                            'color': QColor(data['color']),
                            'item': None  # Will be created when page is displayed
                        }
                    self._update_snips_table()

                # 3. Load results
                if 'results.csv' in file_list:
                    with zf.open('results.csv') as f:
                        self.ocr_results = pd.read_csv(f)
                    self._update_results_table()

            if self.files_list_widget.count() > 0:
                self.files_list_widget.setCurrentRow(0)

            self.status_bar.showMessage(
                f"Session loaded from {os.path.basename(file_path)}", 5000)

        except Exception as e:
            self._show_error_message(f"Failed to load session: {e}")
            self._reset_workspace()

    # --- Exporting ---

    def _export_results(self, format: str):
        """Exports the results table to CSV or Excel."""
        if self.ocr_results.empty:
            QMessageBox.warning(self, "No Results",
                                "There are no OCR results to export.")
            return

        ext, file_filter = (
            '.csv', "CSV Files (*.csv)") if format == 'csv' else ('.xlsx', "Excel Files (*.xlsx)")

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export to {format.upper()}", self.config.get("exports_dir"), file_filter)

        if not file_path:
            return

        if not file_path.lower().endswith(ext):
            file_path += ext

        try:
            if format == 'csv':
                self.ocr_results.to_csv(file_path, index=False)
            else:  # excel
                try:
                    import openpyxl
                    self.ocr_results.to_excel(
                        file_path, index=False, engine='openpyxl')
                except ImportError:
                    self._show_error_message(
                        "The 'openpyxl' library is required to export to Excel.\nPlease install it using: pip install openpyxl")
                    return

            self.status_bar.showMessage(
                f"Results exported to {os.path.basename(file_path)}", 5000)

        except Exception as e:
            self._show_error_message(f"Failed to export results: {e}")


# --- Main Execution ---
# --- Main Execution ---
if __name__ == '__main__':
    # Ensure high DPI scaling is handled correctly.
    # High DPI scaling is enabled by default in PyQt6, so we only need to set the pixmap attribute.

    app = QApplication(sys.argv)

    # Check for Tesseract availability early
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Found Tesseract version {tesseract_version}")
    except pytesseract.TesseractNotFoundError:
        msg = ("Tesseract is not installed or not in your PATH. "
               "Please install it and configure the path in 'config.json'. "
               "The application may not function correctly.")
        QMessageBox.warning(None, "Tesseract Not Found", msg)
        # We don't exit, allowing user to fix config or use other engines

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
