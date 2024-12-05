import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog,
                            QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QPalette, QColor, QFont, QFontDatabase
import torch
from model import ChatModel

class CustomButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 10))
        
        # Default style
        self.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: #00FF9C;
                border: 2px solid #00FF9C;
                border-radius: 20px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #00FF9C;
                color: #1E1E1E;
            }
            QPushButton:pressed {
                background-color: #00CC7A;
                border-color: #00CC7A;
            }
        """)

class CustomLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: 2px solid #404040;
                border-radius: 20px;
                padding: 5px 15px;
            }
            QLineEdit:focus {
                border-color: #00FF9C;
            }
        """)

class CustomTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 2px solid #404040;
                border-radius: 10px;
                padding: 10px;
            }
            QTextEdit:focus {
                border-color: #00FF9C;
            }
        """)

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shakespeare AI Chat")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("background-color: #1E1E1E;")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Shakespeare AI Chat")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #00FF9C; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Chat display frame
        chat_frame = QFrame()
        chat_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-radius: 10px;
            }
        """)
        chat_layout = QVBoxLayout(chat_frame)
        
        # Chat history display
        self.chat_display = CustomTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        chat_layout.addWidget(self.chat_display)
        
        main_layout.addWidget(chat_frame)
        
        # Debug frame
        debug_frame = QFrame()
        debug_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-radius: 10px;
            }
        """)
        debug_layout = QVBoxLayout(debug_frame)
        
        debug_label = QLabel("Debug Information")
        debug_label.setFont(QFont("Segoe UI", 10))
        debug_label.setStyleSheet("color: #00FF9C;")
        debug_layout.addWidget(debug_label)
        
        self.debug_display = CustomTextEdit()
        self.debug_display.setReadOnly(True)
        self.debug_display.setMaximumHeight(150)
        debug_layout.addWidget(self.debug_display)
        
        main_layout.addWidget(debug_frame)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-radius: 10px;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setSpacing(10)
        
        self.input_field = CustomLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = CustomButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        main_layout.addWidget(input_frame)
        
        # Control buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(10)
        
        self.load_button = CustomButton("Load Model")
        self.load_button.clicked.connect(self.load_model_dialog)
        button_layout.addWidget(self.load_button)
        
        main_layout.addWidget(button_frame)
        
        # Status bar
        self.status_label = QLabel("Model Status: Not Loaded")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #888888;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.status_label)
        
        self.model = None
        
        # Welcome message
        self.chat_display.append("Welcome to Shakespeare AI Chat! Please load a model to begin.\n")
        
        # Redirect stdout to capture debug info
        sys.stdout = self
        
    def write(self, text):
        """Capture print statements and show in debug display."""
        self.debug_display.append(text)
    
    def flush(self):
        """Required for stdout redirection."""
        pass
    
    def load_model_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "checkpoints",
            "Model Files (*.pth)"
        )
        if file_name:
            try:
                self.debug_display.clear()
                self.load_model(file_name)
                self.status_label.setText(f"Model Status: Loaded - {os.path.basename(file_name)}")
                self.status_label.setStyleSheet("color: #00FF9C;")
                self.chat_display.append("Model loaded successfully! You can now chat.\n")
            except Exception as e:
                self.status_label.setText(f"Model Status: Error - {str(e)}")
                self.status_label.setStyleSheet("color: #FF4444;")
                self.chat_display.append(f"Error loading model: {str(e)}\n")
    
    def load_model(self, model_path):
        try:
            # Load the trained model
            checkpoint = torch.load(model_path, map_location='cpu')  # First load to CPU
            config = checkpoint.get('config', {})
            
            # Get the saved model's configuration
            saved_n_positions = checkpoint['model_state_dict']['model.transformer.wpe.weight'].size(0)
            print(f"Saved model position size: {saved_n_positions}")
            
            self.model = ChatModel(
                n_positions=saved_n_positions,  # Use the exact size from saved model
                vocab_size=config.get('vocab_size', 50257),
                n_layer=config.get('n_layer', 6),
                n_head=config.get('n_head', 8)
            )
            
            # Load state dict after mapping to CPU
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            
            print(f"Model loaded successfully with config: {config}")
            
        except Exception as e:
            import traceback
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            raise e
    
    def send_message(self):
        if not self.model:
            self.chat_display.append("Please load a model first!\n")
            return
        
        message = self.input_field.text().strip()
        if not message:
            return
        
        # Clear input field
        self.input_field.clear()
        
        # Display user message with styling
        self.chat_display.append(f'<div style="color: #00FF9C; margin: 5px 0px;"><b>You:</b> {message}</div>')
        
        try:
            # Generate response
            response = self.model.generate_text(message)
            
            # Display model response with styling
            self.chat_display.append(f'<div style="color: #FFFFFF; background-color: #2E2E2E; padding: 10px; border-radius: 10px; margin: 5px 0px;"><b>Shakespeare:</b> {response}</div>')
            
        except Exception as e:
            error_msg = str(e)
            self.chat_display.append(f'<div style="color: #FF4444; margin: 5px 0px;"><b>Error:</b> {error_msg}</div>')
            print(f"Error generating response: {error_msg}")

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle("Fusion")
    
    # Create and show the chat window
    window = ChatWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
