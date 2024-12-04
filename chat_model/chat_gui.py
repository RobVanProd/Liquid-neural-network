import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog)
from PyQt6.QtCore import Qt
import torch
from model import ChatModel

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shakespeare Chat Interface")
        self.setGeometry(100, 100, 800, 800)  # Made window taller
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Debug display
        self.debug_display = QTextEdit()
        self.debug_display.setReadOnly(True)
        self.debug_display.setMaximumHeight(200)
        layout.addWidget(QLabel("Debug Information:"))
        layout.addWidget(self.debug_display)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)
        
        # Load Model button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model_dialog)
        layout.addWidget(self.load_button)
        
        # Model status
        self.status_label = QLabel("Model Status: Not Loaded")
        layout.addWidget(self.status_label)
        
        self.model = None
        
        # Welcome message
        self.chat_display.append("Welcome to Shakespeare Chat! Please load a model to begin.\n")
        
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
                self.chat_display.append("Model loaded successfully! You can now chat.\n")
            except Exception as e:
                self.status_label.setText(f"Model Status: Error - {str(e)}")
                self.chat_display.append(f"Error loading model: {str(e)}\n")
    
    def load_model(self, model_path):
        # Load the trained model
        checkpoint = torch.load(model_path)
        config = checkpoint.get('config', {
            'max_length': 64,
            'vocab_size': 50257,
            'n_layer': 6,
            'n_head': 8
        })
        
        self.model = ChatModel(
            n_positions=config.get('max_length', 64),
            vocab_size=config.get('vocab_size', 50257),
            n_layer=config.get('n_layer', 6),
            n_head=config.get('n_head', 8)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded with config: {config}")
    
    def generate_response(self, user_input):
        if self.model is None:
            raise Exception("Please load a model first!")
        
        try:
            # Generate response using the new method
            response = self.model.generate_text(user_input)
            return response
            
        except Exception as e:
            import traceback
            print(f"Generation error: {str(e)}")
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            return
        
        # Display user message
        self.chat_display.append(f"You: {user_message}")
        self.input_field.clear()
        
        # Clear debug display before each generation
        self.debug_display.clear()
        
        try:
            # Get model response
            response = self.generate_response(user_message)
            self.chat_display.append(f"Shakespeare: {response}\n")
        except Exception as e:
            self.chat_display.append(f"Error: {str(e)}\n")

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
