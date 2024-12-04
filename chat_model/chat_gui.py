import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTextEdit, QLineEdit, QPushButton, QLabel)
from PyQt6.QtCore import Qt
import torch
from transformers import AutoTokenizer

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Interface")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)
        
        # Model status
        self.status_label = QLabel("Model Status: Loading...")
        layout.addWidget(self.status_label)
        
        # Try to load the model
        try:
            self.load_model()
            self.status_label.setText("Model Status: Ready")
        except Exception as e:
            self.status_label.setText(f"Model Status: Error - {str(e)}")
    
    def load_model(self):
        # Load the trained model
        model_path = "chat_model.pth"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = torch.load(model_path)
            self.model.eval()
        except FileNotFoundError:
            raise Exception("Model file not found. Please train the model first.")
    
    def generate_response(self, user_input):
        # Tokenize input
        inputs = self.tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=128,
                num_return_sequences=1,
                temperature=0.7
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            return
        
        # Display user message
        self.chat_display.append(f"You: {user_message}")
        self.input_field.clear()
        
        try:
            # Get model response
            response = self.generate_response(user_message)
            self.chat_display.append(f"Bot: {response}\n")
        except Exception as e:
            self.chat_display.append(f"Error: {str(e)}\n")

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
