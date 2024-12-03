import unittest
import torch
import numpy as np
from liquid_neural_network.liquid_s4 import LiquidS4Model, LiquidS4Cell, LiquidS4Layer
from liquid_neural_network.cfc_model import CfCModel, CfCCell, CfCLayer
from liquid_neural_network.visualization import NetworkVisualizer
import os
import shutil

class TestLiquidS4Model(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 100
        self.input_size = 10
        self.hidden_size = 64
        self.output_size = 10
        self.model = LiquidS4Model(
            self.input_size,
            self.hidden_size,
            self.output_size
        )
        
    def test_model_initialization(self):
        """Test if model initializes with correct parameters"""
        self.assertIsInstance(self.model, LiquidS4Model)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.output_size, self.output_size)
        
    def test_forward_pass(self):
        """Test if forward pass produces correct output shape"""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.model(x)
        
        expected_shape = (self.batch_size, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_cell_computation(self):
        """Test individual S4 cell computations"""
        cell = LiquidS4Cell(self.input_size, N=64)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = cell(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.input_size))
        
class TestCfCModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 100
        self.input_size = 10
        self.hidden_size = 64
        self.output_size = 10
        self.model = CfCModel(
            self.input_size,
            self.hidden_size,
            self.output_size
        )
        
    def test_model_initialization(self):
        """Test if model initializes with correct parameters"""
        self.assertIsInstance(self.model, CfCModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.output_size, self.output_size)
        
    def test_forward_pass(self):
        """Test if forward pass produces correct output shape"""
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.model(x)
        
        expected_shape = (self.batch_size, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_cell_computation(self):
        """Test individual CfC cell computations"""
        cell = CfCCell(self.input_size, self.hidden_size)
        x = torch.randn(self.batch_size, self.input_size)
        h = torch.randn(self.batch_size, self.hidden_size)
        output, h_new = cell(x, h)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(h_new.shape, (self.batch_size, self.hidden_size))
        
class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_vis'
        self.visualizer = NetworkVisualizer(log_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            # Wait for TensorBoard to release the files
            import time
            time.sleep(1)
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"Warning: Could not remove {self.test_dir} - it may be in use")
            
    def test_metric_logging(self):
        """Test if metrics are logged correctly"""
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        self.visualizer.log_metrics(metrics, step=0)
        
        # Verify metrics were added to history
        self.assertEqual(self.visualizer.training_history['loss'][-1], 0.5)
        self.assertEqual(self.visualizer.training_history['accuracy'][-1], 0.8)
        
    def test_activation_visualization(self):
        """Test activation visualization"""
        activations = torch.randn(10, 10)  # Sample activations
        self.visualizer.visualize_activations(
            activations,
            'test_layer',
            step=0
        )
        
        # Verify that tensorboard log directory was created
        self.assertTrue(os.path.exists(self.test_dir))
        
def run_tests():
    unittest.main()
    
if __name__ == '__main__':
    run_tests()
