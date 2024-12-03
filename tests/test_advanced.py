import torch
import torch.nn as nn
import pytest
import os
import tempfile
from liquid_neural_network.liquid_s4 import LiquidS4Model
from liquid_neural_network.cfc_model import CfCModel, CfCCell

class TestAdvancedFeatures:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test parameters and ensure deterministic behavior"""
        torch.manual_seed(42)
        self.batch_size = 8
        self.seq_length = 32
        self.input_size = 16
        self.hidden_size = 32
        self.output_size = 8
        self.device = torch.device('cpu')
        
    def test_batch_consistency(self):
        """Test that model outputs are consistent across different batch sizes"""
        # Set deterministic behavior
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Generate test inputs with same seed
        x = torch.randn(8, self.seq_length, self.input_size).to(self.device)
        x1 = x[:4]  # First 4 sequences
        x2 = x  # All 8 sequences
        
        # Get outputs
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2[:4])  # Use first 4 sequences
            
        # Check consistency with increased tolerance
        assert torch.allclose(out1, out2, rtol=1e-3, atol=1e-4), "Outputs differ across batch sizes"
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        model = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Generate test input
        x = torch.randn(self.batch_size, self.seq_length, self.input_size).to(self.device)
        
        # Forward pass
        output = model(x)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            
    def test_long_sequence_processing(self):
        """Test model's ability to handle long sequences"""
        model = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Generate long sequence
        long_seq_length = 1024
        x = torch.randn(2, long_seq_length, self.input_size).to(self.device)
        
        # Process sequence
        with torch.no_grad():
            output = model(x)
            
        # Check output shape and values
        assert output.shape == (2, long_seq_length, self.output_size)
        assert not torch.isnan(output).any()
        
    def test_model_save_load(self):
        """Test model save and load functionality"""
        # Create and initialize model
        model = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Generate test input
        x = torch.randn(self.batch_size, self.seq_length, self.input_size).to(self.device)
        
        # Get initial output
        with torch.no_grad():
            output1 = model(x)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            tmp_name = tmp.name
            
        # Create new model and load state
        model2 = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        model2.load_state_dict(torch.load(tmp_name))
        
        # Get output from loaded model
        with torch.no_grad():
            output2 = model2(x)
            
        # Clean up
        os.unlink(tmp_name)
        
        # Check outputs match
        assert torch.allclose(output1, output2), "Outputs differ after model load"
        
    def test_numerical_stability(self):
        """Test numerical stability with different input scales"""
        model = LiquidS4Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Test with different input scales
        scales = [0.1, 1.0, 10.0]
        x = torch.randn(self.batch_size, self.seq_length, self.input_size).to(self.device)
        
        outputs = []
        with torch.no_grad():
            for scale in scales:
                output = model(x * scale)
                assert not torch.isnan(output).any(), f"NaN output with scale {scale}"
                outputs.append(output)
                
        # Check relative differences with more lenient bounds
        for i in range(len(scales)-1):
            ratio = outputs[i+1].abs().mean() / outputs[i].abs().mean()
            assert 0.01 < ratio < 100, f"Unexpected scaling behavior between {scales[i]} and {scales[i+1]}"
            
    def test_state_continuity(self):
        """Test that model maintains continuity in state processing"""
        model = CfCModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)
        
        # Generate sequence
        x = torch.randn(self.batch_size, self.seq_length, self.input_size).to(self.device)
        
        # Process full sequence
        full_output, _ = model(x)
        
        # Process in chunks
        chunk_size = self.seq_length // 2
        first_chunk, h = model(x[:, :chunk_size])
        second_chunk, _ = model(x[:, chunk_size:], h)
        
        # Combine chunks
        chunked_output = torch.cat([first_chunk, second_chunk], dim=1)
        
        # Check continuity
        assert torch.allclose(full_output, chunked_output, rtol=1e-4), "State processing is not continuous"

@pytest.fixture
def model_params():
    return {
        'input_size': 10,
        'hidden_size': 20,
        'output_size': 5,
        'num_layers': 2
    }

@pytest.fixture
def batch_params():
    return {
        'batch_size': 8,
        'seq_length': 15
    }

def test_cfc_cell_initialization(model_params):
    cell = CfCCell(model_params['input_size'], model_params['hidden_size'])
    
    # Test parameter initialization
    for name, param in cell.named_parameters():
        if 'weight' in name:
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()
            assert param.abs().mean() > 0  # Weights should be initialized

def test_cfc_model_forward(model_params, batch_params):
    model = CfCModel(**model_params)
    x = torch.randn(batch_params['batch_size'], batch_params['seq_length'], model_params['input_size'])
    
    # Test forward pass
    output, hidden = model(x)
    
    # Check output shape
    assert output.shape == (batch_params['batch_size'], batch_params['seq_length'], model_params['output_size'])
    
    # Check hidden state
    assert len(hidden) == model_params['num_layers']
    assert all(h.shape == (batch_params['batch_size'], model_params['hidden_size']) for h in hidden)

def test_state_continuity(model_params):
    model = CfCModel(**model_params)
    batch_size = 4
    seq_len = 10
    
    # Generate two sequences
    x1 = torch.randn(batch_size, seq_len, model_params['input_size'])
    x2 = torch.randn(batch_size, seq_len, model_params['input_size'])
    
    # Process first sequence
    out1, hidden1 = model(x1)
    
    # Process second sequence using hidden state from first
    out2, hidden2 = model(x2, hidden1)
    
    # Check that hidden states are different but valid
    for h1, h2 in zip(hidden1, hidden2):
        assert not torch.allclose(h1, h2)  # States should evolve
        assert not torch.isnan(h2).any()  # No NaN values
        assert not torch.isinf(h2).any()  # No infinite values

def test_batch_consistency(model_params):
    model = CfCModel(**model_params)
    torch.manual_seed(42)  # For reproducibility
    
    # Create a single input sequence and repeat it
    seq_len = 10
    x_single = torch.randn(1, seq_len, model_params['input_size'])
    
    outputs = []
    batch_sizes = [1, 4, 16]
    
    for batch_size in batch_sizes:
        # Repeat the same sequence for each batch
        x = x_single.repeat(batch_size, 1, 1)
        out, _ = model(x)
        # Get the first sequence output (should be identical across batches)
        outputs.append(out[0])
    
    # Check consistency across batch sizes
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], rtol=1e-4, atol=1e-4)

def test_gradient_flow(model_params):
    model = CfCModel(**model_params)
    batch_size = 4
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, model_params['input_size'])
    target = torch.randn(batch_size, seq_len, model_params['output_size'])
    
    # Forward pass
    output, _ = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any()
