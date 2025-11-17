"""
ResNet18 Model Loading and Management

Provides functions to load PyTorch ResNet18 model with pretrained weights
and export to various formats for TVM compilation.
"""

import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import io
import json


def load_resnet18(pretrained=True):
    """
    Load ResNet18 model with optional pretrained weights

    Args:
        pretrained (bool): If True, load ImageNet pretrained weights

    Returns:
        torch.nn.Module: ResNet18 model in evaluation mode
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    model.eval()
    return model


def get_model_info(model=None):
    """
    Get information about the ResNet18 model

    Args:
        model: PyTorch model (if None, creates a new ResNet18)

    Returns:
        dict: Model information including architecture details
    """
    if model is None:
        model = load_resnet18(pretrained=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get input shape
    input_shape = (1, 3, 224, 224)  # Batch, Channels, Height, Width

    # Get model structure info
    model_info = {
        'model_name': 'ResNet18',
        'total_params': str(total_params),
        'trainable_params': str(trainable_params),
        'input_shape': str(input_shape),
        'input_size': '224x224',
        'num_classes': '1000',
        'architecture': 'ResNet',
        'depth': '18'
    }

    return model_info


def export_to_onnx(output_path='resnet18.onnx', input_shape=(1, 3, 224, 224)):
    """
    Export ResNet18 model to ONNX format

    Args:
        output_path (str): Path to save ONNX model
        input_shape (tuple): Input tensor shape (batch, channels, height, width)

    Returns:
        dict: Export status information
    """
    try:
        model = load_resnet18(pretrained=True)
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        return {
            'status': 'success',
            'output_path': output_path,
            'input_shape': str(input_shape),
            'opset_version': '11'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def get_traced_model(input_shape=(1, 3, 224, 224)):
    """
    Get TorchScript traced model for TVM import

    Args:
        input_shape (tuple): Input tensor shape

    Returns:
        tuple: (traced_model, example_input)
    """
    model = load_resnet18(pretrained=True)
    example_input = torch.randn(*input_shape)

    traced_model = torch.jit.trace(model, example_input)

    return traced_model, example_input


def get_scripted_model():
    """
    Get TorchScript scripted model

    Returns:
        torch.jit.ScriptModule: Scripted ResNet18 model
    """
    model = load_resnet18(pretrained=True)
    scripted_model = torch.jit.script(model)

    return scripted_model


def save_model_state(output_path='resnet18_state.pth'):
    """
    Save model state dict

    Args:
        output_path (str): Path to save state dict

    Returns:
        dict: Save status information
    """
    try:
        model = load_resnet18(pretrained=True)
        torch.save(model.state_dict(), output_path)

        return {
            'status': 'success',
            'output_path': output_path,
            'format': 'state_dict'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def preprocess_image(image_path):
    """
    Preprocess image for ResNet18 inference

    Args:
        image_path (str): Path to image file

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define preprocessing pipeline (ImageNet standard)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    return input_batch


def predict_image(image_path):
    """
    Run inference on an image and return top-5 predictions

    Args:
        image_path (str): Path to image file

    Returns:
        dict: Prediction results with top-5 classes and probabilities
    """
    try:
        import time
        import numpy as np

        # Set device (CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and preprocess image
        model = load_resnet18(pretrained=True)
        model = model.to(device)  # Move model to device

        input_batch = preprocess_image(image_path)
        input_batch = input_batch.to(device)  # Move input to device

        # Warmup (5 iterations)
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_batch)

        # Benchmark inference time (10 iterations)
        num_iterations = 10
        inference_times = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                output = model(input_batch)
                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = float(np.mean(inference_times))
        std_time = float(np.std(inference_times))
        min_time = float(np.min(inference_times))
        max_time = float(np.max(inference_times))

        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Load ImageNet class labels
        weights = ResNet18_Weights.IMAGENET1K_V1
        categories = weights.meta["categories"]

        # Prepare results
        predictions = []
        for i in range(5):
            predictions.append({
                'class_id': int(top5_catid[i]),
                'class_name': categories[top5_catid[i]],
                'probability': float(top5_prob[i])
            })

        return {
            'status': 'success',
            'image_path': image_path,
            'device': str(device),
            'top1_class': categories[top5_catid[0]],
            'top1_probability': float(top5_prob[0]),
            'top5_predictions': predictions,
            'avg_inference_time_ms': str(avg_time),
            'std_inference_time_ms': str(std_time),
            'min_inference_time_ms': str(min_time),
            'max_inference_time_ms': str(max_time),
            'num_iterations': str(num_iterations)
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def get_imagenet_classes():
    """
    Get ImageNet class labels

    Returns:
        dict: Class ID to name mapping
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]

    return {
        'num_classes': len(categories),
        'sample_classes': {str(i): categories[i] for i in range(10)}  # First 10 classes as sample
    }
