#!/usr/bin/env python3
"""
Convert weights between PyTorch and MLX formats.

Enables model weights to be used with either backend:
- PyTorch: Universal compatibility (CUDA, MPS, CPU)
- MLX: Optimized for Apple Silicon (unified memory)

Usage:
    # PyTorch → MLX
    python convert_weights.py --input models/transformer_v1/transformer_model_v1.pt --to-mlx

    # MLX → PyTorch
    python convert_weights.py --input models/transformer_v1/transformer_model_v1_mlx.safetensors --to-pytorch

    # Batch convert all transformer models
    python convert_weights.py --convert-all
"""

import argparse
from pathlib import Path
import numpy as np


def pytorch_to_mlx(pt_path: str, mlx_path: str = None):
    """
    Convert PyTorch .pt weights to MLX-compatible safetensors format.

    Args:
        pt_path: Path to PyTorch model weights (.pt file)
        mlx_path: Output path (default: replaces .pt with _mlx.safetensors)
    """
    import torch

    if mlx_path is None:
        mlx_path = pt_path.replace('.pt', '_mlx.safetensors')

    print(f"Converting PyTorch → MLX")
    print(f"  Input:  {pt_path}")
    print(f"  Output: {mlx_path}")

    # Load PyTorch weights
    state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
    print(f"  Loaded {len(state_dict)} parameters")

    # Convert to numpy arrays
    weights = {}
    for name, tensor in state_dict.items():
        np_array = tensor.numpy()
        weights[name] = np_array
        # print(f"    {name}: {np_array.shape}")

    # Save as safetensors
    try:
        from safetensors.numpy import save_file
        save_file(weights, mlx_path)
        print(f"  Saved as safetensors format")
    except ImportError:
        # Fallback to numpy format
        npz_path = mlx_path.replace('.safetensors', '.npz')
        np.savez(npz_path, **weights)
        print(f"  Saved as numpy format (safetensors not installed)")
        mlx_path = npz_path

    print(f"✓ Conversion complete: {mlx_path}")
    return mlx_path


def mlx_to_pytorch(mlx_path: str, pt_path: str = None):
    """
    Convert MLX safetensors/npz weights to PyTorch .pt format.

    Args:
        mlx_path: Path to MLX model weights (.safetensors or .npz file)
        pt_path: Output path (default: replaces _mlx.safetensors with .pt)
    """
    import torch

    if pt_path is None:
        pt_path = mlx_path.replace('_mlx.safetensors', '.pt').replace('_mlx.npz', '.pt')

    print(f"Converting MLX → PyTorch")
    print(f"  Input:  {mlx_path}")
    print(f"  Output: {pt_path}")

    # Load MLX weights
    if mlx_path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            weights = {}
            with safe_open(mlx_path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        except ImportError:
            raise ImportError("safetensors package required. Install with: pip install safetensors")
    elif mlx_path.endswith('.npz'):
        data = np.load(mlx_path)
        weights = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported format: {mlx_path}")

    print(f"  Loaded {len(weights)} parameters")

    # Convert to PyTorch tensors
    state_dict = {}
    for name, np_array in weights.items():
        state_dict[name] = torch.from_numpy(np_array.copy())

    # Save as PyTorch state dict
    torch.save(state_dict, pt_path)
    print(f"✓ Conversion complete: {pt_path}")
    return pt_path


def convert_all_transformers():
    """Convert all transformer model weights in both directions."""
    models_dir = Path('models/transformer_v1')

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    # Find PyTorch models to convert to MLX
    pt_files = list(models_dir.glob('*.pt'))
    for pt_file in pt_files:
        if '_best' not in str(pt_file):  # Skip intermediate checkpoints
            mlx_path = str(pt_file).replace('.pt', '_mlx.safetensors')
            if not Path(mlx_path).exists():
                print(f"\n--- Converting {pt_file.name} ---")
                try:
                    pytorch_to_mlx(str(pt_file), mlx_path)
                except Exception as e:
                    print(f"  Error: {e}")

    # Find MLX models without PyTorch equivalents
    mlx_files = list(models_dir.glob('*_mlx.safetensors')) + list(models_dir.glob('*_mlx.npz'))
    for mlx_file in mlx_files:
        pt_path = str(mlx_file).replace('_mlx.safetensors', '.pt').replace('_mlx.npz', '.pt')
        if not Path(pt_path).exists():
            print(f"\n--- Converting {mlx_file.name} ---")
            try:
                mlx_to_pytorch(str(mlx_file), pt_path)
            except Exception as e:
                print(f"  Error: {e}")


def verify_conversion(pt_path: str, mlx_path: str):
    """Verify that converted weights match the original."""
    import torch

    print(f"\nVerifying conversion...")
    print(f"  PyTorch: {pt_path}")
    print(f"  MLX:     {mlx_path}")

    # Load PyTorch weights
    pt_state = torch.load(pt_path, map_location='cpu', weights_only=True)

    # Load MLX weights
    if mlx_path.endswith('.safetensors'):
        from safetensors import safe_open
        mlx_weights = {}
        with safe_open(mlx_path, framework="numpy") as f:
            for key in f.keys():
                mlx_weights[key] = f.get_tensor(key)
    else:
        data = np.load(mlx_path)
        mlx_weights = {key: data[key] for key in data.files}

    # Compare
    all_match = True
    for name, pt_tensor in pt_state.items():
        if name not in mlx_weights:
            print(f"  Missing in MLX: {name}")
            all_match = False
            continue

        pt_np = pt_tensor.numpy()
        mlx_np = mlx_weights[name]

        if pt_np.shape != mlx_np.shape:
            print(f"  Shape mismatch for {name}: {pt_np.shape} vs {mlx_np.shape}")
            all_match = False
            continue

        if not np.allclose(pt_np, mlx_np, rtol=1e-5, atol=1e-6):
            max_diff = np.max(np.abs(pt_np - mlx_np))
            print(f"  Value mismatch for {name}: max diff = {max_diff}")
            all_match = False

    if all_match:
        print("✓ All weights match!")
    else:
        print("✗ Some weights don't match")

    return all_match


def main():
    parser = argparse.ArgumentParser(description='Convert weights between PyTorch and MLX formats')
    parser.add_argument('--input', type=str, help='Input model weights file')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--to-mlx', action='store_true', help='Convert PyTorch to MLX')
    parser.add_argument('--to-pytorch', action='store_true', help='Convert MLX to PyTorch')
    parser.add_argument('--convert-all', action='store_true', help='Convert all transformer models')
    parser.add_argument('--verify', action='store_true', help='Verify conversion matches')

    args = parser.parse_args()

    if args.convert_all:
        convert_all_transformers()
        return

    if not args.input:
        parser.print_help()
        return

    if args.to_mlx:
        mlx_path = pytorch_to_mlx(args.input, args.output)
        if args.verify:
            verify_conversion(args.input, mlx_path)

    elif args.to_pytorch:
        pt_path = mlx_to_pytorch(args.input, args.output)
        if args.verify:
            verify_conversion(pt_path, args.input)

    else:
        # Auto-detect direction based on file extension
        if args.input.endswith('.pt'):
            mlx_path = pytorch_to_mlx(args.input, args.output)
            if args.verify:
                verify_conversion(args.input, mlx_path)
        elif args.input.endswith('.safetensors') or args.input.endswith('.npz'):
            pt_path = mlx_to_pytorch(args.input, args.output)
            if args.verify:
                verify_conversion(pt_path, args.input)
        else:
            print(f"Unknown file format: {args.input}")
            print("Use --to-mlx or --to-pytorch to specify conversion direction")


if __name__ == '__main__':
    main()
