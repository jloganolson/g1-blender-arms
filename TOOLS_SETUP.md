# External Tools Setup

This project uses external validation tools that are not included in the repository.

## glTF Validator

The GLB validation functionality requires the official Khronos glTF Validator.

### Automatic Setup
The `glb_validator.py` will automatically fall back to `pygltflib` if the official validator is not available.

### Manual Setup (Recommended)
For the most comprehensive validation, download the official validator:

```bash
# Create tools directory
mkdir -p tools && cd tools

# Download the validator (Linux x64)
curl -L -o gltf_validator.tar.xz https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.10/gltf_validator-2.0.0-dev.3.10-linux64.tar.xz

# Extract
tar -xf gltf_validator.tar.xz

# Return to project root
cd ..
```

### Alternative Platforms
For other platforms, visit: https://github.com/KhronosGroup/glTF-Validator/releases

### Fallback Option
If you don't want to download the validator, install the Python fallback:

```bash
uv pip install pygltflib
```

## Why Not Include in Git?

The validator binary is:
- ✗ **Large** (6MB+ extracted)
- ✗ **Platform-specific** (Linux binary won't work on Windows/Mac)  
- ✗ **Frequently updated** (new releases regularly)
- ✓ **Easy to download** (automated instructions provided)

The `tools/` directory is gitignored to keep the repository clean and portable.
