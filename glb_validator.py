#!/usr/bin/env python3
"""
GLB Validation Utility
Provides Python-based validation for GLB files using multiple validation methods.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from pygltflib import GLTF2
    from pygltflib.validator import validate as pygltf_validate, summary as pygltf_summary
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    PYGLTFLIB_AVAILABLE = False
    print("Warning: pygltflib not available. Install with: uv pip install pygltflib")


@dataclass
class ValidationResult:
    """Results from GLB validation."""
    is_valid: bool
    errors: int
    warnings: int
    infos: int
    hints: int
    messages: List[str]
    validation_time_ms: Optional[int] = None
    method: str = "unknown"


class GLBValidator:
    """GLB file validator with multiple validation backends."""
    
    def __init__(self, validator_path: Optional[str] = None):
        """Initialize validator with optional path to gltf_validator binary."""
        self.validator_path = validator_path or self._find_validator()
        
    def _find_validator(self) -> Optional[str]:
        """Find the gltf_validator executable."""
        # Check in tools directory first
        tools_validator = Path(__file__).parent / "tools" / "gltf_validator"
        if tools_validator.exists():
            return str(tools_validator)
            
        # Check if it's in PATH
        try:
            subprocess.run(["gltf_validator", "--help"], 
                         capture_output=True, check=True)
            return "gltf_validator"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return None
        
    def _get_validator_download_instructions(self) -> str:
        """Get instructions for downloading the validator."""
        return """
To download the official glTF validator:

1. Create tools directory: mkdir -p tools && cd tools
2. Download: curl -L -o gltf_validator.tar.xz https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.10/gltf_validator-2.0.0-dev.3.10-linux64.tar.xz
3. Extract: tar -xf gltf_validator.tar.xz
4. Return to project: cd ..

Or install pygltflib as fallback: uv pip install pygltflib
"""
        
    def validate_with_official_validator(self, glb_path: str) -> ValidationResult:
        """Validate using the official Khronos glTF validator."""
        if not self.validator_path:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=[f"Official glTF validator not found.{self._get_validator_download_instructions()}"],
                method="official_validator"
            )
            
        try:
            # Run validator with JSON output
            result = subprocess.run(
                [self.validator_path, glb_path, "--stdout"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Try to parse JSON output
                try:
                    report = json.loads(result.stdout)
                    issues = report.get("issues", {})
                    
                    return ValidationResult(
                        is_valid=issues.get("numErrors", 0) == 0,
                        errors=issues.get("numErrors", 0),
                        warnings=issues.get("numWarnings", 0),
                        infos=issues.get("numInfos", 0),
                        hints=issues.get("numHints", 0),
                        messages=[msg.get("message", "") for msg in issues.get("messages", [])],
                        validation_time_ms=report.get("validationTime"),
                        method="official_validator"
                    )
                except json.JSONDecodeError:
                    # Fallback to simple text parsing
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if "Errors:" in line:
                            parts = line.split(',')
                            errors = int(parts[0].split(':')[1].strip())
                            warnings = int(parts[1].split(':')[1].strip()) if len(parts) > 1 else 0
                            
                            return ValidationResult(
                                is_valid=errors == 0,
                                errors=errors,
                                warnings=warnings,
                                infos=0,
                                hints=0,
                                messages=[result.stdout],
                                method="official_validator"
                            )
                            
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=[f"Validator failed: {result.stderr}"],
                method="official_validator"
            )
            
        except subprocess.TimeoutExpired:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=["Validation timed out"],
                method="official_validator"
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=[f"Validation error: {str(e)}"],
                method="official_validator"
            )
            
    def validate_with_pygltflib(self, glb_path: str) -> ValidationResult:
        """Validate using pygltflib (basic validation)."""
        if not PYGLTFLIB_AVAILABLE:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=["pygltflib not available"],
                method="pygltflib"
            )
            
        try:
            gltf = GLTF2().load(glb_path)
            
            # Capture validation output
            import io
            import contextlib
            
            output = io.StringIO()
            error_occurred = False
            
            try:
                with contextlib.redirect_stdout(output):
                    pygltf_validate(gltf)
                    pygltf_summary(gltf)
            except Exception as e:
                error_occurred = True
                output.write(f"Validation error: {str(e)}")
                
            validation_output = output.getvalue()
            
            return ValidationResult(
                is_valid=not error_occurred,
                errors=1 if error_occurred else 0,
                warnings=0,
                infos=0,
                hints=0,
                messages=[validation_output] if validation_output else ["Validation passed"],
                method="pygltflib"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=[f"Failed to load GLB: {str(e)}"],
                method="pygltflib"
            )
            
    def validate(self, glb_path: str, method: str = "auto") -> ValidationResult:
        """Validate GLB file using specified method or auto-detect best available."""
        glb_path = str(Path(glb_path).resolve())
        
        if not Path(glb_path).exists():
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=[f"File not found: {glb_path}"],
                method="file_check"
            )
            
        if method == "official" or (method == "auto" and self.validator_path):
            return self.validate_with_official_validator(glb_path)
        elif method == "pygltflib" or (method == "auto" and PYGLTFLIB_AVAILABLE):
            return self.validate_with_pygltflib(glb_path)
        else:
            return ValidationResult(
                is_valid=False,
                errors=1,
                warnings=0,
                infos=0,
                hints=0,
                messages=["No validation method available"],
                method="none"
            )
            
    def print_validation_report(self, result: ValidationResult, file_path: str) -> None:
        """Print a formatted validation report."""
        print(f"\n🔍 GLB Validation Report: {Path(file_path).name}")
        print(f"📁 Path: {file_path}")
        print(f"🔧 Method: {result.method}")
        print("-" * 50)
        
        if result.is_valid:
            print("✅ VALID - No errors found!")
        else:
            print("❌ INVALID - Errors detected!")
            
        print(f"📊 Stats:")
        print(f"   Errors:   {result.errors}")
        print(f"   Warnings: {result.warnings}")
        print(f"   Infos:    {result.infos}")
        print(f"   Hints:    {result.hints}")
        
        if result.validation_time_ms:
            print(f"   Time:     {result.validation_time_ms}ms")
            
        if result.messages:
            print(f"\n📝 Messages:")
            for i, message in enumerate(result.messages, 1):
                if message.strip():
                    print(f"   {i}. {message}")
                    
        print("-" * 50)


def validate_glb_file(glb_path: str, method: str = "auto", verbose: bool = True) -> bool:
    """Convenience function to validate a GLB file and return True if valid."""
    validator = GLBValidator()
    result = validator.validate(glb_path, method)
    
    if verbose:
        validator.print_validation_report(result, glb_path)
        
    return result.is_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GLB files")
    parser.add_argument("file", help="Path to GLB file to validate")
    parser.add_argument("--method", choices=["auto", "official", "pygltflib"], 
                       default="auto", help="Validation method to use")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Only show validation result, not full report")
    
    args = parser.parse_args()
    
    is_valid = validate_glb_file(args.file, args.method, not args.quiet)
    sys.exit(0 if is_valid else 1)
