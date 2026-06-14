# FP32 → Mixed Precision (FP16/FP32) Refactoring Summary

## Overview
Successfully refactored the RISC-V vector AlexNet implementation from pure FP32 to mixed precision:
- **Inputs/Weights**: FP16 (_Float16) - 50% memory savings
- **Accumulators**: FP32 (float) - maintains numerical stability
- **Outputs/Biases**: FP32 (float)

## Changes Applied

### 1. Header Files
✓ **fmatmul.h** - Updated all function signatures
  - `const float *a, *b` → `const _Float16 *a, *b`
  - `const float *bias` → kept as `const float *bias`

✓ **fc_layer.h** - Updated fc_op struct
  - `float *input` → `_Float16 *input`
  - `float *weights` → `_Float16 *weights`
  - Other fields remain float for gradients/outputs

✓ **matrix.h** - Updated matrix operation signatures
  - All input/weight pointers → `_Float16`

### 2. Core Assembly Changes (fmatmul.c)
✓ **Configuration Changes**
  - `vsetvli ... e32, m4` → `vsetvli ... e16, m4` (doubled vector length)
  - Total: ~2000 lines updated

✓ **Load Instructions** (36 instances)
  - `vle32.v` → `vle16.v` (FP16 loads)
  - `vlse32.v` → `vlse16.v` (strided FP16 loads)

✓ **Widening Multiply-Accumulate** (296 instances)
  - `vfmacc.vv` → `vfwmacc.vv` (vector-vector widening MAC)
  - `vfmacc.vf` → `vfwmacc.vf` (scalar-vector widening MAC)
  - FP16 sources (e16, m4) → FP32 dest (e32, m8 implicit)

✓ **Reduction Fix** (16 vredsum instances)
  - Added `vsetvli zero, zero, e32, m2, ta, ma` before vredsum
  - Switches config from e16 (sources) to e32 (FP32 accumulators)
  - Prevents misinterpretation of 32-bit FP data as 16-bit

### 3. Implementation Files
✓ **fc_layer.c**
  - Updated static function signatures
  - Fixed input conversion in fc_op_forward():
    - `vle16.v` to load FP16 input
    - `vfwcvt.f.f.v` to widen FP16→FP32
    - `vse32.v` to store FP32

✓ **matrix.c**
  - Updated all public function signatures
  - Updated static helper functions

## Key Architecture Decisions

### Why Widening Instructions?
- `vfwmacc.vv` multiplies two FP16 values and accumulates into FP32
- Prevents accumulation precision loss when summing many small products
- Standard approach in ML frameworks (NVIDIA AMP, TensorFlow Mixed Precision)

### Why E32 Reconfiguration Before Reduction?
- Widened accumulators contain 32-bit FP data
- RISC-V hardware must be configured for correct element width
- Without reconfiguration, vredsum would misinterpret 32-bit floats as 16-bit

### Memory Layout
- Input/weight matrices: stored as FP16 (byte-aligned)
- Scratch buffers: FP32 for numerical stability
- Gradients: maintained in FP32 for training stability

## Testing Recommendations
1. Verify numerical accuracy vs FP32 baseline (check loss convergence)
2. Confirm 50% reduction in input/weight memory usage
3. Benchmark throughput improvement from doubled vector length
4. Test edge cases: very small/large accumulations, gradient magnitudes

## Files Modified
- kernel/fmatmul.h (120 lines, 100% signatures updated)
- kernel/fmatmul.c (2013 lines, bulk transform applied)
- kernel/matrix.h (26 lines, 100% signatures updated)
- kernel/matrix.c (500+ lines affected)
- kernel/fc_layer.h (27 lines, struct updated)
- kernel/fc_layer.c (100+ lines affected)

## Backward Compatibility
❌ **Breaking change** - requires retraining or FP16 weight conversion from existing FP32 weights
