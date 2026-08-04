# fpnew & Ara FPU — Architecture Notes

## Ιεραρχία Module

```
fpnew_top  (= 1 FPU ανά Ara lane, src: deps/fpnew/src/fpnew_top.sv)
│
├── fpnew_opgroup_block [ADDMUL]      (deps/fpnew/src/fpnew_opgroup_block.sv)
│   ├── fpnew_opgroup_fmt_slice [FP16]  → 4 × fpnew_fma  (PARALLEL)
│   ├── fpnew_opgroup_fmt_slice [FP32]  → 2 × fpnew_fma  (PARALLEL)
│   └── fpnew_opgroup_fmt_slice [FP64]  → 1 × fpnew_fma  (PARALLEL)
│
├── fpnew_opgroup_block [NONCOMP]
│   ├── fpnew_opgroup_fmt_slice [FP16]  → 4 × NONCOMP unit  (PARALLEL)
│   ├── fpnew_opgroup_fmt_slice [FP32]  → 2 × NONCOMP unit  (PARALLEL)
│   └── fpnew_opgroup_fmt_slice [FP64]  → 1 × NONCOMP unit  (PARALLEL)
│
├── fpnew_opgroup_block [DIVSQRT]
│   └── fpnew_opgroup_multifmt_slice    → 1 shared unit (MERGED, super_format)
│
├── fpnew_opgroup_block [CONV]
│   └── fpnew_opgroup_multifmt_slice    → 4 × fpnew_cast_multi (MERGED)
│       (NUM_LANES = 4 για FP16+FP32+FP64, μόνο 2 active για FP32→FP16)
│
└── fpnew_opgroup_block [DOTP]          → DISABLED στο Ara
```

**Πολλαπλασιαστής ανά Ara system:** ο αριθμός των `fpnew_top` instances = `NrLanes`.

---

## PARALLEL vs MERGED

| | PARALLEL | MERGED |
|---|---|---|
| Κυκλώματα | Ένα slice ανά format | Ένα κοινό slice για όλα |
| Μέγεθος unit | Ακριβώς στο format (π.χ. 16-bit) | super_format = max(exp,man) ≈ FP64-sized |
| Throughput | Formats τρέχουν ταυτόχρονα | Serialize ανά format |
| Area | Περισσότερο | Λιγότερο |
| Opgroups | ADDMUL, NONCOMP | DIVSQRT, CONV |

**super_format** (fpnew_pkg.sv:394): `max(exp_bits)` + `max(man_bits)` σε όλα τα enabled formats.

---

## Format Encodings (fpnew_pkg.sv:25-28)

| Format | Bits | Exp | Man | Ισοδύναμο |
|--------|------|-----|-----|----------|
| FP32   | 32   | 8   | 23  | IEEE binary32 |
| FP64   | 64   | 11  | 52  | IEEE binary64 |
| FP16   | 16   | 5   | 10  | IEEE binary16 |
| FP8    | 8    | 5   | 2   | E5M2 |
| FP16ALT| 16   | 8   | 7   | **BFloat16** |
| FP8ALT | 8    | 4   | 3   | E4M3 |

---

## FPUSupport Options (ara_pkg.sv:49-58)

6-bit mask: `{FP64, FP32, FP16, FP16ALT, FP8, FP8ALT}`

| Enum | Bits | Formats |
|------|------|---------|
| `FPUSupportNone` | `000000` | καμία |
| `FPUSupportHalf` | `001000` | FP16 |
| `FPUSupportSingle` | `010000` | FP32 |
| `FPUSupportHalfSingle` | `011000` | FP16+FP32 |
| `FPUSupportDouble` | `100000` | FP64 |
| `FPUSupportSingleDouble` | `110000` | FP32+FP64 |
| **`FPUSupportHalfSingleDouble`** | `111000` | FP16+FP32+FP64 **← τρέχον** |
| `FPUSupportAll` | `111111` | όλα |

**Τρέχον build:** `-GFPUSupport=6'h38` (build/verilator/Vlane_7_hierMkArgs.f:395)

**Σημείο αλλαγής:** `ara_soc.sv:15` (parameter default), μεταδίδεται αυτόματα σε όλη την ιεραρχία.

> ⚠️ FP8ALT: η CVA6 δεν το υποστηρίζει (`ara_soc.sv:461` — commented out). Αν ενεργοποιηθεί, πυροδοτείται assertion.

---

## Opgroups & Operations

### ADDMUL (PARALLEL)
- `FMADD`, `FNMSUB` ← από VFMACC, VFNMACC, VFMSAC, VFNMSAC, VFMADD, VFNMADD, VFMSUB, VFNMSUB
- Dispatcher: `vmfpu.sv:910-923`

### DIVSQRT (MERGED)
- `VFDIV`, `VFRDIV`, `VFSQRT`

### NONCOMP (PARALLEL)
- `SGNJ`: VFSGNJ, VFSGNJN, VFSGNJX — bit manipulation του sign
- `MINMAX`: VFMIN, VFMAX
- `CMP`: VMFEQ, VMFLT, VMFLE
- `CLASSIFY`: VFCLASS

### CONV (MERGED)
- `F2F`: VFCVTFF, VFNCVTRODFF — float-to-float (narrowing/widening)
- `F2I`: VFCVTXUF, VFCVTXF, VFCVTRTZXUF, VFCVTRTZXF
- `I2F`: VFCVTFXU, VFCVTFX
- `CPKAB`, `CPKCD`: Cast-and-Pack (PULP extension, δεν εκτίθεται στο Ara ISA)

### DOTP — DISABLED
- `'{default: DISABLED}` στο `vmfpu.sv:867`

---

## FPU Configuration (vmfpu.sv:846-869)

```systemverilog
FPUFeatures = '{
  Width: 64,         // 64-bit lane width
  EnableVectors: 1,
  FpFmtMask: {RVVF, RVVD, RVVH, RVVB, RVVHA, RVVBA}
};

FPUImplementation = '{
  UnitTypes: '{
    '{default: PARALLEL}, // ADDMUL
    '{default: MERGED},   // DIVSQRT
    '{default: PARALLEL}, // NONCOMP
    '{default: MERGED},   // CONV
    '{default: DISABLED}  // DOTP
  },
  PipeConfig: DISTRIBUTED
};
```

---

## SIMD Lanes ανά Format (Width=64)

`num_lanes(Width, fmt) = Width / fp_width(fmt)`

| Format | fp_width | Lanes (ADDMUL/NONCOMP PARALLEL) |
|--------|---------|--------------------------------|
| FP16   | 16      | **4** |
| FP32   | 32      | **2** |
| FP64   | 64      | **1** |

`FPULanes = max_num_lanes(64, mask) = 64/16 = 4` (vmfpu.sv:1074)

**MERGED slices** (CONV): `NUM_LANES = 4`, αλλά ενεργά lanes ανά operation:

| Operation | Ενεργά lanes |
|---|---|
| vfncvt FP64→FP32 (src=FP64) | 1 |
| vfncvt FP32→FP16 (src=FP32) | 2 |
| vfcvt FP16↔INT16 (src/dst=FP16) | **4** |

---

## Pipeline Latencies (ara_pkg.sv:89-100)

| Operation | Cycles | Throughput |
|-----------|--------|-----------|
| FP16 FMA (ADDMUL) | 3 | 1/cycle |
| FP32 FMA (ADDMUL) | 4 | 1/cycle |
| FP64 FMA (ADDMUL) | 5 | 1/cycle |
| FP8 FMA | 2 | 1/cycle |
| DIV/SQRT | 3 | (variable) |
| NONCOMP (min/max/cmp) | 1 | 1/cycle |
| CONV (vfcvt/vfncvt) | 2 | 1/cycle* |

*`vfncvt` narrowing: **2 passes ανά 64-bit word** → πρακτικά 4 cycles/word.

---

## Narrowing Mechanism (vfncvt FP32→FP16)

Πρόβλημα: FPU παράγει 2×FP16 = 32 bits, αλλά destination word = 64 bits.

**Pass 0** (`narrowing_select=0`): 2 FP16 → bits[31:0], `be=8'b00001111`
**Pass 1** (`narrowing_select=1`): 2 FP16 → bits[63:32], `be=8'b11110000`

Το result_queue entry γίνεται valid μόνο όταν:
- `narrowing_select_out_q == 1` (pass 1 complete), **ή**
- `to_process_cnt_d == 0` (τελευταίο στοιχείο — early flush)

Κώδικας: `vmfpu.sv:1686-1700`

---

## Widening — Δεν χρειάζεται 2 passes

FP16→FP32: 2 FP16 in (32 bits) → 2 FP32 out (64 bits) → γεμίζει το destination σε 1 pass.

---

## in_valid Mechanism (PARALLEL activation)

```
opgroup_block.sv:101:
  assign in_valid = in_valid_i & (dst_fmt_i == fmt);   // PARALLEL: μόνο matching format

opgroup_block.sv:178:
  assign in_valid = in_valid_i & (FmtUnitTypes[dst_fmt_i] == MERGED);  // MERGED gate
```

Μέσα στο `fpnew_fma.sv`:
```
reg_ena = inp_pipe_ready[i] & inp_pipe_valid_q[i];
FFL(register, source, reg_ena, reset)
→ όταν in_valid=0: reg_ena=0, κανένα flip-flop δεν αλλάζει → μηδενικό dynamic power
```

---

## vfwmacc (Widening MACC, FP16→FP32)

Dispatcher (`ara_dispatcher.sv:2509-2516`) μετατρέπει σε:
```systemverilog
ara_req.op         = VFMACC;
ara_req.vtype.vsew = EW16.next() = EW32;       // FPU βλέπει FP32
ara_req.conversion_vs1 = OpQueueConversionWideFP2;  // FP16→FP32 στο operand queue
ara_req.conversion_vs2 = OpQueueConversionWideFP2;
ara_req.eew_vd_op  = EW32;                      // destination = EW32
```

Ροή: **FP16 bits → operand_queue (combinational expand) → FP32 PARALLEL slice (2 FMADDs)**

Δεν χρησιμοποιεί CONV/MERGED. Αποτέλεσμα FP32.

---

## LMUL & vsetvli

LMUL αλλάζει **μόνο** μέσω `vsetvli`/`vsetvl`/`vsetivli`.

**WAIT_IDLE** (`ara_dispatcher.sv:695`) ενεργοποιείται όταν LMUL **μειώνεται**:
```systemverilog
if (!csr_vtype_q.vlmul[2] && (csr_vtype_d.vlmul[2:0] < csr_vtype_q.vlmul[2:0]))
  state_d = WAIT_IDLE;
```

| Αλλαγή | Stall |
|---|---|
| Αύξηση LMUL | Όχι |
| Μείωση LMUL (≥LMUL_1 → μικρότερο) | **Ναι — αναμένει `ara_idle_i`** |

Κόστος μείωσης: **εκατοντάδες cycles** αν υπάρχουν in-flight instructions.

---

## Αρχεία Αναφοράς

| Αρχείο | Περιεχόμενο |
|--------|------------|
| `deps/fpnew/src/fpnew_pkg.sv` | Format enums, encodings, utility functions, DEFAULT_NOREGS |
| `deps/fpnew/src/fpnew_top.sv` | Top-level FPU, opgroup routing, output arbiter |
| `deps/fpnew/src/fpnew_opgroup_block.sv` | PARALLEL slices (gen:90), MERGED slice (gen:171), in_valid gating |
| `deps/fpnew/src/fpnew_opgroup_fmt_slice.sv` | Single-format SIMD lanes, fpnew_fma instantiation |
| `deps/fpnew/src/fpnew_opgroup_multifmt_slice.sv` | Multi-format MERGED unit, narrowing CPK logic, cast instantiation |
| `deps/fpnew/src/fpnew_fma.sv` | FMA pipeline, reg_ena mechanism, pipeline stages |
| `deps/fpnew/src/fpnew_cast_multi.sv` | F2F/F2I/I2F unified conversion pipeline |
| `include/ara_pkg.sv` | FPUSupport enum, latency localparams (LatFComp*, LatFConv...) |
| `src/lane/vmfpu.sv` | FPUFeatures/FPUImplementation config, narrowing_select FSM, op dispatch |
| `src/ara_dispatcher.sv` | Instruction decode, vfwmacc→VFMACC transform, LMUL stall FSM |
| `src/lane/operand_queue.sv` | OpQueueConversionWideFP2: FP16→FP32 bit expansion |
| `src/ara_soc.sv` | Top-level FPUSupport parameter entry point |
