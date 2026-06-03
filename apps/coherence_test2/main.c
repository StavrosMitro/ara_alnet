#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#define printf_ printf
#else
#include "printf.h"
#endif

#define ITER 100
#define MAX_ARRAY_SIZE 4096

// Ευθυγραμμισμένοι πίνακες για τέλειο memory burst
float __attribute__((aligned(64))) data_a[MAX_ARRAY_SIZE];
float __attribute__((aligned(64))) data_b[MAX_ARRAY_SIZE];
uint32_t __attribute__((aligned(64))) gather_idx[MAX_ARRAY_SIZE];

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

int main() {
    int64_t t0, t1, diff;
    size_t vl;

    // Ζητάμε το ΑΠΟΛΥΤΟ MAX Vector Length (m8) που αντέχει το hardware σου
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(MAX_ARRAY_SIZE));

    printf_("==========================================\n");
    printf_("   ARA VECTOR MICRO-BENCHMARK SUITE       \n");
    printf_("==========================================\n");
    printf_("Vector Length (VL) for m8: %d elements\n", (int)vl);
    printf_("Iterations per test: %d\n\n", ITER);

    // Αρχικοποίηση δεδομένων
    for (int i = 0; i < MAX_ARRAY_SIZE; i++) {
        data_a[i] = 1.0f;
        data_b[i] = 2.0f;
        gather_idx[i] = (i * 2) * sizeof(float); // Τυχαίο jump ανά 2 elements (σε Bytes!)
    }

    // Φόρτωση αρχικών δεδομένων στους καταχωρητές (Warmup)
    asm volatile("vle32.v v8, (%0)" :: "r"(data_a));
    asm volatile("vle32.v v16, (%0)" :: "r"(data_b));
    asm volatile("vle32.v v24, (%0)" :: "r"(gather_idx));
    
    float dummy_sync; // Για να αναγκάζουμε το sync

    // ---------------------------------------------------------
    // 1. UNIT-STRIDE LOAD (vle32.v) - Μέγιστο Memory Bandwidth
    // ---------------------------------------------------------
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vle32.v v8, (%0)" :: "r"(data_a));
    }
    asm volatile("fence"); // Συγχρονισμός μνήμης
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vle32.v]    Unit-Stride Load: %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    // ---------------------------------------------------------
    // 2. STRIDED LOAD (vlse32.v) - Διάβασμα με κενά
    // ---------------------------------------------------------
    long stride_bytes = 8; // Πηδάμε 1 float
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vlse32.v v8, (%0), %1" :: "r"(data_a), "r"(stride_bytes));
    }
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vlse32.v]   Strided Load    : %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    // ---------------------------------------------------------
    // 3. GATHER LOAD (vluxei32.v) - Διάσπαρτο Διάβασμα
    // ---------------------------------------------------------
    // Τα indices είναι ήδη στον v24
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vluxei32.v v8, (%0), v24" :: "r"(data_a));
    }
    asm volatile("fence");
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vluxei32.v] Gather Load     : %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    // ---------------------------------------------------------
    // 4. FMA (vfmacc.vv) - Ο βασιλιάς του Compute
    // ---------------------------------------------------------
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vfmacc.vv v8, v16, v8"); // v8 = v8 + v16 * v8
    }
    // Συγχρονισμός ALU: Ο CVA6 περιμένει να πάρει το 1ο στοιχείο του v8
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfmacc.vv]  Multiply-Accum  : %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    // ---------------------------------------------------------
    // 5. REDUCTION (vfredsum.vs) - Η "στενωπός" του hardware
    // ---------------------------------------------------------
    // Προετοιμάζουμε τον v0 (Vector) ως scalar accumulator
    asm volatile("vmv.v.i v0, 0");
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vfredsum.vs v0, v8, v0"); 
    }
    asm volatile("vfmv.f.s %0, v0" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vfredsum.vs] Vector Reduce    : %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    // ---------------------------------------------------------
    // 6. VECTOR SLIDE DOWN (vslidedown.vx) - Μετατόπιση
    // ---------------------------------------------------------
    long slide_amount = 1;
    t0 = fc_cycle_count_local();
    for(int i=0; i<ITER; i++) {
        asm volatile("vslidedown.vx v8, v16, %0" :: "r"(slide_amount));
    }
    asm volatile("vfmv.f.s %0, v8" : "=f"(dummy_sync));
    t1 = fc_cycle_count_local();
    diff = t1 - t0;
    printf_("[vslidedown] Vector Slide      : %6ld cycles | ~%4.1f cycles/inst\n", (long)diff, (float)diff/ITER);

    printf_("==========================================\n");
    return 0;
}