/* FPGA (Cheshire) linker script — template; do not use directly.
   Build system copies this to link.fpga.ld and replaces 8
   with 4*nr_lanes (AxiWideBeWidth) via align_sections.sh.

   Key differences from arch.link.ld (Verilator):
     ORIGIN  = 0xC0000000  (LlcOutRegionStart — LLC scratchpad)
     LENGTH  = 0x00800000  (8 MB — adjust to match your LLC size)
     fake_uart             = 0x03002000  (Cheshire UART TX FIFO)

   crt0.S is shared with Verilator but three patterns required #ifdef FPGA:
     _stack_top            — la sp, _stack_top  replaces the Verilator
                             ld sp, 0(dram_end_address_reg) indirection
                             (no TB on FPGA to pre-load that register).
     eoc_address_reg       — crt0 writes the return code here and loops;
                             points to LLC scratchpad (harmless on FPGA).
     stvec_handler / rvtest_init / mtvec_handler weak-symbol LA calls are
     skipped on FPGA because undefined weak symbols resolve to 0x0, and
     the AUIPC PCREL offset from 0xC0000000 to 0x0 is -3 GB — beyond the
     ±2 GB PCREL range.  Verilator code at 0x80000000 is exactly at -2 GB,
     which just fits.

   event_trigger / hw_cnt_en_reg are aliased to LLC scratchpad so apps
   using runtime.h still link.  Adjust offsets if LLC < 8 MB.
*/

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  LLC : ORIGIN = 0x40000000, LENGTH = 0x20000000
}

SECTIONS {
  .text : {
    *(.text.init)
    *(.text)
    *(.text.*)
  } > LLC

  .data : ALIGN(8) {
    *(.data)
    *(.data.*)
  } > LLC

  .rodata  : ALIGN(8) { *(.rodata .rodata.* .gnu.linkonce.r.*) } > LLC
  .rodata1 : ALIGN(8) { *(.rodata1) } > LLC
  .sdata2  : ALIGN(8) {
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
  } > LLC

  .sdata   : ALIGN(8) {
    __global_pointer$ = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } > LLC

  .bss : ALIGN(8) {
    __bss_start = .;
    *(.bss)
    *(.sbss .sbss.* .sbss2 .sbss2.* .gnu.linkonce.sb2.*);
    __bss_end = .;
  } > LLC

  .l2 : ALIGN(8) {
    *(.l2)
    timer = .;
    . = . + 0x8;
    l2_alloc_base = ALIGN(8);
  } > LLC

  .comment : ALIGN(8) { *(.comment) } > LLC

  /* Top of stack: crt0.S does  la sp, _stack_top  under #ifdef FPGA.
     = ORIGIN(LLC) + LENGTH(LLC); adjust if your LLC is smaller than 8 MB. */
  _stack_top = 0x60000000;

  /* crt0.S writes the return code here and loops.  Points to LLC scratchpad
     so the write is harmless on FPGA. */
  eoc_address_reg = 0x5FFFF010;

  /* Cheshire UART TX FIFO — used by serial.c under #ifdef FPGA. */
  fake_uart = 0x03002000;

  /* Verilator TB register aliases → LLC scratchpad (harmless on FPGA). */
  event_trigger = 0x5FFFF000;
  hw_cnt_en_reg = 0x5FFFF008;
}
