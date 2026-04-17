#include <stdint.h>

// #define UART_MMIO_ADDR 0xC0000000UL

#ifdef SPIKE
static volatile uint64_t spike_magic_mem[8] __attribute__((aligned(64)));
static volatile char spike_out_char;
#endif
extern char fake_uart;

void _putchar(char character) {
#ifdef SPIKE
    extern volatile uint64_t tohost;
    extern volatile uint64_t fromhost;
    //same code as syscall func in syscalls.c
    spike_out_char = character;
    spike_magic_mem[0] = 64;                       // SYS_write
    spike_magic_mem[1] = 1;                        // stdout fd
    spike_magic_mem[2] = (uintptr_t)&spike_out_char;
    spike_magic_mem[3] = 1;                        // one byte

    __sync_synchronize();
    tohost = (uintptr_t)spike_magic_mem;
    while (fromhost == 0)
        ;
    fromhost = 0;
    __sync_synchronize();
#else
    // *(volatile uint8_t *)UART_MMIO_ADDR = (uint8_t)character; //initial there was another solution
    fake_uart = character;
#endif
}