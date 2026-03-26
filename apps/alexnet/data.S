.section .rodata

.global cifar10_data
.align 4
cifar10_data:
    .incbin "alexnet/kernel/cifar10_data.bin"

.global cifar10_offsets
.align 4
cifar10_offsets:
    .incbin "alexnet/kernel/cifar10_offsets.bin"

.global cifar10_labels
.align 4
cifar10_labels:
    .incbin "alexnet/kernel/cifar10_labels.bin"