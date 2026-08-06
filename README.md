# AVX512 BMM Introduction

AVX512 BMM is an instruction set extension that adds bit matrix multiplies and a bit reversal, which is expected to debut with the upcoming Zen 6. This was presumably intended for accelerating 1-bit LLMs however it appears to be more widely useful for generalized bit manipulation through the bit matrix multiples.

There also exists the bit reversal within a byte boundary instruction `VBITREV`. It will likely have slightly better performance characteristics then the current GFNI method and doesn't rely on a constant operand from memory.

# Bit Matrix Multiply Applications

The bit matrix multiples are very interesting. There are two variations `VBMACOR16X16X16` and `VBMACXOR16X16X16`, each of which are finite field-like matrix multiplies and accumulates (addition). The XOR version is very similar to the GFNI instruction `GF2P8AFFINEQB` but instead operates at a 16-bit granularity, is column-major ordered, and have a variable vector operand as the accumulator. There are multiple different ways of interpreting what they do, each of which is useful for a different application:

## Linear algebra interpretation
It performs a matrix multiply between each 16-bit elements in the second operand (interpreted as a 16 by 1-bit column matrix) and the corresponding 256 lane in the third (a column-major 16 by 16-bit matrix), which produces intermediate 16-bit products. Instead of a regular multiply, this is instead done through "carryless" (or modular 1-bit) arithmetic for the XOR version, or boolean algebra (or 1-bit saturation arithmetic) for the OR. The first operand is treated as an addended, which is applied to the product using the same arithmetic structure as the multiply.

Keep in mind that this is a SIMD operation, with a mismatch between the size of the multiplicands so each column matrix within a 256-bit lane shares the same square matrix.

## Arbitrary Bitwise permutations (vector as a source)
The second source is treated as the 16-bit input, with the matrix determining arbitrary combinations of OR/XORs between input bits for every position. This can allow you to, for example, set `bit0 = bit3 ^ bit4`, `bit1 = bit0 ^ bit4 ^ bit7`, etc.
Trivially, this can allow for within-a-lane bit shuffles, which can be used to emulate 16-bit reversals and bit shifts of any type. Probably not super useful due to the well supported nature of 16-bit shifts, but the bit reverse might be slightly faster then a `VBITREV` + `VPSHLDW` or when a reverse needs to be combine with a shift.

The more advantageous configuration is when multiple bits need to be combine using a OR/XOR. This can be used for error-correcting codes, cryptographic mixing/substitutions, and cumulative bitwise operations. Notably, an extended [16,11] Hamming code can be computed with a single instruction which is large enough to be efficient. Both configs will need to set the first operand to be zero if it is not immediately followed by a regular bitwise operation of the same type.

Notably `VBMACOR16X16X16` can be used in 8-bit situations where `GF2P8AFFINEQB`'s XOR permutation needs to be a OR instead. It also has the advantage where a AND can be emulated by using the identity `a & b = ~(~a | ~b)`.

The set bits within each 16-bit element determines the locations where one source bit will be included in the result, with the element's position corresponding to the bit's. So, if the second matrix element is ` 0x8101`, the second input bit will be ORed to output positions 0, 8, and 15. The neutral/identity matrix is comprised of the following `element[i] = 1 << i`. In contrast, this is transposed relative to `GF2P8AFFINEQB` and in a different endian order. Alternatively, you can use the following affine formula:

bmacor16x16x16: `res.word[i].bit[j] = a.word[i].bit[j] | (b.word[i] & c.column[j] != 0)`

bmacxor16x16x16: `res.word[i].bit[j] = a.word[i].bit[j] ^ parity(b.word[i] & c.column[j])`

## Matrix transpose (matrix as a source)
The matrix operand is treated as a 256-bit value, with the second operand selecting which 16-bit lanes will be OR/XORed reduced together. Which lanes are selected is based on set bits of the vector, with the 1st bit corresponding to the 1st lane, 2nd to the second, etc. Notably, multiple of these reductions can be performed at once and placed into separate lanes of the output.

This can be used to performance a single instruction reduction like ` _mm256_reduce_or_epi16 ` to 16-bits or any larger multiples. For 512-bit vectors, they will first need to be manually reduced to 256-bits as they only operate within each 256-bit lane. You can also perform a conditional/masked reduction if the input bits are variable.

Another application is combining a shuffle pattern like `shuffle16(x, IDX1) ^ shuffle16(x, IDX2)` into one operation. The indexes need to be converted to 16-bits (if not already), and then they should be raised to a power of 2, with zeroed lanes being set to zero. It might also be beneficial to combine a regular shuffle + XOR/OR with as well by taking advantage of the "addend" assuming the BMM instructions are fast enough. The shuffle can't cross any 256-bit lane.

# Identities
TODO

# Documentation and Intrinsics

## Pseudo code
TODO

## Intrinsics

### Intel Intrinsics
The matrix instructions lack kmask support, so only unmasked versions exist. Added in GCC 16.1 or Clang 23. Requires target feature `avx512bmm`.
```c
__m256i _mm256_bmacor16x16x16(__m256i, __m256i, __m256i);
__m512i _mm512_bmacor16x16x16(__m512i, __m512i, __m512i);

__m256i _mm256_bmacxor16x16x16(__m256i, __m256i, __m256i);
__m512i _mm512_bmacxor16x16x16(__m512i, __m512i, __m512i);
```

### LLVM
Added in LLVM 23, might get backported.
```llvm
declare <16 x i16> @llvm.x86.avx512.vbmacor.v16hi(<16 x i16>, <16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.x86.avx512.vbmacor.v32hi(<32 x i16>, <32 x i16>, <32 x i16>)

declare <16 x i16> @llvm.x86.avx512.vbmacxor.v16hi(<16 x i16>, <16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.x86.avx512.vbmacxor.v32hi(<32 x i16>, <32 x i16>, <32 x i16>)
```

## References
- GCC patch: https://sourceware.org/pipermail/binutils/2025-November/145449.html
- LLVM patch: https://github.com/llvm/llvm-project/pull/182556
- `AMD64 Bit Matrix Multiply
and Bit Reversal Instructions`: https://docs.amd.com/v/u/en-US/69192-PUB
