#include <immintrin.h>

__m256i _mm256_bmacor16x16x16(__m256i, __m256i, __m256i);
__m256i _mm256_bmacxor16x16x16(__m256i, __m256i, __m256i);



// ========================
//      Basic Examples
// ========================

// Returns the unchanged input (for any width)
__m256i _noOperation_i256(__m256i x) {
    const __m256i IDENTITY_MATRIX = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, IDENTITY_MATRIX);
}

// Bitwise XOR (_mm256_xor_si256)
__m256i _bitwiseXor_i256(__m256i a, __m256i b) {
    const __m256i IDENTITY_MATRIX = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    // The first operand applies a regular XOR after the matrix "multiply"
    // (Zero for most examples as the standalone matrix transform is more useful)
    return _mm256_bmacxor16x16x16(a, b, IDENTITY_MATRIX);
}

// Bit reversal within a 16-bit boundary
__m256i _bitReverse_i16x16(__m256i x) {
    const __m256i REVERSE_MATRIX = _mm256_setr_epi16(
        1<<15, 1<<14, 1<<13, 1<<12, 1<<11, 1<<10, 1<<9, 1<<8, 1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1<<0
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, REVERSE_MATRIX);
}

// Fills each lane with bit 7 (0 indexed, "8th bit")
__m256i _fillWithBit7_i16x16(__m256i x) {
    const __m256i BIT7_FILL_MATRIX = _mm256_set1_epi16(1<<7);

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, BIT7_FILL_MATRIX);
}



// ====================
//      Bit shifts
// ====================

// Logical left shift of all 16-bit elements by 1
__m256i _logicalShiftLeft4_i16x16(__m256i x) {
    const __m256i SHL4_MATRIX = _mm256_setr_epi16(
        // bit0 -> bit4, bit1 -> bit5, ... bit12 -> nowhere, ...
        1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 0, 0, 0, 0
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, SHL4_MATRIX);
}

// Arithmetic right shift of all 16-bit elements by 3
__m256i _arithShiftRight3_i16x16(__m256i x) {
    // ... bit15 -> {bit12, bit13, bit14, bit15}
    const __m256i ASR3_MATRIX = _mm256_setr_epi16(
        0, 0, 0, 1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 0b1111<<12
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, ASR3_MATRIX);
}

// Rotation left shift of all 16-bit elements by 1
__m256i _rotateLeft1_i16x16(__m256i x) {
    const __m256i ROL1_MATRIX = _mm256_setr_epi16(
        1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 1<<0
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, ROL1_MATRIX);
}

// Rotation right shift of all 8-bit elements by 2
__m256i _rotateRight2_i8x32(__m256i x) {
    const __m256i ROR2_MATRIX = _mm256_setr_epi16(
        1<<6,   1<<7,   1<<0,   1<<1,   1<<2,   1<<3,   1<<4,   1<<5,
        1<<14,  1<<15,  1<<8,   1<<9,   1<<10,  1<<11,  1<<12,  1<<13
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), x, ROR2_MATRIX);
}



// ============================
//      Bitwise reductions
// ============================

// Horizontal OR between all 16-bit lanes
short _orReduce_i16x16(__m256i x) {
    // Could also use AllOnes vector as we only extract the least element
    const __m256i REDUCE_256 =_mm256_castsi128_si256(_mm_cvtsi32_si128(0x0000ffff));

    __m256i vecReduce = _mm256_bmacor16x16x16(_mm256_setzero_si256(), REDUCE_256, x);
    return (short)_mm256_cvtsi256_si32(vecReduce);
}

// Horizontal XOR between all 16-bit lanes
short _xorReduce_i16x16(__m256i x) {
    const __m256i REDUCE_256 = _mm256_set1_epi16(-1);

    __m256i vecReduce = _mm256_bmacxor16x16x16(_mm256_setzero_si256(), REDUCE_256, x);
    return (short)_mm256_cvtsi256_si32(vecReduce);
}

// Horizontal XOR between all 16-bit lanes
short _xorReduce_i16x32(__m512i x) {
    const __m256i REDUCE_256 = _mm256_set1_epi16(-1);

    __m256i halfReduce = _mm256_xor_si256(_mm512_castsi512_si256(x), _mm512_extracti64x4_epi64(x, 1));
    __m256i vecReduce = _mm256_bmacxor16x16x16(_mm256_setzero_si256(), REDUCE_256, halfReduce);
    return (short)_mm256_cvtsi256_si32(vecReduce);
}

// Horizontal XOR between all 32-bit lanes
int _xorReduce_i32x8(__m256i x) {
    // Operation is 16-bit, so reduce the even/odd lanes separately
    const __m128i REDUCE_128 = _mm_setr_epi16(
        0b0101010101010101, 0b1010101010101010, 0,0,0,0,0,0
    );

    __m256i vecReduce = _mm256_bmacxor16x16x16(_mm256_setzero_si256(), _mm256_castsi128_si256(REDUCE_128), x);
    return _mm256_cvtsi256_si32(vecReduce);
}



// ==================
//      Shuffles
// ==================

// Duplicates the even lanes into the odd. Same as permute16(x, {0,0, 2,2, 4,4, ... 14,14})
__m256i _duplicateEven_i16x16(__m256i x) {
    const __m256i DUP_EVEN_VECTOR = _mm256_setr_epi16(
        1<<0, 1<<0, 1<<2, 1<<2, 1<<4, 1<<4, 1<<6, 1<<6, 1<<8, 1<<8, 1<<10, 1<<10, 1<<12, 1<<12, 1<<14, 1<<14
    );

    return _mm256_bmacor16x16x16(_mm256_setzero_si256(), DUP_EVEN_VECTOR, x);
}



// =========================
//      Bit duplication
// =========================

// Duplicate the low order bits so each is double its width
// E.g: `0bPONMLKJIHGFEDCBA => 0bHHGGFFEEDDCCBBAA`
__m256i _bitDouble_i16x16(__m256i x) {
    const __m256i BIT_EXPAND2X = _mm256_setr_epi16(
        0b11<<0, 0b11<<2, 0b11<<4, 0b11<<6, 0b11<<8, 0b11<<10, 0b11<<12, 0b11<<14, 0,0,0,0,0,0,0,0
    );

    return _mm256_bmacxor16x16x16(_mm256_setzero_si256(), x, BIT_EXPAND2X);
}

// Duplicate the high order bits so each is double its width
// E.g: `0bPONMLKJIHGFEDCBA => 0bPPOONNMMLLKKJJII`
__m256i _bitDoubleHi_i16x16(__m256i x) {
    const __m256i BIT_EXPAND2X = _mm256_setr_epi16(
        0,0,0,0,0,0,0,0, 0b11<<0, 0b11<<2, 0b11<<4, 0b11<<6, 0b11<<8, 0b11<<10, 0b11<<12, 0b11<<14
    );

    return _mm256_bmacxor16x16x16(_mm256_setzero_si256(), x, BIT_EXPAND2X);
}

// Duplicate the low order bits so each is quadruple its width
// E.g: `0bPONMLKJIHGFEDCBA => 0bDDDDCCCCBBBBAAAA`
__m256i _bitQuadruple_i16x16(__m256i x) {
    const __m256i BIT_EXPAND4X = _mm256_setr_epi16(
        0b1111<<0, 0b1111<<4, 0b1111<<8, 0b1111<<12, 0,0,0,0,0,0,0,0,0,0,0,0 
    );

    return _mm256_bmacxor16x16x16(_mm256_setzero_si256(), x, BIT_EXPAND4X);
}



// ======================
//      Hamming code
// ======================

// Calculates a Extended Hamming (16,11) code
// Input:   [0:10] - data, [11:15] - padding (unused)
// Output:  {P1, P2, D1, P4, D2, D3, D4, P8, D5, D6, D7, D8, D9, D10, D11, P0}
__m256i _encodeExtHamming16_11_i256(__m256i x) {
    // Each parity bit corresponds to an index bit of the data's output position (not input!)
    const __m256i HAMMING_16_11_MATRIX = _mm256_setr_epi16(
        0b10101011011, 0b11001101101, 1<<0, 0b11110001110, 1<<1, 1<<2, 1<<3, 0b11111110000, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 0b10010110111
    );

    return _mm256_bmacxor16x16x16(_mm256_setzero_si256(), x, HAMMING_16_11_MATRIX);
}
