/*
    Software emulation of the BMM bit matrix multiply instructions.
*/

#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

// bmacor16x16x16:  res.word[i].bit[j] = a.word[i].bit[j] | (b.word[i] & c.column[i] != 0)
// bmacxor16x16x16: res.word[i].bit[j] = a.word[i].bit[j] ^ parity(b.word[i] & c.column[i])




// Calculate the reduction of one element per iteration
__m256i bmmOr_horizontal(__m256i a, __m256i b, __m256i matrix) {
    const __m256i INCREASING_BITS = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    uint16_t b_arr[16], result[16];
    _mm256_storeu_si256((__m256i*)b_arr, b);
 
    for (size_t i = 0; i < 16; i++) {
        // Spread one bit to each element to make up for (AVX512: vpmovm2w)
        __m256i dup = _mm256_set1_epi16(b_arr[i]);
        __m256i bitBroadcast = _mm256_cmpeq_epi16(_mm256_andnot_si256(dup, INCREASING_BITS), _mm256_setzero_si256());

        __m256i toReduce = _mm256_and_si256(bitBroadcast, matrix);
        __m128i reduction0 = _mm_or_si128(_mm256_castsi256_si128(toReduce), _mm256_extracti128_si256(toReduce, 1));
        __m128i reduction1 = _mm_or_si128(reduction0, _mm_bsrli_si128(reduction0, 8));
        __m128i reduction2 = _mm_or_si128(reduction1, _mm_srli_epi64(reduction1, 32));
        __m128i reduction3 = _mm_or_si128(reduction2, _mm_srli_epi32(reduction2, 16));

        result[i] = (uint16_t)_mm_cvtsi128_si32(reduction3);
    }

    return _mm256_or_si256(a, _mm256_loadu_si256((__m256i*)result));
}

__m256i bmmXor_horizontal(__m256i a, __m256i b, __m256i matrix) {
    const __m256i INCREASING_BITS = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    uint16_t b_arr[16], result[16];
    _mm256_storeu_si256((__m256i*)b_arr, b);
 
    for (size_t i = 0; i < 16; i++) {
        // Spread one bit to each element to make up for (AVX512: vpmovm2w)
        __m256i dup = _mm256_set1_epi16(b_arr[i]);
        __m256i bitBroadcast = _mm256_cmpeq_epi16(_mm256_andnot_si256(dup, INCREASING_BITS), _mm256_setzero_si256());

        __m256i toReduce = _mm256_and_si256(bitBroadcast, matrix);
        __m128i reduction0 = _mm_xor_si128(_mm256_castsi256_si128(toReduce), _mm256_extracti128_si256(toReduce, 1));
        __m128i reduction1 = _mm_xor_si128(reduction0, _mm_bsrli_si128(reduction0, 8));
        __m128i reduction2 = _mm_xor_si128(reduction1, _mm_srli_epi64(reduction1, 32));
        __m128i reduction3 = _mm_xor_si128(reduction2, _mm_srli_epi32(reduction2, 16));

        result[i] = (uint16_t)_mm_cvtsi128_si32(reduction3);
    }

    return _mm256_xor_si256(a, _mm256_loadu_si256((__m256i*)result));
}




// Calculate 1-16th of the reduction each iteration for all elements
__m256i bmmXOr_verticalRot(__m256i a, __m256i b, __m256i matrix) {
    // permute = (broadcast(b.bit[0]) & c[0]) ^ (broadcast(b.bit[1]) & c[1])... ^ (broadcast(b.bit[7]) & c[7])
    __m256i bitMask = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 
        1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    __m256i reduction = _mm256_setzero_si256(); 
    for (size_t i = 0; i < 16; i++) {
        __m256i selectedBit = _mm256_andnot_si256(b, bitMask);
        __m256i broadcastedBit = _mm256_cmpeq_epi16(selectedBit, _mm256_setzero_si256());

        reduction = _mm256_xor_si256(reduction, _mm256_and_si256(broadcastedBit, matrix));

        bitMask = _mm256_or_si256(_mm256_slli_epi16(bitMask, 1), _mm256_srli_epi16(bitMask, 15));
        matrix = _mm256_alignr_epi8(matrix, _mm256_permute2x128_si256(matrix,matrix, _MM_SHUFFLE2(0,1)), 2);
    }

    return _mm256_xor_si256(a, reduction);
}

__m256i bmmOr_verticalRot(__m256i a, __m256i b, __m256i matrix) {
    // permute = (broadcast(b.bit[0]) & c[0]) ^ (broadcast(b.bit[1]) & c[1])... ^ (broadcast(b.bit[7]) & c[7])
    __m256i bitMask = _mm256_setr_epi16(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 
        1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15
    );

    __m256i reduction = _mm256_setzero_si256(); 
    for (size_t i = 0; i < 16; i++) {
        __m256i selectedBit = _mm256_andnot_si256(b, bitMask);
        __m256i broadcastedBit = _mm256_cmpeq_epi16(selectedBit, _mm256_setzero_si256());

        reduction = _mm256_or_si256(reduction, _mm256_and_si256(broadcastedBit, matrix));

        bitMask = _mm256_or_si256(_mm256_slli_epi16(bitMask, 1), _mm256_srli_epi16(bitMask, 15));
        matrix = _mm256_alignr_epi8(matrix, _mm256_permute2x128_si256(matrix,matrix, _MM_SHUFFLE2(0,1)), 2);
    }

    return _mm256_or_si256(a, reduction);
}




// Split the operation based on which 16-bit half it operates on
__m256i bmmXor_GFNI(__m256i a, __m256i b, __m256i matrix) {
    // Reverse order as GFNI transpose trick works in reverse order
    const __m256i LO_SRC_ROWS = _mm256_setr_epi8(
        14,12,10,8,6,4,2,0,     15,13,11,9,7,5,3,1,
        14,12,10,8,6,4,2,0,     15,13,11,9,7,5,3,1
    );

    const __m256i HI_SRC_ROWS = _mm256_setr_epi8(
        30,28,26,24,22,20,18,16,    31,29,27,25,23,21,19,17,
        30,28,26,24,22,20,18,16,    31,29,27,25,23,21,19,17
    );

    // Split the matrix based on which 16-bit half it takes it sources from
    // [SrcLoForLo, SrcLoForHi]     [SrcLoForLo, SrcLoForHi]
    // [SrcHiForLo, SrcHiForHi] =>  [SrcLoForLo, SrcLoForHi]
    __m256i matThatSelsLo = _mm256_permutexvar_epi8(LO_SRC_ROWS, matrix);
    __m256i matThatSelsHi = _mm256_permutexvar_epi8(HI_SRC_ROWS, matrix);

    // Transpose the matrix to GFNI's orientation (row-major, little-endian)
    const __m256i COLUMN_TO_ROW = _mm256_set1_epi64x(0x0102040810204080ull);
    __m256i gfniMatrix_lo = _mm256_gf2p8affine_epi64_epi8(COLUMN_TO_ROW, matThatSelsLo, 0);
    __m256i gfniMatrix_hi = _mm256_gf2p8affine_epi64_epi8(COLUMN_TO_ROW, matThatSelsHi, 0);

    // Truncate then broadcast within each 128-bit lane
    __m256i loDupe = _mm256_shuffle_epi8(b, _mm256_set1_epi64x(0x0e0c0a0806040200));
    __m256i hiDupe = _mm256_shuffle_epi8(b, _mm256_set1_epi64x(0x0f0d0b0907050301));

    __m256i reductionFromLo = _mm256_gf2p8affine_epi64_epi8(loDupe, gfniMatrix_lo, 0);
    __m256i reductionFromHi = _mm256_gf2p8affine_epi64_epi8(hiDupe, gfniMatrix_hi, 0);
    __m256i reductionSplit = _mm256_xor_si256(reductionFromLo, reductionFromHi);

    const __m256i INTERLEAVE_WITH_UPPER = _mm256_setr_epi8(
        0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15,
        0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
    );

    __m256i reduction = _mm256_shuffle_epi8(reductionSplit, INTERLEAVE_WITH_UPPER);
    
    return _mm256_xor_si256(a, reduction);
}




// Direct port of AMD's pseudo code
__m256i bmmOr_documentation(__m256i av, __m256i bv, __m256i cv) {
    uint16_t a[16], b[16], c[16], result[16] = {0};
    _mm256_storeu_si256((__m256i*)a, av);
    _mm256_storeu_si256((__m256i*)b, bv);
    _mm256_storeu_si256((__m256i*)c, cv);

    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            bool reduction_bit = (a[i] >> j) & 1;
            for (size_t k = 0; k < 16; k++) {
                reduction_bit |= ((b[i] >> k) & (c[k] >> j)) & 1;
            }
            result[i] |= (uint16_t)reduction_bit << j;
        }
    }

    return _mm256_loadu_si256((__m256i*)result);
}

__m256i bmmXor_documentation(__m256i av, __m256i bv, __m256i cv) {
    uint16_t a[16], b[16], c[16], result[16] = {0};
    _mm256_storeu_si256((__m256i*)a, av);
    _mm256_storeu_si256((__m256i*)b, bv);
    _mm256_storeu_si256((__m256i*)c, cv);

    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            bool reduction_bit = (a[i] >> j) & 1;
            for (size_t k = 0; k < 16; k++) {
                reduction_bit ^= ((b[i] >> k) & (c[k] >> j)) & 1;
            }
            result[i] |= (uint16_t)reduction_bit << j;
        }
    }

    return _mm256_loadu_si256((__m256i*)result);
}
