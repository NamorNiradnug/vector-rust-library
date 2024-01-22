#include "vectorclass_bench.hpp"

#define VCL_NAMESPACE vcl
#include "vcl/vectorclass.h"

float benches::DotprodVec8fVCL(const int32_t n, const float *vec1, const float *vec2) {
    using vcl::Vec8f;

    Vec8f vec1_part;
    Vec8f vec2_part;
    Vec8f sum = 0;

    int i = 0;
    for (; i < (n & ~7); i += Vec8f::size()) {
        vec1_part.load(vec1 + i);
        vec2_part.load(vec2 + i);
        sum += vec1_part * vec2_part;
    }

    if (i < n) {
        vec1_part.load_partial(n - i, vec1 + i);
        vec2_part.load_partial(n - i, vec2 + i);
        sum += vec1_part * vec2_part;
    }

    return vcl::horizontal_add(sum);
}

float benches::DotprodVec8fVCLFused(int n, const float *vec1, const float *vec2) {
    using vcl::Vec8f;

    Vec8f vec1_part;
    Vec8f vec2_part;
    Vec8f sum = 0;

    int i = 0;
    for (; i < (n & ~7); i += Vec8f::size()) {
        vec1_part.load(vec1 + i);
        vec2_part.load(vec2 + i);
        sum = vcl::mul_add(vec1_part, vec2_part, sum);
    }

    if (i < n) {
        vec1_part.load_partial(n - i, vec1 + i);
        vec2_part.load_partial(n - i, vec2 + i);
        sum += vcl::mul_add(vec1_part, vec2_part, sum);
    }

    return vcl::horizontal_add(sum);
}
