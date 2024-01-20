use std::{iter::zip, ops::Range};

use rand::{rngs::SmallRng, Rng};
use vrl::prelude::*;

pub fn dotprod_simple(vec1: &[f32], vec2: &[f32]) -> f32 {
    zip(vec1, vec2).map(|(x, y)| x * y).sum()
}

pub fn dotprod_vec8f_chunks(vec1: &[f32], vec2: &[f32]) -> f32 {
    zip(vec1.chunks(Vec8f::N), vec2.chunks(Vec8f::N))
        .map(|(x, y)| Vec8f::load_partial(x) * Vec8f::load_partial(y))
        .sum::<Vec8f>()
        .sum()
}

pub fn dotprod_vec8f_loop(mut vec1: &[f32], mut vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = Vec8f::default();
    while vec1.len() >= Vec8f::N {
        let (head1, tail1) = vec1.split_at(Vec8f::N);
        let (head2, tail2) = vec2.split_at(Vec8f::N);
        sum += Vec8f::load_checked(head1) * Vec8f::load_checked(head2);
        vec1 = tail1;
        vec2 = tail2;
    }
    sum += Vec8f::load_partial(vec1) * Vec8f::load_partial(vec2);
    sum.sum()
}

pub fn dotprod_vec8f_ptr(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = Vec8f::default();
    let whole_iters = vec1.len() / Vec8f::N;
    for i in 0..whole_iters {
        // SAFETY: 8 * whole_iters <= vec1.len() hence (8 * i)..(8 * i + 7) < vec1.len()
        sum += unsafe {
            Vec8f::load_ptr(vec1.as_ptr().add(8 * i)) * Vec8f::load_ptr(vec2.as_ptr().add(8 * i))
        }
    }
    if whole_iters * Vec8f::N < vec1.len() {
        sum += Vec8f::load_partial(vec1.split_at(whole_iters * Vec8f::N).1)
            * Vec8f::load_partial(vec2.split_at(whole_iters * Vec8f::N).1)
    }
    sum.sum()
}

pub fn generate_rand_vector(len: usize, range: Range<f32>, rand_gen: &mut SmallRng) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    result.resize_with(len, || rand_gen.gen_range(range.clone()));
    result
}

#[test]
fn test_dotprod() {
    use approx::assert_ulps_eq;
    use rand::SeedableRng;

    const TEST_VEC_LEN: usize = 20;
    const TEST_SEED: u64 = 57;
    let mut rand_gen = SmallRng::seed_from_u64(TEST_SEED);
    let vec1 = generate_rand_vector(TEST_VEC_LEN, -1.0..1.0, &mut rand_gen);
    let vec2 = generate_rand_vector(TEST_VEC_LEN, -1.0..1.0, &mut rand_gen);
    for i in 0..TEST_VEC_LEN {
        let vec1 = vec1.split_at(i).0;
        let vec2 = vec2.split_at(i).0;
        let expected = dotprod_simple(vec1, vec2);
        assert_ulps_eq!(dotprod_vec8f_chunks(vec1, vec2), expected);
        assert_ulps_eq!(dotprod_vec8f_loop(vec1, vec2), expected);
        assert_ulps_eq!(dotprod_vec8f_ptr(vec1, vec2), expected);
    }
}
