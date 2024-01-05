use std::{iter::zip, ops::Range, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use vrl::{SIMDVector, Vec8f};

fn dotprod_simple(vec1: &[f32], vec2: &[f32]) -> f32 {
    zip(vec1, vec2).map(|(x, y)| x * y).sum()
}

fn dotprod_vec8f_chunks(vec1: &[f32], vec2: &[f32]) -> f32 {
    zip(vec1.chunks(Vec8f::ELEMENTS), vec2.chunks(Vec8f::ELEMENTS))
        .map(|(x, y)| Vec8f::load_partial(x) * Vec8f::load_partial(y))
        .sum::<Vec8f>()
        .sum()
}

fn dotprod_vec8f_loop(mut vec1: &[f32], mut vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = Vec8f::default();
    while vec1.len() >= Vec8f::ELEMENTS {
        let (head1, tail1) = vec1.split_at(Vec8f::ELEMENTS);
        let (head2, tail2) = vec2.split_at(Vec8f::ELEMENTS);
        sum += Vec8f::load_checked(head1) * Vec8f::load_checked(head2);
        vec1 = tail1;
        vec2 = tail2;
    }
    sum += Vec8f::load_partial(vec1) * Vec8f::load_partial(vec2);
    sum.sum()
}

fn dotprod_vec8f_ptr(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = Vec8f::default();
    let ptr1 = vec1.as_ptr() as *const [f32; 8];
    let ptr2 = vec2.as_ptr() as *const [f32; 8];
    let whole_iters = vec1.len() / Vec8f::ELEMENTS;
    for i in 0..whole_iters {
        sum += unsafe { Vec8f::load_ptr(ptr1.add(i)) * Vec8f::load_ptr(ptr2.add(i)) }
    }
    if whole_iters * Vec8f::ELEMENTS < vec1.len() {
        sum += Vec8f::load_partial(vec1.split_at(whole_iters * Vec8f::ELEMENTS).1)
            * Vec8f::load_partial(vec2.split_at(whole_iters * Vec8f::ELEMENTS).1)
    }
    sum.sum()
}

fn generate_rand_vector(len: usize, range: Range<f32>, rand_gen: &mut SmallRng) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    result.resize_with(len, || rand_gen.gen_range(range.clone()));
    result
}

fn dotprod_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("dotprod");
    let mut rand_gen = SmallRng::seed_from_u64(57);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    for vec_len in [
        1,
        3,
        4,
        6,
        8,
        16,
        256,
        256 + 3,
        256 + 7,
        1024,
        1024 + 3,
        1024 + 7,
    ] {
        let vec1 = generate_rand_vector(vec_len, -1.0..1.0, &mut rand_gen);
        let vec2 = generate_rand_vector(vec_len, -1.0..1.0, &mut rand_gen);
        let input = (vec1.as_slice(), vec2.as_slice());

        group.throughput(Throughput::Elements(vec_len as u64));
        group.bench_with_input(
            BenchmarkId::new("no SIMD", vec_len),
            &input,
            |b, (vec1, vec2)| b.iter(|| dotprod_simple(vec1, vec2)),
        );
        group.bench_with_input(
            BenchmarkId::new("using chunks iterator", vec_len),
            &input,
            |b, (vec1, vec2)| b.iter(|| dotprod_vec8f_chunks(vec1, vec2)),
        );
        group.bench_with_input(
            BenchmarkId::new("handwritten loop", vec_len),
            &input,
            |b, (vec1, vec2)| b.iter(|| dotprod_vec8f_loop(vec1, vec2)),
        );
        group.bench_with_input(
            BenchmarkId::new("handwritten loop with raw pointers", vec_len),
            &input,
            |b, (vec1, vec2)| b.iter(|| dotprod_vec8f_ptr(vec1, vec2)),
        );
    }
    group.finish();
}
criterion_group!(dotprod, dotprod_bench);
criterion_main!(dotprod);

// TODO: how to execute this test?
#[test]
fn test_dotprod() {
    const TEST_VEC_LEN: usize = 20;
    const TEST_SEED: usize = 57;
    let mut rand_gen = SmallRng::seed_from_u64(TEST_SEED);
    let vec1 = generate_rand_vector(TEST_VEC_LEN, -1.0..1.0, &mut rand_gen);
    let vec2 = generate_rand_vector(TEST_VEC_LEN, -1.0..1.0, &mut rand_gen);
    for i in 0..TEST_SEED {
        let vec1 = vec1.split_at(i).0;
        let vec2 = vec2.split_at(i).0;
        assert_eq!(dotprod_simple(vec1, vec2), dotprod_vec8f_chunks(vec1, vec2));
        assert_eq!(dotprod_simple(vec1, vec2), dotprod_vec8f_loop(vec1, vec2));
        assert_eq!(dotprod_simple(vec1, vec2), dotprod_vec8f_loop(vec1, vec2));
        assert_eq!(dotprod_simple(vec1, vec2), dotprod_vec8f_ptr(vec1, vec2));
    }
}
