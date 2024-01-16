use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, SeedableRng};

#[path = "../tests/dotprod.rs"]
mod dotprod;
use dotprod::*;

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
        group.bench_with_input(
            BenchmarkId::new("handwritten loop with fused add-mul", vec_len),
            &input,
            |b, (vec1, vec2)| b.iter(|| dotprod_vec8f_loop_fused(vec1, vec2)),
        );
    }
    group.finish();
}

criterion_group!(dotprod, dotprod_bench);
criterion_main!(dotprod);
