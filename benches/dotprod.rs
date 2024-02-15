#![cfg_attr(feature = "portable_simd_bench", feature(portable_simd))]

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use rand::{rngs::SmallRng, SeedableRng};

#[path = "../tests/dotprod.rs"]
mod dotprod;
use dotprod::*;

#[cfg(feature = "portable_simd_bench")]
fn dotprod_portable_simd(vec1: &[f32], vec2: &[f32]) -> f32 {
    use std::simd::prelude::*;
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = f32x8::default();

    let mut i = 0;
    while i < vec1.len() & !7 {
        sum += f32x8::from_slice(vec1.split_at(i).1) * f32x8::from_slice(vec2.split_at(i).1);
        i += 8;
    }
    if i < vec1.len() {
        // TODO: probably that's not the best way to implement `partial_load`
        sum += f32x8::gather_or_default(
            vec1,
            [i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7].into(),
        );
    }
    sum.reduce_sum()
}

fn dotprod_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("dotprod");
    let mut rand_gen = SmallRng::seed_from_u64(57);
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    for vec_len in [
        16,
        256,
        256 + 3,
        256 + 7,
        512,
        512 + 3,
        512 + 7,
        1024,
        1024 + 3,
        1024 + 7,
    ] {
        let vec1 = generate_rand_vector(vec_len, -1.0..1.0, &mut rand_gen);
        let vec2 = generate_rand_vector(vec_len, -1.0..1.0, &mut rand_gen);
        let input = (vec1.as_slice(), vec2.as_slice());

        group.throughput(Throughput::Elements(vec_len as u64));

        macro_rules! bench_dotprod {
            ($dotprod_fn: tt, $name: literal) => {
                group.bench_with_input(
                    BenchmarkId::new($name, vec_len),
                    &input,
                    |b, (vec1, vec2)| b.iter(|| $dotprod_fn(vec1, vec2)),
                );
            };
        }

        bench_dotprod!(dotprod_simple, "no SIMD");
        bench_dotprod!(dotprod_vec8f_chunks, "using chunks iterator");
        bench_dotprod!(dotprod_vec8f_loop, "handwritten loop");
        bench_dotprod!(dotprod_vec8f_ptr, "handwritten loop with raw pointers");
        bench_dotprod!(
            dotprod_vec8f_loop_fused,
            "handwritten loop with fused add-mul"
        );

        #[cfg(feature = "portable_simd_bench")]
        bench_dotprod!(dotprod_portable_simd, "portable simd f32x8");
    }
    group.finish();
}

criterion_group!(dotprod, dotprod_bench);
criterion_main!(dotprod);
