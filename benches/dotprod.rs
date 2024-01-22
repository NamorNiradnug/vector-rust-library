use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, SeedableRng};

#[path = "../tests/dotprod.rs"]
mod dotprod;
use dotprod::*;

#[cfg(all(
    feature = "vectorclass_bench",
    not(target_arch = "x86"),
    not(target_arch = "x86_64")
))]
compile_error!("Vector Class Library only available on x86 and x86-64 platforms");

#[cfg(feature = "vectorclass_bench")]
mod vcl_bench {
    #[cxx::bridge(namespace = "benches")]
    mod ffi {
        unsafe extern "C++" {
            include!("vector-rust-library/benches/vectorclass_bench.hpp");
            unsafe fn DotprodVec8fVCL(n: i32, vec1: *const f32, vec2: *const f32) -> f32;
            unsafe fn DotprodVec8fVCLFused(n: i32, vec1: *const f32, vec2: *const f32) -> f32;
        }
    }

    #[inline]
    pub fn dotprod_vec8f_vectorclass(vec1: &[f32], vec2: &[f32]) -> f32 {
        assert_eq!(vec1.len(), vec2.len());
        // SAFETY: hopefully there's no UB in the C++ code.
        unsafe { ffi::DotprodVec8fVCL(vec1.len() as i32, vec1.as_ptr(), vec2.as_ptr()) }
    }

    #[inline]
    pub fn dotprod_vec8f_vectorclass_fused(vec1: &[f32], vec2: &[f32]) -> f32 {
        assert_eq!(vec1.len(), vec2.len());
        // SAFETY: hopefully there's no UB in the C++ code.
        unsafe { ffi::DotprodVec8fVCLFused(vec1.len() as i32, vec1.as_ptr(), vec2.as_ptr()) }
    }
}

fn dotprod_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("dotprod");
    let mut rand_gen = SmallRng::seed_from_u64(57);
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

        #[cfg(feature = "vectorclass_bench")]
        {
            use vcl_bench::*;
            bench_dotprod!(dotprod_vec8f_vectorclass, "VCL");
            bench_dotprod!(dotprod_vec8f_vectorclass_fused, "VCL with fused mul-add");
        }
    }
    group.finish();
}

criterion_group!(dotprod, dotprod_bench);
criterion_main!(dotprod);
