use cfg_aliases::cfg_aliases;

fn main() {
    #[cfg(feature = "vectorclass_bench")]
    {
        println!("cargo:rerun-if-changed=benches/vectorclass_bench.cpp");
        cxx_build::bridge("benches/dotprod.rs")
            .file("benches/vectorclass_bench.cpp")
            .flag("-march=native")
            .flag("-O3")
            .compile("vectorclass_bench");
    }

    cfg_aliases! {
        sse: { target_feature = "sse" },
        no_sse: { not(sse) },
        sse41: { target_feature = "sse4.1" },
        no_sse41: { not(sse41) },
        avx: { target_feature = "avx" },
        no_avx: { not(avx) },
        neon: { target_feature = "neon" },
    }
}
