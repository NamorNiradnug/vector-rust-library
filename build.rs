use cfg_aliases::cfg_aliases;

fn main() {
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
