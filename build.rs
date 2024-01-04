use std::env;

fn feature_shortcut(feature: &str) {
    if env::var("CARGO_CFG_TARGET_FEATURE")
        .map_or(false, |cfg| cfg.split(',').any(|f| f == feature))
    {
        println!("cargo:rustc-cfg={}", feature);
    } else {
        println!("cargo:rustc-cfg=no_{}", feature)
    }
}

fn main() {
    feature_shortcut("sse");
    feature_shortcut("avx");
}
