use std::env;

fn feature_shortcut(feature: &str, shortcut_name: &str) {
    if env::var("CARGO_CFG_TARGET_FEATURE")
        .map_or(false, |cfg| cfg.split(',').any(|f| f == feature))
    {
        println!("cargo:rustc-cfg={}", shortcut_name);
    } else {
        println!("cargo:rustc-cfg=no_{}", shortcut_name)
    }
}

fn feature_shortcut_same(feature: &str) {
    feature_shortcut(feature, feature);
}

fn main() {
    feature_shortcut_same("sse");
    feature_shortcut("sse4.1", "sse41");
    feature_shortcut_same("avx");
}
