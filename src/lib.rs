//! Port (__NOT__ bindings) of Agner Fog's [Vector Class Library](https://github.com/vectorclass/version2) to Rust.
//! `vrl` stands for **V**ector **R**ust **L**ibrary.
//!
//! This is a library for using the SIMD (Single Instruction Multiple Data) instructions on modern
//! x86 and x86-64 CPUs.

mod common;
mod macros;
mod vec256f;

pub use vec256f::*;
