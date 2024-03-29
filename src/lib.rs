//! Port (__NOT__ bindings) of Agner Fog's [Vector Class Library](https://github.com/vectorclass/version2) to Rust.
//! `vrl` stands for **V**ector **R**ust **L**ibrary.
//!
//! This is a library for using the SIMD (Single Instruction Multiple Data) instructions on modern
//! x86 and x86-64 CPUs.

mod common;
mod macros;

mod vec4f;
mod vec8f;

mod intrinsics {
    #[cfg(target_arch = "x86_64")]
    pub use core::arch::x86_64::*;

    #[cfg(target_arch = "x86")]
    pub use core::arch::x86::*;

    #[cfg(target_arch = "aarch64")]
    pub use core::arch::aarch64::*;
}

pub mod prelude {
    pub use super::{
        common::*,
        vec4f::{Vec4f, Vec4fBase},
        vec8f::{Vec8f, Vec8fBase},
    };
}
