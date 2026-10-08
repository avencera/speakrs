//! Operand conversion for offline FP16 emulation builds, not an FP16 MMA kernel

use cuda_device::ptx_asm;

// evaluate the environment on the host: device MIR must contain no string pointers
const ENABLED: bool = option_env!("SPEAKRS_FP16_EMU_BUILD").is_some();

/// Whether this offline kernel build is for the private operand experiment
#[inline(always)]
pub(crate) fn enabled() -> bool {
    ENABLED
}

/// Converts through FP16 with a fixed power-of-two scale, then restores FP32
///
/// The measured 36-file range stays below overflow at scale 1024
#[inline(always)]
pub(crate) fn round(value: f32) -> f32 {
    let scaled = value * 1024.0;
    let rounded: f32;
    // safety: register conversions with no memory access
    unsafe {
        ptx_asm!(
            "{ .reg .b16 h; cvt.rn.f16.f32 h, %1; cvt.f32.f16 %0, h; }",
            out("=f") rounded,
            in("f") scaled,
            options(register_only),
        );
    }
    rounded * (1.0 / 1024.0)
}
