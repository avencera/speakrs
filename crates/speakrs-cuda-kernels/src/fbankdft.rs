//! Record-owned filterbank DFT producer
//!
//! One kernel replaces the always-on `fbank` area's frame/window kernel, its cuBLAS
//! DFT and its sparse mel kernel. It writes only the mel energies `[frames, 80]`;
//! the always-on `fbank_log_cmn` consumer is unchanged. Keeping this kernel in its own
//! area leaves the always-on module's PTX and its pinned driver JIT byte-identical
//!
//! Each block stages eight overlapping frames, then each warp runs one 512-point real
//! DFT as a packed 256-point complex FFT in two-term FP32 expansions, so large
//! intermediate terms that cancel keep their residuals. Inputs, tables, power and
//! energies stay FP32; no FP64 or TF32 instruction, device trigonometry or float
//! atomic is used. The host supplies the window, 512 twiddle pairs rounded once from
//! f64, and the 80 mel filters as runs of at most 16 bins inside bins 1..=255

use cuda_device::{DisjointSlice, SharedArray, kernel, ptx_asm, thread, warp};

// residual recovery needs rounded instructions that neither LLVM nor ptxas may contract
macro_rules! rounded_binary {
    ($name:ident, $instruction:literal) => {
        #[inline(always)]
        fn $name(a: f32, b: f32) -> f32 {
            let result: f32;
            // SAFETY: a register-only instruction with no memory access
            unsafe {
                ptx_asm!($instruction, out("=f") result, in("f") a, in("f") b, options(register_only));
            }
            result
        }
    };
}
rounded_binary!(pair_add_rn, "add.rn.f32 %0, %1, %2;");
rounded_binary!(pair_sub_rn, "sub.rn.f32 %0, %1, %2;");
rounded_binary!(pair_mul_rn, "mul.rn.f32 %0, %1, %2;");

// two-term expansions protect cancellation without using the GPU's slow FP64 pipeline
#[derive(Clone, Copy)]
struct FloatPair {
    hi: f32,
    lo: f32,
}

impl FloatPair {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    #[inline(always)]
    fn rounded(self) -> f32 {
        self.hi + self.lo
    }
}

impl From<f32> for FloatPair {
    #[inline(always)]
    fn from(hi: f32) -> Self {
        Self { hi, lo: 0.0 }
    }
}

impl core::ops::Add for FloatPair {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let sum = pair_add_rn(self.hi, other.hi);
        let virtual_other = pair_sub_rn(sum, self.hi);
        let error = pair_add_rn(
            pair_sub_rn(self.hi, pair_sub_rn(sum, virtual_other)),
            pair_sub_rn(other.hi, virtual_other),
        );
        let tail = pair_add_rn(error, pair_add_rn(self.lo, other.lo));
        let hi = pair_add_rn(sum, tail);
        Self {
            hi,
            lo: pair_add_rn(pair_sub_rn(sum, hi), tail),
        }
    }
}

impl core::ops::Neg for FloatPair {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl core::ops::Sub for FloatPair {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl core::ops::Mul<f32> for FloatPair {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: f32) -> Self {
        let product = pair_mul_rn(self.hi, other);
        let tail = self.hi.mul_add(other, -product) + self.lo * other;
        let hi = pair_add_rn(product, tail);
        Self {
            hi,
            lo: pair_add_rn(pair_sub_rn(product, hi), tail),
        }
    }
}

impl core::ops::Mul<FloatPair> for f32 {
    type Output = FloatPair;
    #[inline(always)]
    fn mul(self, other: FloatPair) -> FloatPair {
        other * self
    }
}

#[inline(always)]
fn pair_shuffle_xor(value: FloatPair, lane: u32) -> FloatPair {
    FloatPair {
        hi: warp::shuffle_xor_f32(value.hi, lane),
        lo: warp::shuffle_xor_f32(value.lo, lane),
    }
}

#[inline(always)]
fn pair_shuffle(value: FloatPair, lane: u32) -> FloatPair {
    FloatPair {
        hi: warp::shuffle_f32(value.hi, lane),
        lo: warp::shuffle_f32(value.lo, lane),
    }
}

/// Uses compensated FP32 butterflies in the packed real FFT
///
/// Launch 256 threads and a two-dimensional grid `(ceil(frames_per_row/8), rows)`
/// The window and twiddles are rounded once from host f64; no device trigonometry is used
///
/// `mel_width` must be 16, the stride of the staged transposed weights, and each
/// filter's run must lie inside bins 1..=255, the bins the real post-pass writes
#[kernel]
pub fn fbankdft_fft_mel_accurate(
    samples_per_row: u32,
    frames_per_row: u32,
    mel_width: u32,
    waveform: &[f32],
    window: &[f32],
    twiddle: &[f32],
    mel_first: &[u32],
    mel_count: &[u32],
    mel_weights: &[f32],
    mut energies: DisjointSlice<f32>,
) {
    static mut WAVE: SharedArray<f32, 1536> = SharedArray::UNINIT;
    static mut WINDOW: SharedArray<f32, 400> = SharedArray::UNINIT;
    static mut COS: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut SIN: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut FIRST: SharedArray<u32, 80> = SharedArray::UNINIT;
    static mut COUNT: SharedArray<u32, 80> = SharedArray::UNINIT;
    static mut WEIGHT: SharedArray<f32, 1280> = SharedArray::UNINIT;
    static mut POWER: SharedArray<f32, 2056> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let group = tid / 32;
    let first_frame = thread::blockIdx_x() * 8;
    let row = thread::blockIdx_y();
    let local_frame = first_frame + group;
    let frame = row * frames_per_row + local_frame;
    // SAFETY: the host checks the fixed table shapes and one scalar type owns each array
    let (wave, win, cos, sin, first, count, weight, power) = unsafe {
        (
            SharedArray::as_raw_mut_ptr(&raw mut WAVE),
            SharedArray::as_raw_mut_ptr(&raw mut WINDOW),
            SharedArray::as_raw_mut_ptr(&raw mut COS),
            SharedArray::as_raw_mut_ptr(&raw mut SIN),
            SharedArray::as_raw_mut_ptr(&raw mut FIRST),
            SharedArray::as_raw_mut_ptr(&raw mut COUNT),
            SharedArray::as_raw_mut_ptr(&raw mut WEIGHT),
            SharedArray::as_raw_mut_ptr(&raw mut POWER).add((group * 257) as usize),
        )
    };
    let mut i = tid;
    while i < 1536 {
        let sample = first_frame * 160 + i;
        let value = if sample < samples_per_row {
            waveform[(row * samples_per_row + sample) as usize]
        } else {
            0.0
        };
        unsafe { wave.add(i as usize).write(value) };
        i += 256;
    }

    i = tid;
    while i < 400 {
        unsafe { win.add(i as usize).write(window[i as usize]) };
        i += 256;
    }

    i = tid;
    while i < 512 {
        let slot = (i ^ (i >> 5)) as usize;
        unsafe {
            cos.add(slot).write(twiddle[(i * 2) as usize]);
            sin.add(slot).write(twiddle[(i * 2 + 1) as usize]);
        }
        i += 256;
    }

    i = tid;
    while i < 1280 {
        let mel = i % 80;
        let offset = i / 80;
        unsafe {
            weight
                .add(i as usize)
                .write(mel_weights[(mel * mel_width + offset) as usize])
        };
        i += 256;
    }

    if tid < 80 {
        unsafe {
            first.add(tid as usize).write(mel_first[tid as usize]);
            count.add(tid as usize).write(mel_count[tid as usize]);
        }
    }

    thread::sync_threads();
    // inactive tail warps leave only after the block-wide table initialization
    if local_frame >= frames_per_row {
        return;
    }

    let base = group * 160;
    let mut sum = 0.0_f32;
    i = lane;
    while i < 400 {
        sum += unsafe { wave.add((base + i) as usize).read() } * 32768.0;
        i += 32;
    }

    let mean = warp::reduce_sum_f32(sum) / 400.0;
    let mut real = [FloatPair::ZERO; 8];
    let mut imaginary = [FloatPair::ZERO; 8];
    let mut r = 0;
    #[unroll]
    while r < 8 {
        let sample = (lane + 32 * r as u32) * 2;
        if sample < 400 {
            unsafe {
                let previous =
                    wave.add((base + sample.saturating_sub(1)) as usize).read() * 32768.0 - mean;
                let current = wave.add((base + sample) as usize).read() * 32768.0 - mean;
                let next = wave.add((base + sample + 1) as usize).read() * 32768.0 - mean;
                real[r] =
                    FloatPair::from((current - 0.97 * previous) * win.add(sample as usize).read());
                imaginary[r] = FloatPair::from(
                    (next - 0.97 * current) * win.add((sample + 1) as usize).read(),
                );
            }
        }
        r += 1;
    }

    macro_rules! rotate {
        ($re:expr, $im:expr, $angle:expr) => {{
            let angle = $angle;
            let slot = (angle ^ (angle >> 5)) as usize;
            let c = unsafe { cos.add(slot).read() };
            let s = unsafe { sin.add(slot).read() };
            (c * $re - s * $im, s * $re + c * $im)
        }};
    }
    // radix-four DIF retains binary bit reversal, which makes the post-pass register-local
    macro_rules! local_four {
        ($a:literal, $b:literal, $c:literal, $d:literal, $j:expr) => {{
            let ar = real[$a];
            let ai = imaginary[$a];
            let br = real[$b];
            let bi = imaginary[$b];
            let cr = real[$c];
            let ci = imaginary[$c];
            let dr = real[$d];
            let di = imaginary[$d];
            let u0r = ar + cr;
            let u0i = ai + ci;
            let u1r = br + dr;
            let u1i = bi + di;
            let u2r = ar - cr;
            let u2i = ai - ci;
            let u3r = br - dr;
            let u3i = bi - di;
            real[$a] = u0r + u1r;
            imaginary[$a] = u0i + u1i;
            (real[$b], imaginary[$b]) = rotate!(u0r - u1r, u0i - u1i, $j * 4);
            (real[$c], imaginary[$c]) = rotate!(u2r - u3i, u2i + u3r, $j * 2);
            (real[$d], imaginary[$d]) = rotate!(u2r + u3i, u2i - u3r, $j * 6);
        }};
    }
    local_four!(0, 2, 4, 6, lane);
    local_four!(1, 3, 5, 7, lane + 32);
    macro_rules! local_two {
        ($a:literal, $b:literal) => {{
            let ar = real[$a];
            let ai = imaginary[$a];
            let br = real[$b];
            let bi = imaginary[$b];
            real[$a] = ar + br;
            imaginary[$a] = ai + bi;
            (real[$b], imaginary[$b]) = rotate!(ar - br, ai - bi, lane * 8);
        }};
    }
    local_two!(0, 1);
    local_two!(2, 3);
    local_two!(4, 5);
    local_two!(6, 7);
    macro_rules! warp_four {
        ($r:literal, $half:literal) => {{
            let ar = real[$r];
            let ai = imaginary[$r];
            let br = pair_shuffle_xor(ar, $half);
            let bi = pair_shuffle_xor(ai, $half);
            let (mut ur, mut ui) = if lane & $half == 0 {
                (ar + br, ai + bi)
            } else {
                (br - ar, bi - ai)
            };
            if lane & $half != 0 && lane & ($half / 2) != 0 {
                (ur, ui) = (-ui, ur);
            }
            let vr = pair_shuffle_xor(ur, $half / 2);
            let vi = pair_shuffle_xor(ui, $half / 2);
            let (re, im) = if lane & ($half / 2) == 0 {
                (ur + vr, ui + vi)
            } else {
                (vr - ur, vi - ui)
            };
            let q = (lane / ($half / 2)) % 4;
            let factor = if q == 1 {
                2
            } else if q == 2 {
                1
            } else {
                3
            };
            let angle = (lane % ($half / 2)) * (512 / ($half * 2)) * factor;
            (real[$r], imaginary[$r]) = if q == 0 {
                (re, im)
            } else {
                rotate!(re, im, angle)
            };
        }};
    }
    warp_four!(0, 16);
    warp_four!(1, 16);
    warp_four!(2, 16);
    warp_four!(3, 16);
    warp_four!(4, 16);
    warp_four!(5, 16);
    warp_four!(6, 16);
    warp_four!(7, 16);
    warp_four!(0, 4);
    warp_four!(1, 4);
    warp_four!(2, 4);
    warp_four!(3, 4);
    warp_four!(4, 4);
    warp_four!(5, 4);
    warp_four!(6, 4);
    warp_four!(7, 4);
    macro_rules! final_two {
        ($r:literal) => {{
            let ar = real[$r];
            let ai = imaginary[$r];
            let br = pair_shuffle_xor(ar, 1);
            let bi = pair_shuffle_xor(ai, 1);
            (real[$r], imaginary[$r]) = if lane & 1 == 0 {
                (ar + br, ai + bi)
            } else {
                (br - ar, bi - ai)
            };
        }};
    }
    final_two!(0);
    final_two!(1);
    final_two!(2);
    final_two!(3);
    final_two!(4);
    final_two!(5);
    final_two!(6);
    final_two!(7);
    macro_rules! real_power {
        ($r:literal, $partner:literal) => {{
            let bin = (lane + 32 * $r).reverse_bits() >> 24;
            let other = (256 - bin) & 255;
            let other_lane = (other >> 3).reverse_bits() >> 27;
            let ar = real[$r];
            let ai = imaginary[$r];
            let br = pair_shuffle(real[$partner], other_lane);
            let bi = pair_shuffle(imaginary[$partner], other_lane);
            let slot = (bin ^ (bin >> 5)) as usize;
            let c = unsafe { cos.add(slot).read() };
            let s = unsafe { sin.add(slot).read() };
            let re = 0.5 * (ar + br + s * (ar - br) + c * (ai + bi));
            let im = 0.5 * (ai - bi + s * (ai + bi) - c * (ar - br));
            let re = re.rounded();
            let im = im.rounded();
            unsafe { power.add(slot).write(re * re + im * im) };
        }};
    }
    real_power!(0, 0);
    real_power!(1, 1);
    real_power!(2, 3);
    real_power!(3, 2);
    real_power!(4, 7);
    real_power!(5, 6);
    real_power!(6, 5);
    real_power!(7, 4);
    warp::sync_mask(u32::MAX);
    let mut mel = lane;
    while mel < 80 {
        let first_bin = unsafe { first.add(mel as usize).read() };
        let count = unsafe { count.add(mel as usize).read() };
        let mut sum = 0.0_f32;
        let mut offset = 0;
        while offset < count {
            let bin = first_bin + offset;
            let slot = (bin ^ (bin >> 5)) as usize;
            unsafe {
                sum += power.add(slot).read() * weight.add((offset * 80 + mel) as usize).read()
            };
            offset += 1;
        }
        unsafe { *energies.get_unchecked_mut((frame * 80 + mel) as usize) = sum };
        mel += 32;
    }
}
