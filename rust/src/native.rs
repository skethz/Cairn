use anyhow::{Result, bail};
use std::ffi::{CStr, c_char};
use tch::Tensor;
use torch_sys::C_tensor;
unsafe extern "C" {
    fn cairn_scan(read: *const C_tensor, local: *const C_tensor, transition: *const C_tensor,
        write: *const C_tensor, initial: *const C_tensor, output: *mut *mut C_tensor, final_state: *mut *mut C_tensor) -> i32;
    fn cairn_memory(reset: i32, allocated: *mut f64, reserved: *mut f64) -> i32;
    fn cairn_routes(selected: *const C_tensor, experts: i64, routes: *mut *mut C_tensor, counts: *mut *mut C_tensor) -> i32;
    fn cairn_autocast_enter(enabled: i32) -> i32;
    fn cairn_autocast_exit(previous: i32);
    fn cairn_error() -> *const c_char;
    fn cairn_settings() -> i32;
    fn cairn_deterministic(enabled: i32) -> i32;
    fn cairn_normal(t: *const C_tensor, seed: u64, std: f64) -> i32;
    fn cairn_clear_grad(t: *const C_tensor) -> i32;
    fn cairn_backward(out: *const C_tensor, grad: *const C_tensor) -> i32;
    fn cairn_clip(p: *const *const C_tensor, n: usize, max: f64, result: *mut f64) -> i32;
    fn cairn_adamw(
        p: *const *const C_tensor,
        m: *const *const C_tensor,
        v: *const *const C_tensor,
        s: *const *const C_tensor,
        n: usize,
        lr: f64,
        wd: f64,
    ) -> i32;
    fn cairn_rng_get(cuda: i32, out: *mut *mut C_tensor) -> i32;
    fn cairn_rng_set(cuda: i32, state: *const C_tensor) -> i32;
}
pub fn scan(read: &Tensor, local: &Tensor, transition: &Tensor, write: &Tensor, initial: &Tensor) -> Result<(Tensor,Tensor)> {
    let (mut out,mut state)=(std::ptr::null_mut(),std::ptr::null_mut());
    check(unsafe { cairn_scan(read.as_ptr(),local.as_ptr(),transition.as_ptr(),write.as_ptr(),initial.as_ptr(),&mut out,&mut state) })?;
    Ok((unsafe {Tensor::from_ptr(out)},unsafe {Tensor::from_ptr(state)}))
}
pub fn memory(reset: bool) -> Result<(f64,f64)> {
    let (mut a,mut r)=(0.,0.);
    check(unsafe { cairn_memory(reset as i32,&mut a,&mut r) })?;
    Ok((a/(1024_f64.powi(3)),r/(1024_f64.powi(3))))
}
static OPTIMIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static RETAINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static DETERMINISTIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn deterministic_enabled() -> bool { DETERMINISTIC.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_scan(enabled: bool) { SCAN.store(enabled,std::sync::atomic::Ordering::Relaxed); }
pub fn scan_enabled() -> bool { SCAN.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_retained(n: usize) { RETAINED.store(n, std::sync::atomic::Ordering::Relaxed); }
pub fn retained() -> usize { RETAINED.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_optimized(enabled: bool) { OPTIMIZED.store(enabled, std::sync::atomic::Ordering::Relaxed); }
pub fn optimized() -> bool { OPTIMIZED.load(std::sync::atomic::Ordering::Relaxed) }
pub fn routes(selected: &Tensor, experts: i64) -> Result<(Tensor, Vec<i64>)> {
    let mut routes=std::ptr::null_mut();
    let mut counts=std::ptr::null_mut();
    check(unsafe { cairn_routes(selected.as_ptr(), experts, &mut routes, &mut counts) })?;
    let routes=unsafe { Tensor::from_ptr(routes) };
    let counts=unsafe { Tensor::from_ptr(counts) };
    Ok((routes, Vec::try_from(&counts)?))
}
pub fn autocast<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Guard(i32);
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                cairn_autocast_exit(self.0);
            }
        }
    }
    let _guard = Guard(unsafe { cairn_autocast_enter(enabled as i32) });
    f()
}
fn check(code: i32) -> Result<()> {
    if code != 0 {
        bail!(
            "Native backend: {}",
            unsafe { CStr::from_ptr(cairn_error()) }.to_string_lossy()
        );
    }
    Ok(())
}
pub fn settings() -> Result<()> {
    check(unsafe { cairn_settings() })
}
pub fn deterministic(enabled: bool) -> Result<()> {
    check(unsafe { cairn_deterministic(enabled as i32) })?;
    DETERMINISTIC.store(enabled,std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub fn normal(t: &Tensor, seed: u64, std: f64) -> Result<()> {
    check(unsafe { cairn_normal(t.as_ptr(), seed, std) })
}
pub fn clear_grad(t: &Tensor) -> Result<()> {
    check(unsafe { cairn_clear_grad(t.as_ptr()) })
}
pub fn backward(out: &Tensor, grad: &Tensor) -> Result<()> {
    check(unsafe { cairn_backward(out.as_ptr(), grad.as_ptr()) })
}
pub fn clip(params: &[&Tensor], max: f64) -> Result<f64> {
    let p: Vec<_> = params.iter().map(|t| t.as_ptr()).collect();
    let mut value = 0.;
    check(unsafe { cairn_clip(p.as_ptr(), p.len(), max, &mut value) })?;
    Ok(value)
}
pub fn adamw(
    p: &[&Tensor],
    m: &[&Tensor],
    v: &[&Tensor],
    s: &[&Tensor],
    lr: f64,
    wd: f64,
) -> Result<()> {
    let pointers = |ts: &[&Tensor]| ts.iter().map(|t| t.as_ptr()).collect::<Vec<_>>();
    let (p, m, v, s) = (pointers(p), pointers(m), pointers(v), pointers(s));
    anyhow::ensure!(
        p.len() == m.len() && p.len() == v.len() && p.len() == s.len(),
        "Optimizer size mismatch"
    );
    check(unsafe {
        cairn_adamw(
            p.as_ptr(),
            m.as_ptr(),
            v.as_ptr(),
            s.as_ptr(),
            p.len(),
            lr,
            wd,
        )
    })
}
pub fn rng_get(cuda: bool) -> Result<Tensor> {
    let mut out = std::ptr::null_mut();
    check(unsafe { cairn_rng_get(cuda as i32, &mut out) })?;
    Ok(unsafe { Tensor::from_ptr(out) })
}
pub fn rng_set(cuda: bool, state: &Tensor) -> Result<()> {
    check(unsafe { cairn_rng_set(cuda as i32, state.as_ptr()) })
}
