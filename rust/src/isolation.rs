//! Fail-closed OS boundary. This module does not depend on the tensor runtime.
use anyhow::{Result, bail, ensure};
#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::{CString, c_char, c_void};
    #[repr(C)]
    struct Argument {
        arg: u32,
        op: i32,
        a: u64,
        b: u64,
    }
    #[link(name = "libseccomp.so.2", kind = "dylib", modifiers = "+verbatim")]
    unsafe extern "C" {
        fn seccomp_init(default: u32) -> *mut c_void;
        fn seccomp_release(ctx: *mut c_void);
        fn seccomp_syscall_resolve_name(name: *const c_char) -> i32;
        fn seccomp_rule_add_array(
            ctx: *mut c_void,
            action: u32,
            syscall: i32,
            n: u32,
            args: *const Argument,
        ) -> i32;
        fn seccomp_attr_set(ctx: *mut c_void, attr: u32, value: u32) -> i32;
        fn seccomp_load(ctx: *mut c_void) -> i32;
    }
    pub fn activate(tight: bool) -> Result<()> {
        unsafe {
            let p = seccomp_init(0x7fff0000);
            ensure!(!p.is_null(), "seccomp_init failed");
            let result = (|| -> Result<()> {
                for n in ["socket", "socketpair"] {
                    let number = seccomp_syscall_resolve_name(CString::new(n)?.as_ptr());
                    let argument = Argument {
                        arg: 0,
                        op: 1,
                        a: libc::AF_UNIX as u64,
                        b: 0,
                    };
                    ensure!(
                        number >= 0
                            && seccomp_rule_add_array(
                                p,
                                0x00050000 | libc::EPERM as u32,
                                number,
                                1,
                                &argument
                            ) == 0,
                        "Domain filter failed"
                    );
                }
                let mut denied = vec![
                    "socketcall",
                    "io_uring_setup",
                    "io_uring_enter",
                    "io_uring_register",
                    "ptrace",
                    "process_vm_readv",
                    "process_vm_writev",
                ];
                if tight {
                    denied.extend([
                        "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg",
                        "sendmmsg", "recvfrom", "recvmsg", "recvmmsg", "execve", "execveat",
                    ]);
                }
                for n in denied {
                    let num = seccomp_syscall_resolve_name(CString::new(n)?.as_ptr());
                    if num >= 0 {
                        ensure!(
                            seccomp_rule_add_array(
                                p,
                                0x00050000 | libc::EPERM as u32,
                                num,
                                0,
                                std::ptr::null()
                            ) == 0,
                            "System-call filter failed: {n}"
                        );
                    }
                }
                ensure!(
                    seccomp_attr_set(p, 4, 1) == 0 && seccomp_load(p) == 0,
                    "Cannot enforce isolation on all threads"
                );
                Ok(())
            })();
            seccomp_release(p);
            result?;
            for domain in [libc::AF_INET, libc::AF_INET6, libc::AF_PACKET] {
                let fd = libc::socket(domain, libc::SOCK_STREAM, 0);
                if fd >= 0 {
                    libc::close(fd);
                    bail!("OS network negative control failed");
                }
                ensure!(
                    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM),
                    "Unexpected network error"
                );
            }
            Ok(())
        }
    }
}
#[cfg(target_os = "linux")]
pub use linux::activate;
#[cfg(target_os = "linux")]
#[allow(dead_code)] // Used by the worker probe; shared launcher module omits it.
pub fn tight_negative_controls() -> Result<()> {
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0);
        ensure!(fd >= 0, "Cannot create local negative-control descriptor");
        for operation in [libc::SYS_connect, libc::SYS_bind] {
            let status = libc::syscall(operation, fd, std::ptr::null::<libc::sockaddr>(), 0u32);
            ensure!(
                status == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM),
                "Local IPC operation was not blocked"
            );
        }
        libc::close(fd);
        let path = std::ffi::CString::new("/__cairn_exec_denial_probe__")?;
        let result = libc::syscall(
            libc::SYS_execve,
            path.as_ptr(),
            std::ptr::null::<*const i8>(),
            std::ptr::null::<*const i8>(),
        );
        ensure!(
            result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM),
            "Process execution was not blocked"
        );
        let result = libc::syscall(libc::SYS_io_uring_setup, 1u32, std::ptr::null::<u8>());
        ensure!(
            result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM),
            "io_uring was not blocked"
        );
        Ok(())
    }
}
#[cfg(not(target_os = "linux"))]
pub fn tight_negative_controls() -> Result<()> {
    bail!("Unsupported platform")
}
#[cfg(not(target_os = "linux"))]
pub fn activate(_: bool) -> Result<()> {
    bail!("This build requires Linux seccomp; refusing an unisolated runtime")
}
