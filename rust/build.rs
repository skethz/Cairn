use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    println!("cargo:rerun-if-changed=native/routing.cu");
    println!("cargo:rerun-if-changed=native/compile_cuda.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let root =
        PathBuf::from(env::var("LIBTORCH").expect(
            "Set LIBTORCH to audited local 2.11.0 native libraries; downloads are disabled",
        ));
    let mut b = cc::Build::new();
    let version = std::fs::read_to_string(root.join("include/torch/headeronly/version.h"))
        .expect("Missing native version header");
    for required in [
        "#define TORCH_VERSION_MAJOR 2",
        "#define TORCH_VERSION_MINOR 11",
        "#define TORCH_VERSION_PATCH 0",
    ] {
        assert!(
            version.lines().any(|line| line.trim() == required),
            "Requires native LibTorch 2.11.0; version bypass is not supported"
        );
    }
    b.cpp(true)
        .file("native/bridge.cpp")
        .include(root.join("include"))
        .include(root.join("include/torch/csrc/api/include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-g0")
        .opt_level(1);
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let cuda = PathBuf::from(env::var("CUDA_HOME").expect("Set CUDA_HOME to local CUDA 12.8 toolkit"));
        b.include(cuda.join("include"));
        let out=PathBuf::from(env::var("OUT_DIR").unwrap());
        let compiler=out.join("compile_cuda");
        let status=std::process::Command::new("c++")
            .args(["-O2","-std=c++17","native/compile_cuda.cpp","-I"])
            .arg(cuda.join("include")).arg("-L").arg(cuda.join("lib64"))
            .arg(format!("-Wl,-rpath,{}",cuda.join("lib64").display()))
            .args(["-lnvrtc","-o"]).arg(&compiler).status().expect("Cannot build local CUDA compiler");
        assert!(status.success(),"Build-only NVRTC helper compilation failed");
        let status=std::process::Command::new(compiler).env("LD_LIBRARY_PATH",cuda.join("lib64")).arg("native/routing.cu")
            .arg(out.join("routing_cubin.h")).status().expect("Cannot compile local CUDA source");
        assert!(status.success(),"Offline CUDA cubin compilation failed");
        b.include(&out);
        println!("cargo:rustc-link-search=native={}",cuda.join("lib64").display());
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-arg-bin=cairn=-Wl,-rpath,{}",cuda.join("lib64").display());
        println!("cargo:rustc-link-lib=c10_cuda");
        println!("cargo:rustc-link-search=native={}",cuda.join("lib64/stubs").display());
        println!("cargo:rustc-link-lib=cuda");
        b.define("_GLIBCXX_USE_CXX11_ABI", Some("1"));
        b.define("CAIRN_CUDA", None);
        // CUDA registers operators at library load; --as-needed would drop it.
        // Apply only to the worker, keeping the pre-exec isolation launcher small.
        println!("cargo:rustc-link-arg-bin=cairn=-Wl,--no-as-needed");
        println!(
            "cargo:rustc-link-arg-bin=cairn={}",
            root.join("lib/libtorch_cuda.so").display()
        );
        println!("cargo:rustc-link-arg-bin=cairn=-Wl,--as-needed");
    }
    b.compile("cairn_native");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        root.join("lib").display()
    );
}
