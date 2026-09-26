// Copyright 2026
// See LICENSE.txt file for terms
//
// Compiles `awslc-shared/csrc/sshkdf_shim.c`, which reaches AWS-LC's
// SSHKDF() directly since aws-lc-sys never exposes it (see that file
// for why). `DEP_AWS_LC_0_34_0_INCLUDE` is aws-lc-sys 0.34.0's
// `links`-derived include-path variable (Cargo's standard
// `DEP_<LINKS>_<KEY>` convention); update this key if the aws-lc-sys
// version pinned in Cargo.toml changes.

fn main() {
    println!("cargo:rerun-if-changed=../awslc-shared/csrc/sshkdf_shim.c");
    let include = std::env::var("DEP_AWS_LC_0_34_0_INCLUDE").expect(
        "aws-lc-sys did not export its include path; did its version change?",
    );
    cc::Build::new()
        .file("../awslc-shared/csrc/sshkdf_shim.c")
        .include(include)
        .compile("kryoptic_awslc_sshkdf_shim");
}
