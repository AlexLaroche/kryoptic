//! Safe Rust wrappers around AWS-LC-FIPS (via `aws-lc-fips-sys`), providing
//! the primitives kryoptic's `src/awslc/*` module tree needs to implement
//! kryoptic's backend-agnostic `Mechanism` traits. Shares its entire
//! implementation with the `awslc` crate via `include!`d files in
//! `awslc-shared/` — see that crate's `src/lib.rs` and
//! `docs/superpowers/specs/2026-09-22-awslc-fips-support-design.md` §4.

pub mod error;
pub use error::{Error, ErrorKind};
pub mod cipher;
pub mod dh;
pub mod digest;
pub mod ec;
pub mod eddsa;
pub mod hkdf;
pub mod kbkdf;
pub mod mac;
pub mod mldsa;
pub mod mlkem;
pub mod pbkdf2;
pub mod rand;
pub mod rsa;
pub mod sshkdf;
pub mod x25519;
