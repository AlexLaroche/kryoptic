//! Safe Rust wrappers around AWS-LC (via `aws-lc-sys`), providing the
//! primitives kryoptic's `src/awslc/*` module tree needs to implement
//! kryoptic's backend-agnostic `Mechanism` traits.

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
