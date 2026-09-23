// Error type for the `awslc` crate, mirroring the shape of `ossl::Error`
// so `src/error.rs` can convert both the same way.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorKind {
    /// AWS-LC returned a NULL pointer where a valid one was expected.
    NullPtr,
    /// An AWS-LC C function returned a failure status code.
    BackendError,
    /// A caller-provided buffer was too small.
    BufferSize,
    /// An error internal to this crate's Rust-level logic (not AWS-LC's).
    WrapperError,
    /// A cryptographic verification (e.g. an AEAD tag check) failed.
    VerifyFailed,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn new(kind: ErrorKind) -> Error {
        Error { kind }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "awslc backend error: {:?}", self.kind)
    }
}

impl std::error::Error for Error {}
