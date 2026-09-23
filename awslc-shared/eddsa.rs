// Ed25519, wrapping AWS-LC's dedicated raw ED25519_* functions (no
// EC_KEY/EVP_PKEY involved -- Ed25519 has its own simpler API in AWS-LC,
// following BoringSSL's convention).

use crate::error::{Error, ErrorKind};

/// Holds the AWS-LC 64-byte expanded form (32-byte seed + 32-byte public
/// key) -- what ED25519_sign actually needs -- plus the 32-byte seed alone,
/// separately, since that's PKCS#11's CKA_VALUE storage format and the two
/// aren't trivially separable from the 64-byte form without re-deriving.
pub struct Ed25519Key {
    seed: [u8; 32],
    expanded_private: [u8; 64],
    public: [u8; 32],
}

/// Hand-written rather than `#[derive(Debug)]`: the derived impl would
/// print `seed`/`expanded_private` -- the raw private key material -- in
/// cleartext on any `{:?}` format of this type (or of anything that
/// transitively derives `Debug` over a value holding one, e.g.
/// `crate::awslc::eddsa::EddsaKey`/`EddsaOperation` in the kryoptic crate).
/// Only the public key is safe to show.
impl std::fmt::Debug for Ed25519Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ed25519Key")
            .field("seed", &"[REDACTED]")
            .field("expanded_private", &"[REDACTED]")
            .field("public", &self.public)
            .finish()
    }
}

impl Ed25519Key {
    pub fn generate() -> Result<Ed25519Key, Error> {
        let mut public = [0u8; 32];
        let mut expanded_private = [0u8; 64];
        unsafe {
            ffi::ED25519_keypair(
                public.as_mut_ptr(),
                expanded_private.as_mut_ptr(),
            );
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&expanded_private[..32]);
        Ok(Ed25519Key {
            seed,
            expanded_private,
            public,
        })
    }

    /// Reconstructs a key from PKCS#11's stored 32-byte seed.
    pub fn from_seed(seed: &[u8; 32]) -> Result<Ed25519Key, Error> {
        let mut public = [0u8; 32];
        let mut expanded_private = [0u8; 64];
        // Confirmed against the vendored aws-lc-sys bindings
        // (x86_64_unknown_linux_gnu_crypto.rs): ED25519_keypair_from_seed
        // returns void, matching ED25519_keypair's own convention -- unlike
        // ED25519_sign/ED25519_verify, which do return c_int.
        unsafe {
            ffi::ED25519_keypair_from_seed(
                public.as_mut_ptr(),
                expanded_private.as_mut_ptr(),
                seed.as_ptr(),
            );
        }
        Ok(Ed25519Key {
            seed: *seed,
            expanded_private,
            public,
        })
    }

    pub fn seed(&self) -> [u8; 32] {
        self.seed
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.public
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        let mut sig = [0u8; 64];
        let ret = unsafe {
            ffi::ED25519_sign(
                sig.as_mut_ptr(),
                msg.as_ptr(),
                msg.len(),
                self.expanded_private.as_ptr(),
            )
        };
        // Unlike ED25519_keypair_from_seed, ED25519_sign does return
        // c_int (1 on success). The interface fixed by this crate's
        // consumers returns the signature bytes directly rather than a
        // Result, so on the (effectively unreachable outside of an
        // allocation failure) error path we panic rather than silently
        // handing back a bogus, uninitialized-looking all-zero signature.
        assert_eq!(ret, 1, "ED25519_sign unexpectedly failed");
        sig
    }

    pub fn verify(
        public_key: &[u8; 32],
        msg: &[u8],
        sig: &[u8; 64],
    ) -> Result<(), Error> {
        let ret = unsafe {
            ffi::ED25519_verify(
                msg.as_ptr(),
                msg.len(),
                sig.as_ptr(),
                public_key.as_ptr(),
            )
        };
        if ret != 1 {
            return Err(Error::new(ErrorKind::VerifyFailed));
        }
        Ok(())
    }
}

impl Drop for Ed25519Key {
    fn drop(&mut self) {
        for byte in self.seed.iter_mut() {
            unsafe { std::ptr::write_volatile(byte, 0) };
        }
        for byte in self.expanded_private.iter_mut() {
            unsafe { std::ptr::write_volatile(byte, 0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

unsafe impl Send for Ed25519Key {}
unsafe impl Sync for Ed25519Key {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_sign_verify_round_trip() {
        let key = Ed25519Key::generate().unwrap();
        let msg = b"ed25519 test message";
        let sig = key.sign(msg);
        Ed25519Key::verify(&key.public_key(), msg, &sig).unwrap();
    }

    #[test]
    fn tampered_message_rejected() {
        let key = Ed25519Key::generate().unwrap();
        let sig = key.sign(b"real message");
        assert!(Ed25519Key::verify(
            &key.public_key(),
            b"different message",
            &sig
        )
        .is_err());
    }

    #[test]
    fn from_seed_reproduces_same_keypair() {
        let key1 = Ed25519Key::generate().unwrap();
        let seed = key1.seed();
        let key2 = Ed25519Key::from_seed(&seed).unwrap();
        assert_eq!(key1.public_key(), key2.public_key());
        // A signature from the reconstructed key must verify against the
        // original's public key too (same keypair).
        let sig = key2.sign(b"consistency check");
        Ed25519Key::verify(&key1.public_key(), b"consistency check", &sig)
            .unwrap();
    }

    /// `{:?}`-formatting an `Ed25519Key` must never leak the raw private
    /// seed or expanded private key -- only the (safe to expose) public
    /// key should be visible.
    #[test]
    fn debug_format_redacts_private_key_material() {
        let key = Ed25519Key::generate().unwrap();
        let debug_str = format!("{:?}", key);

        assert!(!debug_str.contains(&format!("{:?}", key.seed)));
        assert!(!debug_str.contains(&format!("{:?}", key.expanded_private)));
        assert!(debug_str.contains("REDACTED"));
        // The public key is not secret and is fine to show.
        assert!(debug_str.contains(&format!("{:?}", key.public)));
    }
}
