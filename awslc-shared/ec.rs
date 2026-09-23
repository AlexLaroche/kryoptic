// ECDSA and ECDH, wrapping AWS-LC's classic EC_KEY/EC_GROUP/ECDSA_SIG API.

use crate::error::{Error, ErrorKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EcCurve {
    P256,
    P384,
    P521,
}

impl EcCurve {
    fn nid(self) -> i32 {
        match self {
            EcCurve::P256 => ffi::NID_X9_62_prime256v1,
            EcCurve::P384 => ffi::NID_secp384r1,
            EcCurve::P521 => ffi::NID_secp521r1,
        }
    }

    /// Order size in bytes -- the fixed width for r/s and for the raw
    /// private scalar. (P-521's order needs 66 bytes: ceil(521/8).)
    pub fn order_bytes(self) -> usize {
        match self {
            EcCurve::P256 => 32,
            EcCurve::P384 => 48,
            EcCurve::P521 => 66,
        }
    }
}

#[derive(Debug)]
pub struct EcKey {
    key: *mut ffi::EC_KEY,
    curve: EcCurve,
}

impl EcKey {
    pub fn generate(curve: EcCurve) -> Result<EcKey, Error> {
        let key = unsafe { ffi::EC_KEY_new_by_curve_name(curve.nid()) };
        if key.is_null() {
            return Err(Error::new(ErrorKind::NullPtr));
        }
        if unsafe { ffi::EC_KEY_generate_key(key) } != 1 {
            unsafe { ffi::EC_KEY_free(key) };
            return Err(Error::new(ErrorKind::BackendError));
        }
        Ok(EcKey { key, curve })
    }

    /// Reconstructs a key from a raw, fixed-width private scalar (matching
    /// PKCS#11's CKA_VALUE for CKK_EC). Also derives and sets the public
    /// point, since EC_KEY_set_private_key alone doesn't compute it.
    pub fn from_private_scalar(
        curve: EcCurve,
        scalar: &[u8],
    ) -> Result<EcKey, Error> {
        if scalar.len() != curve.order_bytes() {
            return Err(Error::new(ErrorKind::WrapperError));
        }
        let key = unsafe { ffi::EC_KEY_new_by_curve_name(curve.nid()) };
        if key.is_null() {
            return Err(Error::new(ErrorKind::NullPtr));
        }
        let result = (|| -> Result<(), Error> {
            let bn = unsafe {
                ffi::BN_bin2bn(
                    scalar.as_ptr(),
                    scalar.len(),
                    std::ptr::null_mut(),
                )
            };
            if bn.is_null() {
                return Err(Error::new(ErrorKind::NullPtr));
            }
            let ret = unsafe { ffi::EC_KEY_set_private_key(key, bn) };
            unsafe { ffi::BN_free(bn) };
            if ret != 1 {
                return Err(Error::new(ErrorKind::WrapperError));
            }
            // Derive the public point: pub = priv * G.
            let group = unsafe { ffi::EC_KEY_get0_group(key) };
            let point = unsafe { ffi::EC_POINT_new(group) };
            if point.is_null() {
                return Err(Error::new(ErrorKind::NullPtr));
            }
            let priv_bn = unsafe { ffi::EC_KEY_get0_private_key(key) };
            let ret = unsafe {
                ffi::EC_POINT_mul(
                    group,
                    point,
                    priv_bn,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            if ret != 1 {
                unsafe { ffi::EC_POINT_free(point) };
                return Err(Error::new(ErrorKind::BackendError));
            }
            let ret = unsafe { ffi::EC_KEY_set_public_key(key, point) };
            unsafe { ffi::EC_POINT_free(point) };
            if ret != 1 {
                return Err(Error::new(ErrorKind::BackendError));
            }
            Ok(())
        })();
        if let Err(e) = result {
            unsafe { ffi::EC_KEY_free(key) };
            return Err(e);
        }
        Ok(EcKey { key, curve })
    }

    /// Reconstructs a public-key-only key from a raw, uncompressed public
    /// point (PKCS#11's CKA_EC_POINT raw-octet format: 0x04 || X || Y, same
    /// as `public_point()`'s own output). No private scalar is set or
    /// derivable from this -- the resulting `EcKey` is only ever valid for
    /// `verify()`, never `sign()` (AWS-LC's `ECDSA_do_sign` would fail on it
    /// since `EC_KEY_get0_private_key` returns null). This is the
    /// counterpart to `from_private_scalar` for the case PKCS#11 verify
    /// operations always have: a `CKO_PUBLIC_KEY` object with a point but no
    /// `CKA_VALUE`.
    pub fn from_public_point(
        curve: EcCurve,
        point: &[u8],
    ) -> Result<EcKey, Error> {
        let key = unsafe { ffi::EC_KEY_new_by_curve_name(curve.nid()) };
        if key.is_null() {
            return Err(Error::new(ErrorKind::NullPtr));
        }
        let result = (|| -> Result<(), Error> {
            let group = unsafe { ffi::EC_KEY_get0_group(key) };
            let ec_point = unsafe { ffi::EC_POINT_new(group) };
            if ec_point.is_null() {
                return Err(Error::new(ErrorKind::NullPtr));
            }
            // Parse the caller-supplied raw octets into `ec_point`, the same
            // way `derive_shared_secret` parses a peer's point below.
            let ret = unsafe {
                ffi::EC_POINT_oct2point(
                    group,
                    ec_point,
                    point.as_ptr(),
                    point.len(),
                    std::ptr::null_mut(),
                )
            };
            if ret != 1 {
                unsafe { ffi::EC_POINT_free(ec_point) };
                return Err(Error::new(ErrorKind::WrapperError));
            }
            // EC_KEY_set_public_key copies the point's data into the key's
            // own internally-owned point (same as in from_private_scalar
            // above) -- `ec_point` is never owned by `key` and must be freed
            // here regardless of the outcome.
            let ret = unsafe { ffi::EC_KEY_set_public_key(key, ec_point) };
            unsafe { ffi::EC_POINT_free(ec_point) };
            if ret != 1 {
                return Err(Error::new(ErrorKind::BackendError));
            }
            Ok(())
        })();
        if let Err(e) = result {
            unsafe { ffi::EC_KEY_free(key) };
            return Err(e);
        }
        Ok(EcKey { key, curve })
    }

    /// Raw, fixed-width big-endian private scalar (PKCS#11's CKA_VALUE
    /// format for CKK_EC).
    ///
    /// Returns `Err` (rather than dereferencing a null `BIGNUM*`) if `self`
    /// has no private scalar at all -- possible since `from_public_point`
    /// produces a key with only a public point set, for which
    /// `EC_KEY_get0_private_key` returns NULL (confirmed against AWS-LC's
    /// own `crypto/fipsmodule/ec/ec_key.c`), and `BN_bn2bin_padded`
    /// dereferences its `BIGNUM*` argument unconditionally (confirmed
    /// against AWS-LC's own `crypto/fipsmodule/bn/bytes.c`). Not reachable
    /// today (this method is only ever called on `generate`/
    /// `from_private_scalar` output), but this is a latent null-deref that
    /// a future caller could otherwise trip with no compile-time signal.
    pub fn private_scalar(&self) -> Result<Vec<u8>, Error> {
        let bn = unsafe { ffi::EC_KEY_get0_private_key(self.key) };
        if bn.is_null() {
            return Err(Error::new(ErrorKind::NullPtr));
        }
        let mut out = vec![0u8; self.curve.order_bytes()];
        let ret = unsafe {
            ffi::BN_bn2bin_padded(out.as_mut_ptr(), out.len(), bn)
        };
        if ret != 1 {
            return Err(Error::new(ErrorKind::BackendError));
        }
        Ok(out)
    }

    /// Uncompressed EC point encoding (PKCS#11's CKA_EC_POINT format,
    /// though the caller is responsible for the extra ASN.1 OCTET STRING
    /// wrapping PKCS#11 actually requires -- this returns just the raw
    /// point octets: 0x04 || X || Y).
    pub fn public_point(&self) -> Vec<u8> {
        let group = unsafe { ffi::EC_KEY_get0_group(self.key) };
        let point = unsafe { ffi::EC_KEY_get0_public_key(self.key) };
        let len = unsafe {
            ffi::EC_POINT_point2oct(
                group,
                point,
                ffi::point_conversion_form_t::POINT_CONVERSION_UNCOMPRESSED,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        let mut out = vec![0u8; len];
        unsafe {
            ffi::EC_POINT_point2oct(
                group,
                point,
                ffi::point_conversion_form_t::POINT_CONVERSION_UNCOMPRESSED,
                out.as_mut_ptr(),
                out.len(),
                std::ptr::null_mut(),
            );
        }
        out
    }

    /// Signs a pre-computed digest, returning raw fixed-width (r, s) --
    /// never DER. `digest` is whatever bytes the caller's chosen hash
    /// produced; ECDSA itself is agnostic to the hash algorithm and to
    /// whether digest.len() matches the curve's order size (kryoptic's own
    /// mechanism dispatch has already picked the hash before calling this).
    pub fn sign(&self, digest: &[u8]) -> Result<(Vec<u8>, Vec<u8>), Error> {
        let sig = unsafe {
            ffi::ECDSA_do_sign(digest.as_ptr(), digest.len(), self.key)
        };
        if sig.is_null() {
            return Err(Error::new(ErrorKind::BackendError));
        }
        let order_bytes = self.curve.order_bytes();
        let mut r = vec![0u8; order_bytes];
        let mut s = vec![0u8; order_bytes];
        let r_bn = unsafe { ffi::ECDSA_SIG_get0_r(sig) };
        let s_bn = unsafe { ffi::ECDSA_SIG_get0_s(sig) };
        let ret_r =
            unsafe { ffi::BN_bn2bin_padded(r.as_mut_ptr(), r.len(), r_bn) };
        let ret_s =
            unsafe { ffi::BN_bn2bin_padded(s.as_mut_ptr(), s.len(), s_bn) };
        unsafe { ffi::ECDSA_SIG_free(sig) };
        if ret_r != 1 || ret_s != 1 {
            return Err(Error::new(ErrorKind::BackendError));
        }
        Ok((r, s))
    }

    /// Verifies a raw fixed-width (r, s) signature against a pre-computed
    /// digest.
    pub fn verify(
        &self,
        digest: &[u8],
        r: &[u8],
        s: &[u8],
    ) -> Result<(), Error> {
        let order_bytes = self.curve.order_bytes();
        if r.len() != order_bytes || s.len() != order_bytes {
            return Err(Error::new(ErrorKind::WrapperError));
        }
        let r_bn = unsafe {
            ffi::BN_bin2bn(r.as_ptr(), r.len(), std::ptr::null_mut())
        };
        let s_bn = unsafe {
            ffi::BN_bin2bn(s.as_ptr(), s.len(), std::ptr::null_mut())
        };
        if r_bn.is_null() || s_bn.is_null() {
            // Free whichever of the two, if either, was actually allocated.
            if !r_bn.is_null() {
                unsafe { ffi::BN_free(r_bn) };
            }
            if !s_bn.is_null() {
                unsafe { ffi::BN_free(s_bn) };
            }
            return Err(Error::new(ErrorKind::NullPtr));
        }
        let sig = unsafe { ffi::ECDSA_SIG_new() };
        if sig.is_null() {
            unsafe {
                ffi::BN_free(r_bn);
                ffi::BN_free(s_bn);
            }
            return Err(Error::new(ErrorKind::NullPtr));
        }
        // ECDSA_SIG_set0 takes ownership of r_bn/s_bn on success.
        if unsafe { ffi::ECDSA_SIG_set0(sig, r_bn, s_bn) } != 1 {
            unsafe { ffi::ECDSA_SIG_free(sig) };
            return Err(Error::new(ErrorKind::BackendError));
        }
        let ret = unsafe {
            ffi::ECDSA_do_verify(digest.as_ptr(), digest.len(), sig, self.key)
        };
        unsafe { ffi::ECDSA_SIG_free(sig) };
        if ret != 1 {
            return Err(Error::new(ErrorKind::VerifyFailed));
        }
        Ok(())
    }

    /// Computes the ECDH shared secret with a peer's public point (raw
    /// uncompressed point octets, same format `public_point()` returns).
    pub fn derive_shared_secret(
        &self,
        peer_point: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let group = unsafe { ffi::EC_KEY_get0_group(self.key) };
        let point = unsafe { ffi::EC_POINT_new(group) };
        if point.is_null() {
            return Err(Error::new(ErrorKind::NullPtr));
        }
        let ret = unsafe {
            ffi::EC_POINT_oct2point(
                group,
                point,
                peer_point.as_ptr(),
                peer_point.len(),
                std::ptr::null_mut(),
            )
        };
        if ret != 1 {
            unsafe { ffi::EC_POINT_free(point) };
            return Err(Error::new(ErrorKind::WrapperError));
        }
        // Field size in bytes -- the shared secret's x-coordinate is this
        // wide (rounding up, matching order_bytes' same ceil-division
        // shape for these curves).
        let field_bytes = self.curve.order_bytes();
        let mut secret = vec![0u8; field_bytes];
        let n = unsafe {
            ffi::ECDH_compute_key(
                secret.as_mut_ptr() as *mut std::os::raw::c_void,
                secret.len(),
                point,
                self.key,
                None,
            )
        };
        unsafe { ffi::EC_POINT_free(point) };
        if n < 0 {
            return Err(Error::new(ErrorKind::BackendError));
        }
        secret.truncate(n as usize);
        Ok(secret)
    }
}

impl Drop for EcKey {
    fn drop(&mut self) {
        unsafe { ffi::EC_KEY_free(self.key) };
    }
}

unsafe impl Send for EcKey {}
unsafe impl Sync for EcKey {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p256_sign_verify_round_trip() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let digest = [0x22u8; 32];
        let (r, s) = key.sign(&digest).unwrap();
        assert_eq!(r.len(), 32);
        assert_eq!(s.len(), 32);
        key.verify(&digest, &r, &s).unwrap();
    }

    #[test]
    fn p256_tampered_signature_rejected() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let digest = [0x33u8; 32];
        let (mut r, s) = key.sign(&digest).unwrap();
        r[0] ^= 0xFF;
        assert!(key.verify(&digest, &r, &s).is_err());
    }

    #[test]
    fn p384_and_p521_sign_verify_round_trip() {
        for (curve, order_bytes) in
            [(EcCurve::P384, 48), (EcCurve::P521, 66)]
        {
            let key = EcKey::generate(curve).unwrap();
            let digest = [0x44u8; 32]; // ECDSA signs whatever digest bytes it's given, regardless of curve size
            let (r, s) = key.sign(&digest).unwrap();
            assert_eq!(r.len(), order_bytes);
            assert_eq!(s.len(), order_bytes);
            key.verify(&digest, &r, &s).unwrap();
        }
    }

    #[test]
    fn verify_rejects_wrong_length_r() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let digest = [0x55u8; 32];
        let (r, s) = key.sign(&digest).unwrap();
        // Truncate r so it's no longer the curve's fixed order width; this
        // must be rejected before ever reaching AWS-LC's BN_bin2bn/verify.
        let short_r = &r[1..];
        assert!(key.verify(&digest, short_r, &s).is_err());
    }

    #[test]
    fn private_scalar_round_trip() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let scalar = key.private_scalar().unwrap();
        assert_eq!(scalar.len(), 32);
        let key2 = EcKey::from_private_scalar(EcCurve::P256, &scalar).unwrap();
        // Same private scalar must produce the same public point.
        assert_eq!(key.public_point(), key2.public_point());
    }

    /// Regression test (whole-phase review, Finding 2): `private_scalar()`
    /// on a key built via `from_public_point` (no private scalar at all --
    /// exactly the shape a PKCS#11 CKO_PUBLIC_KEY object's key produces)
    /// must return an `Err`, not null-deref/panic/UB.
    #[test]
    fn private_scalar_on_public_only_key_errs() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let pub_only =
            EcKey::from_public_point(EcCurve::P256, &key.public_point())
                .unwrap();
        assert!(pub_only.private_scalar().is_err());
    }

    #[test]
    fn from_public_point_verifies_signature_from_private_key() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let digest = [0x66u8; 32];
        let (r, s) = key.sign(&digest).unwrap();
        // Reconstruct a public-key-only key from just the raw point octets
        // (no private scalar involved at all) and confirm it can verify a
        // signature the original (full) key produced -- this is the exact
        // shape a PKCS#11 CKO_PUBLIC_KEY object is in: a point, no CKA_VALUE.
        let pub_only =
            EcKey::from_public_point(EcCurve::P256, &key.public_point())
                .unwrap();
        pub_only.verify(&digest, &r, &s).unwrap();
    }

    #[test]
    fn from_public_point_round_trip_p384_and_p521() {
        for curve in [EcCurve::P384, EcCurve::P521] {
            let key = EcKey::generate(curve).unwrap();
            let digest = [0x77u8; 32];
            let (r, s) = key.sign(&digest).unwrap();
            let pub_only =
                EcKey::from_public_point(curve, &key.public_point()).unwrap();
            pub_only.verify(&digest, &r, &s).unwrap();
        }
    }

    #[test]
    fn from_public_point_rejects_malformed_point() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let point = key.public_point();
        // Truncated point octets: AWS-LC's EC_POINT_oct2point must reject
        // this outright, mirroring ecdh_rejects_malformed_peer_point above.
        let truncated = &point[..point.len() - 1];
        assert!(EcKey::from_public_point(EcCurve::P256, truncated).is_err());

        // Empty input.
        assert!(EcKey::from_public_point(EcCurve::P256, &[]).is_err());
    }

    #[test]
    fn from_public_point_rejects_point_on_wrong_curve() {
        // A well-formed point, but for the wrong curve: same length class
        // never applies here (P256 vs P384 points differ in length), so
        // this exercises EC_POINT_oct2point's own length/format validation
        // against the P384 group rather than a length mismatch.
        let key = EcKey::generate(EcCurve::P256).unwrap();
        let point = key.public_point();
        assert!(EcKey::from_public_point(EcCurve::P384, &point).is_err());
    }

    #[test]
    fn ecdh_p256_shared_secret_agreement() {
        let key1 = EcKey::generate(EcCurve::P256).unwrap();
        let key2 = EcKey::generate(EcCurve::P256).unwrap();
        let secret1 = key1.derive_shared_secret(&key2.public_point()).unwrap();
        let secret2 = key2.derive_shared_secret(&key1.public_point()).unwrap();
        assert_eq!(secret1, secret2);
        assert_eq!(secret1.len(), 32); // P-256's field size
    }

    #[test]
    fn ecdh_rejects_malformed_peer_point() {
        let key = EcKey::generate(EcCurve::P256).unwrap();
        // Truncated point octets: AWS-LC's EC_POINT_oct2point must reject
        // this outright (wrong length for the uncompressed-point format on
        // this curve), not silently accept a truncated/garbage point.
        let peer_point = key.public_point();
        let truncated = &peer_point[..peer_point.len() - 1];
        assert!(key.derive_shared_secret(truncated).is_err());

        // Empty input.
        assert!(key.derive_shared_secret(&[]).is_err());
    }
}
