// HMAC_DRBG per NIST SP 800-90A Rev. 1, Section 10.1.2, built on top of
// this crate's HMAC primitive, plus a thin wrapper over AWS-LC's system
// entropy source (`RAND_bytes`).

use crate::digest::DigestAlg;
use crate::error::{Error, ErrorKind};
use crate::mac::Hmac;

/// Fills `buf` with output from AWS-LC's system CSPRNG.
pub fn get_random(buf: &mut [u8]) -> Result<(), Error> {
    let ret = unsafe { ffi::RAND_bytes(buf.as_mut_ptr(), buf.len()) };
    if ret != 1 {
        return Err(Error::new(ErrorKind::BackendError));
    }
    Ok(())
}

#[derive(Debug)]
pub struct HmacDrbg {
    alg: DigestAlg,
    k: Vec<u8>,
    v: Vec<u8>,
}

impl HmacDrbg {
    /// Core HMAC_DRBG instantiate function (SP 800-90A 10.1.2.3), taking
    /// explicit entropy and nonce. Deterministic given the same inputs —
    /// used directly by tests, and by `new` below.
    pub fn instantiate(
        alg: DigestAlg,
        entropy: &[u8],
        nonce: &[u8],
        personalization: &[u8],
    ) -> Result<HmacDrbg, Error> {
        let outlen = crate::digest::Digest::new(alg)?.size();
        let mut drbg = HmacDrbg {
            alg,
            k: vec![0x00; outlen],
            v: vec![0x01; outlen],
        };
        let mut seed = Vec::with_capacity(
            entropy.len() + nonce.len() + personalization.len(),
        );
        seed.extend_from_slice(entropy);
        seed.extend_from_slice(nonce);
        seed.extend_from_slice(personalization);
        drbg.update(Some(&seed))?;
        Ok(drbg)
    }

    /// Production constructor: draws entropy and nonce from AWS-LC's
    /// system entropy source. The nonce length (half the entropy length)
    /// follows SP 800-90A 8.6.7, which allows drawing the nonce from the
    /// same entropy source as long as it is independent of the entropy
    /// input and at least security_strength/2 bits.
    pub fn new(
        alg: DigestAlg,
        personalization: &[u8],
    ) -> Result<HmacDrbg, Error> {
        let outlen = crate::digest::Digest::new(alg)?.size();
        let mut entropy = vec![0u8; outlen];
        let mut nonce = vec![0u8; outlen / 2];
        get_random(&mut entropy)?;
        get_random(&mut nonce)?;
        Self::instantiate(alg, &entropy, &nonce, personalization)
    }

    /// HMAC_DRBG_Update (SP 800-90A 10.1.2.2).
    fn update(&mut self, provided_data: Option<&[u8]>) -> Result<(), Error> {
        let mut km_input = self.v.clone();
        km_input.push(0x00);
        if let Some(data) = provided_data {
            km_input.extend_from_slice(data);
        }
        self.k = Hmac::mac(self.alg, &self.k, &km_input)?;
        self.v = Hmac::mac(self.alg, &self.k, &self.v)?;

        if let Some(data) = provided_data {
            let mut km_input = self.v.clone();
            km_input.push(0x01);
            km_input.extend_from_slice(data);
            self.k = Hmac::mac(self.alg, &self.k, &km_input)?;
            self.v = Hmac::mac(self.alg, &self.k, &self.v)?;
        }
        Ok(())
    }

    /// HMAC_DRBG_Reseed (SP 800-90A 10.1.2.4). In addition to the
    /// caller-supplied `entropy` (PKCS#11 callers such as C_SeedRandom are
    /// not required to supply full-entropy input), this always mixes in
    /// fresh entropy from AWS-LC's system source, matching the security
    /// posture of `ossl::rand::EvpRandCtx::reseed`, which always sets
    /// OpenSSL's `prediction_resistance` flag.
    pub fn reseed(
        &mut self,
        entropy: &[u8],
        addtl: &[u8],
    ) -> Result<(), Error> {
        let mut fresh = vec![0u8; self.v.len()];
        get_random(&mut fresh)?;
        let mut seed =
            Vec::with_capacity(fresh.len() + entropy.len() + addtl.len());
        seed.extend_from_slice(&fresh);
        seed.extend_from_slice(entropy);
        seed.extend_from_slice(addtl);
        self.update(Some(&seed))
    }

    /// HMAC_DRBG_Generate (SP 800-90A 10.1.2.5).
    pub fn generate(
        &mut self,
        addtl: &[u8],
        output: &mut [u8],
    ) -> Result<(), Error> {
        if !addtl.is_empty() {
            self.update(Some(addtl))?;
        }
        let mut filled = 0;
        while filled < output.len() {
            self.v = Hmac::mac(self.alg, &self.k, &self.v)?;
            let take = std::cmp::min(self.v.len(), output.len() - filled);
            output[filled..filled + take].copy_from_slice(&self.v[..take]);
            filled += take;
        }
        self.update(if addtl.is_empty() { None } else { Some(addtl) })?;
        Ok(())
    }

    /// Approximation of the SP 800-90A minimum entropy input length: the
    /// underlying hash's output length. This is exact for SHA-256 (256-bit
    /// security strength) and conservative (larger than strictly required)
    /// for SHA-512.
    pub fn security_strength_bytes(&self) -> usize {
        self.v.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::DigestAlg;

    #[test]
    fn deterministic_given_same_inputs() {
        let entropy = [0x11u8; 32];
        let nonce = [0x22u8; 16];
        let perso = b"kryoptic-awslc-test";
        let mut a =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, perso)
                .unwrap();
        let mut b =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, perso)
                .unwrap();
        let mut out_a = [0u8; 64];
        let mut out_b = [0u8; 64];
        a.generate(&[], &mut out_a).unwrap();
        b.generate(&[], &mut out_b).unwrap();
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn diverges_on_personalization() {
        let entropy = [0x11u8; 32];
        let nonce = [0x22u8; 16];
        let mut a =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, b"a")
                .unwrap();
        let mut b =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, b"b")
                .unwrap();
        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        a.generate(&[], &mut out_a).unwrap();
        b.generate(&[], &mut out_b).unwrap();
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn successive_generates_differ() {
        let entropy = [0x33u8; 32];
        let nonce = [0x44u8; 16];
        let mut d =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, b"")
                .unwrap();
        let mut first = [0u8; 32];
        let mut second = [0u8; 32];
        d.generate(&[], &mut first).unwrap();
        d.generate(&[], &mut second).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn reseed_changes_output() {
        let entropy = [0x55u8; 32];
        let nonce = [0x66u8; 16];
        let mut d =
            HmacDrbg::instantiate(DigestAlg::Sha2_256, &entropy, &nonce, b"")
                .unwrap();
        let mut before = [0u8; 32];
        d.generate(&[], &mut before).unwrap();
        d.reseed(&[0x77u8; 32], &[]).unwrap();
        let mut after = [0u8; 32];
        d.generate(&[], &mut after).unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn get_random_fills_buffer() {
        let mut buf = [0u8; 32];
        get_random(&mut buf).unwrap();
        assert_ne!(buf, [0u8; 32]);
    }

    #[test]
    fn new_produces_working_generator() {
        // Uses the system-entropy-seeded constructor end to end.
        let mut d = HmacDrbg::new(DigestAlg::Sha2_256, b"kryoptic").unwrap();
        let mut out = [0u8; 32];
        d.generate(&[], &mut out).unwrap();
        assert_ne!(out, [0u8; 32]);
    }
}
