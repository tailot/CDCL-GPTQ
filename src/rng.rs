//! Tiny deterministic PRNG (splitmix64 + Box-Muller). No external dependency.
//!
//! Determinism is the point, not a nicety: `bench` and the engine tests must generate *exactly* the same synthetic
//! weights and activations on every run and every machine, otherwise an A/B comparison between two methods could be
//! explained by a different random draw instead of by the algorithm. That rules out `rand`, whose distributions and
//! seeding are part of its public API surface and can change between versions.
//!
//! Splitmix64 is 3 instructions per output and passes the usual small-state statistical screens well beyond what a
//! benchmark needs (it is not a cryptographic or long-run generator, and it is never used for anything security related).

pub struct Rng {
    s: u64,
    /// Box-Muller produces normals in pairs; the second one is parked here so consecutive calls do not throw away half
    /// their work (and so the stream stays reproducible regardless of call order parity).
    spare: Option<f32>,
}

impl Rng {
    /// Seed with any u64. The golden-ratio increment is added up front so seed 0 and other small seeds do not start at a
    /// weak state (splitmix64's output depends on the *increment*, so an unpremixed zero state would still work, but this
    /// makes neighbouring seeds produce clearly independent streams).
    pub fn new(seed: u64) -> Self {
        Rng { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15), spare: None }
    }

    /// splitmix64 output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    ///
    /// Uses the top 24 bits (the highest-quality bits of a splitmix64 output) and adds 0.5 before scaling, so the result
    /// can never be exactly 0 — `normal()` takes `ln(u1)` and would return -infinity otherwise.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (((self.next_u64() >> 40) as f32) + 0.5) / (1u64 << 24) as f32
    }

    /// Standard normal via Box-Muller, with the partner value cached (see [`Rng::spare`]).
    #[inline]
    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}
