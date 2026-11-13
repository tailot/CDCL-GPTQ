//! Tiny deterministic PRNG (splitmix64 + Box-Muller). No external dependency.

pub struct Rng {
    s: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15), spare: None }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (((self.next_u64() >> 40) as f32) + 0.5) / (1u64 << 24) as f32
    }

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
