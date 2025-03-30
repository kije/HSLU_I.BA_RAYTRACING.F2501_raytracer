use rand::{SeedableRng, thread_rng};

use rand::prelude::SmallRng;

#[inline(always)]
pub(crate) fn pseudo_rng() -> SmallRng {
    SmallRng::from_rng(thread_rng()).expect("ThreadRng initialize Failed")
}
