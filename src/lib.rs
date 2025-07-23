//! A library for calculating player performance ratings based on battle results.
//!
//! This library implements a Bradley-Terry model based performance system that estimates player performance
//! levels from head-to-head battle outcomes. The algorithm uses Newton-Raphson iteration to find
//! maximum likelihood estimates of player ratings that best explain the observed win/loss patterns.
//!
//! # Usage
//!
//! ```rust
//! use nypc_perf::{BattleResult, PerfCalc, Rating};
//!
//! // Create battle results between players
//! let battles = vec![
//!     BattleResult {
//!         i: 0,           // Player 0
//!         j: 1,           // Player 1
//!         wij: 2.0,       // Player 0 won twice against Player 1
//!         wji: 1.0,       // Player 1 won once against Player 0
//!     },
//!     BattleResult {
//!         i: 0,
//!         j: 2,
//!         wij: 1.0,
//!         wji: 0.0,
//!     }
//! ];
//!
//! // Initialize performance ratings to 0
//! let mut perf = vec![Rating::new(0.0), Rating::new(0.0), Rating::new_fixed(0.0)];
//!
//! // Run the rating calculation
//! let result = PerfCalc::new()
//!     .max_iters(100)     // Maximum iterations (default: 100)
//!     .epsilon(1e-6)      // Convergence threshold (default: 1e-6)
//!     .run(&mut perf, &battles);
//!
//! // Check if calculation converged
//! match result {
//!     Ok(iters) => println!("Converged after {} iterations", iters),
//!     Err(err) => println!("Did not converge, final error: {}", err)
//! }
//!
//! // perf now contains the estimated log-performance ratings
//! // Higher values indicate better performance
//! ```
//!
//! # Mathematical Model
//!
//! The library uses the Bradley-Terry model. Under this model, the probability of
//! player i winning against player j is:
//!
//! P(i beats j) = 1 / (1 + exp(βⱼ - βᵢ))
//!
//! where βᵢ is the log-performance rating of player i.
//!
//! The algorithm finds values of βᵢ that maximize the likelihood of the observed
//! battle outcomes, with normal prior. For more details on the mathematical foundations
//! and implementation, see `docs/rating.pdf`.
//!
//! # Performance
//!
//! The algorithm typically converges within 10-20 iterations for moderately sized problems
//! (100s of players, 10000s of battles).

/// Represents the outcome of battles between two players.
/// wij is the number of wins of i over j, wji is the number of wins of j over i.
/// The scale matter, since we have the normal prior. Higher number means higher observance.
#[derive(Debug, Clone, Copy)]
pub struct BattleResult {
    pub i: usize,
    pub j: usize,
    /// Number of wins of i over j
    pub wij: f64,
    /// Number of wins of j over i
    pub wji: f64,
}

/// Represents a player's rating.
/// If fixed, the value is fixed and cannot be changed.
/// If not fixed, the value is a variable and can be changed.
/// The value is the log-performance rating of the player.
#[derive(Debug, Clone, Copy)]
pub struct Rating {
    pub fixed: bool,
    pub value: f64,
}

impl Rating {
    pub fn new(value: f64) -> Self {
        Self {
            fixed: false,
            value,
        }
    }
    pub fn new_fixed(value: f64) -> Self {
        Self { fixed: true, value }
    }
}

/// Calculates new beta values based on current beta values and battle results.
///
/// # Arguments
/// * `beta` - Current log(performance π_i) values
/// * `battles` - Iterator of battle results
///
/// # Returns
/// Vector of new beta values after one Newton-Raphson iteration
fn iterate(beta: &[Rating], battles: impl IntoIterator<Item = BattleResult>) -> Vec<Rating> {
    let n = beta.len();
    let mut f = vec![0.0; n];
    let mut df = vec![0.0; n];

    // Pre-calculate sums for each player
    for BattleResult { i, j, wij, wji } in battles {
        let eji = (beta[j].value - beta[i].value).exp();
        let eij = (beta[i].value - beta[j].value).exp();
        f[i] += (eji * wij - wji) / (1.0 + eji);
        df[i] += (eji * (wij + wji)) / ((1.0 + eji) * (1.0 + eji));
        f[j] += (eij * wji - wij) / (1.0 + eij);
        df[j] += (eij * (wji + wij)) / ((1.0 + eij) * (1.0 + eij));
    }

    // Apply Newton-Raphson iteration for each player
    beta.iter()
        .zip(f.iter().zip(&df))
        .map(|(b, (f, df))| Rating {
            fixed: b.fixed,
            value: if b.fixed {
                b.value
            } else {
                b.value - (b.value - f) / (1.0 + df)
            },
        })
        .collect()
}

/// Builder for configuring and running the performance calculation algorithm
/// This builder is used to configure the number of iterations and the convergence threshold.
/// The default settings are:
/// - Maximum iterations: 100
/// - Convergence threshold: 1e-6
///
/// # Example
///
/// ```rust
/// use nypc_perf::{BattleResult, PerfCalc, Rating};
/// let mut perf = vec![Rating::new(0.0); 2];
/// let battles = vec![
///     BattleResult {
///         i: 0,
///         j: 1,
///         wij: 2.0,
///         wji: 1.0,
///     }
/// ];
///
/// let calc = PerfCalc::new().max_iters(100).epsilon(1e-6);
/// calc.run(&mut perf, &battles);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PerfCalc {
    num_iters: usize,
    eps: f64,
}

impl Default for PerfCalc {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfCalc {
    /// Creates a new `PerfCalc` with default settings.
    ///
    /// The default settings are:
    /// - Maximum iterations: 100
    /// - Convergence threshold: 1e-6
    pub fn new() -> Self {
        Self {
            num_iters: 100,
            eps: 1e-6,
        }
    }

    /// Sets the maximum number of iterations.
    pub fn max_iters(mut self, max_iterations: usize) -> Self {
        self.num_iters = max_iterations;
        self
    }

    /// Sets the convergence threshold.
    ///
    /// ```rust
    /// use nypc_perf::PerfCalc;
    ///
    /// let calc = PerfCalc::new().epsilon(1e-6);
    /// ```
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.eps = epsilon;
        self
    }

    /// Runs the iterative algorithm to calculate performance ratings.
    ///
    /// # Arguments
    /// * `perf` - Current performance rating values
    /// * `battles` - Iterator of battle results
    ///
    /// # Returns
    /// Ok(iterations) if convergence is reached within max_iterations,
    /// or Err(final_error) if not converged.
    pub fn run(self, perf: &mut [Rating], battles: &[BattleResult]) -> Result<usize, f64> {
        let mut err = f64::NAN;
        for i in 0..self.num_iters {
            let new_perf = iterate(perf, battles.iter().copied());

            err = new_perf
                .iter()
                .zip(perf.iter())
                .map(|(a, b)| (a.value - b.value).abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            perf.copy_from_slice(&new_perf);

            if err < self.eps {
                return Ok(i + 1);
            }
        }
        Err(err)
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_simple() {
        let battles = vec![
            BattleResult {
                i: 0,
                j: 1,
                wij: 2.0,
                wji: 1.0,
            },
            BattleResult {
                i: 0,
                j: 2,
                wij: 1.0,
                wji: 0.0,
            },
        ];

        let mut perf = vec![Rating::new(0.0), Rating::new(0.0), Rating::new_fixed(0.0)];
        let res = PerfCalc::new().epsilon(1e-6).run(&mut perf, &battles);
        assert!(res.is_ok());
        assert!((perf[0].value - 0.473034477).abs() < 1e-6);
        assert!((perf[1].value - (-0.0891364)).abs() < 1e-6);
        assert!((perf[2].value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_coalesece() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2025);
        const N: usize = 10;
        let mut battles = vec![];
        let mut battles_merge = vec![];
        for i in 0..N {
            for j in 0..i {
                let mut wij = 0.0;
                let mut wji = 0.0;
                for _ in 0..5 {
                    let s = rng.random_bool(0.5);
                    let p = rng.random_range(0..=2) as f64 / 2.0;
                    if s {
                        wij += p;
                        wji += 1.0 - p;
                        battles.push(BattleResult {
                            i,
                            j,
                            wij: p,
                            wji: 1.0 - p,
                        });
                    } else {
                        wij += 1.0 - p;
                        wji += p;
                        battles.push(BattleResult {
                            j,
                            i,
                            wij: 1.0 - p,
                            wji: p,
                        });
                    }
                }
                battles_merge.push(BattleResult { i, j, wij, wji });
            }
        }

        let mut perf = vec![Rating::new(0.0); N];
        eprintln!(
            "Convergence (Non-merged): {:?}",
            PerfCalc::new()
                .epsilon(1e-3)
                .run(&mut perf, &battles)
                .unwrap()
        );

        let mut perf_merge = vec![Rating::new(0.0); N];
        eprintln!(
            "Convergence (Merged): {:?}",
            PerfCalc::new()
                .epsilon(1e-3)
                .run(&mut perf_merge, &battles_merge)
                .unwrap()
        );

        for (p, q) in perf.iter().zip(perf_merge.iter()) {
            assert!((p.value - q.value).abs() < 1e-6);
        }
    }

    #[test]
    fn test_full() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2025);
        const N: usize = 100;
        const B: usize = 100_000;
        const M: u64 = 10;
        let real_perf = (0..N)
            .map(|_| rng.sample(rand_distr::StandardNormal))
            .collect::<Vec<f64>>();
        let mut battles = vec![];
        for b in 0..B {
            let i = b % N;
            let mut j = rng.random_range(0..N - 1);
            if j >= i {
                j += 1;
            }
            let logit = 1.0 / (1.0 + (real_perf[j] - real_perf[i]).exp());
            let wij = rng.sample(rand_distr::Binomial::new(M, logit).unwrap()) as f64;
            let wji = M as f64 - wij;
            battles.push(BattleResult { i, j, wij, wji });
        }

        let mut perf = vec![Rating::new(0.0); N];

        let start = std::time::Instant::now();
        let res = PerfCalc::new()
            .epsilon(1e-4)
            .run(&mut perf, &battles)
            .unwrap();
        let duration = start.elapsed();
        eprintln!("Convergence: {:?}", res);
        eprintln!("Time: {:?}", duration);

        // Calculate MSE between real and estimated performance
        let sqrt_mse = (real_perf
            .iter()
            .zip(perf.iter())
            .map(|(r, p)| (r - p.value).powi(2))
            .sum::<f64>()
            / N as f64)
            .sqrt();
        let mx_diff = real_perf
            .iter()
            .zip(perf.iter())
            .map(|(r, p)| (r - p.value).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        eprintln!("MSE: {}", sqrt_mse);
        eprintln!("Max diff: {}", mx_diff);
        assert!(sqrt_mse < 0.2 && mx_diff < 0.4);
    }
}
