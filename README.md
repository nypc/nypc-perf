# nypc-perf

[![Crates.io](https://img.shields.io/crates/v/nypc-perf.svg)](https://crates.io/crates/nypc-perf)
[![Documentation](https://docs.rs/nypc-perf/badge.svg)](https://docs.rs/nypc-perf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust library for calculating player performance based on battle results using the Bradley-Terry model.

## Overview

This library implements a Bradley-Terry model based performance system that estimates player performance levels from head-to-head battle outcomes. The algorithm uses Newton-Raphson iteration to find maximum likelihood estimates of player performances that best explain the observed win/loss patterns.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
nypc-perf = "0.1.2"
```

## Usage

### Basic Example

```rust
use nypc_perf::{BattleResult, PerfCalc, Rating};

// Create battle results between players
let battles = vec![
    BattleResult {
        i: 0,           // Player 0
        j: 1,           // Player 1
        wij: 2.0,       // Player 0 won twice against Player 1
        wji: 1.0,       // Player 1 won once against Player 0
    },
    BattleResult {
        i: 0,
        j: 2,
        wij: 1.0,
        wji: 0.0,
    }
];

// Initialize performance ratings to 0
let mut perf = vec![
    Rating::new(0.0),       // Variable rating
    Rating::new(0.0),       // Variable rating
    Rating::new_fixed(0.0)  // Fixed anchor rating
];

// Run the rating calculation
let result = PerfCalc::new()
    .max_iters(100)     // Maximum iterations (default: 100)
    .epsilon(1e-6)      // Convergence threshold (default: 1e-6)
    .run(&mut perf, &battles);

// Check if calculation converged
match result {
    Ok(iters) => println!("Converged after {} iterations", iters),
    Err(err) => println!("Did not converge, final error: {}", err)
}

// perf now contains the estimated log-performance ratings
// Higher values indicate better performance
for (i, rating) in perf.iter().enumerate() {
    println!("Player {}: {:.6}", i, rating.value);
}
```

## Authors

**NEXON Algorithm Research Team** - [\_algorithm@nexon.co.kr](mailto:_algorithm@nexon.co.kr)

## Documentation

For detailed explanation of rating calculation, please refer to the [docs/](docs/).

For detailed API documentation, visit [docs.rs/nypc-perf](https://docs.rs/nypc-perf).

## Node.js binding

We provide node.js binding. For this, please visit [@nypc-perf/perf](https://www.npmjs.com/package/@nypc-perf/perf) [![npm version](https://img.shields.io/npm/v/@nypc-perf/perf.svg)](https://www.npmjs.com/package/@nypc-perf/perf)
