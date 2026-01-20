//! Parallelization utilities.
//!
//! This module provides helpers for adaptively choosing between parallel and
//! sequential execution based on workload size, avoiding threading overhead
//! for small computations.

use rayon::prelude::*;

/// Adaptively choose parallel or sequential iteration based on workload size.
///
/// This helper automatically uses Rayon's parallel iterators for large workloads
/// and sequential iteration for small ones to avoid threading overhead.
///
/// # Arguments
/// * `n` - Number of items to process
/// * `threshold` - Minimum size to use parallelization
/// * `f` - Function to map over each index `0..n`
///
/// # Returns
/// A vector of results from applying `f` to each index.
///
/// # Performance
///
/// Threading overhead becomes significant for small workloads. Based on
/// benchmarking, a threshold of 10-100 items is typically optimal, but this
/// depends on the computational cost of `f`.
///
/// # Examples
///
/// ```rust
/// use rsona::utils::parallel::adaptive_map;
///
/// // Process frames adaptively
/// let n_frames = 1000;
/// let results = adaptive_map(n_frames, 10, |i| {
///     // Computation for frame i
///     i * 2
/// });
/// assert_eq!(results.len(), 1000);
/// ```
pub fn adaptive_map<T, F>(n: usize, threshold: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if n > threshold {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Adaptively map with a fixed threshold of 10.
///
/// This is a convenience wrapper for [`adaptive_map`] with the most common
/// threshold value. Use this when you want automatic parallelization without
/// tuning the threshold.
///
/// # Arguments
/// * `n` - Number of items to process
/// * `f` - Function to map over each index `0..n`
///
/// # Examples
///
/// ```rust
/// use rsona::utils::parallel::auto_map;
///
/// let results = auto_map(100, |i| i * i);
/// assert_eq!(results[5], 25);
/// ```
pub fn auto_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    adaptive_map(n, 10, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_map_sequential() {
        let result = adaptive_map(5, 10, |i| i * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_adaptive_map_parallel() {
        let result = adaptive_map(20, 10, |i| i * 2);
        assert_eq!(result.len(), 20);
        assert_eq!(result[10], 20);
    }

    #[test]
    fn test_auto_map() {
        let result = auto_map(15, |i| i + 1);
        assert_eq!(result.len(), 15);
        assert_eq!(result[0], 1);
        assert_eq!(result[14], 15);
    }

    #[test]
    fn test_empty_input() {
        let result = auto_map(0, |i| i);
        assert_eq!(result.len(), 0);
    }
}
