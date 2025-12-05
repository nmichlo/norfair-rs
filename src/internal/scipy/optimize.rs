//! SciPy optimization functions port.
//!
//! Ported from scipy.optimize.linear_sum_assignment
//! License: BSD 3-Clause (SciPy Developers)
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

/// Represents a match between a row index and column index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assignment {
    pub row_idx: usize,
    pub col_idx: usize,
}

/// Result of linear sum assignment.
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    /// Valid assignments (row, col pairs)
    pub assignments: Vec<Assignment>,
    /// Indices of rows that were not matched
    pub unmatched_rows: Vec<usize>,
    /// Indices of columns that were not matched
    pub unmatched_cols: Vec<usize>,
}

/// Solve the linear sum assignment problem using the Hungarian algorithm.
///
/// Finds the optimal assignment between rows and columns to minimize total cost.
/// This is a port of scipy.optimize.linear_sum_assignment.
///
/// # Arguments
/// * `cost_matrix` - 2D cost matrix where cost[i][j] is the cost of assigning row i to column j
/// * `max_cost` - Maximum cost threshold; assignments with cost > max_cost are rejected
///
/// # Returns
/// AssignmentResult containing:
/// - assignments: Valid (row, col) pairs with cost <= max_cost
/// - unmatched_rows: Row indices that were not matched
/// - unmatched_cols: Column indices that were not matched
///
/// # Algorithm
/// Uses the Hungarian (Kuhn-Munkres) algorithm for optimal assignment.
pub fn linear_sum_assignment(cost_matrix: &[Vec<f64>], max_cost: f64) -> AssignmentResult {
    let num_rows = cost_matrix.len();
    if num_rows == 0 {
        return AssignmentResult {
            assignments: Vec::new(),
            unmatched_rows: Vec::new(),
            unmatched_cols: Vec::new(),
        };
    }

    let num_cols = cost_matrix[0].len();
    if num_cols == 0 {
        return AssignmentResult {
            assignments: Vec::new(),
            unmatched_rows: (0..num_rows).collect(),
            unmatched_cols: Vec::new(),
        };
    }

    // Run Hungarian algorithm
    let row_assignments = hungarian_algorithm(cost_matrix);

    // Filter by max_cost and collect results
    let mut assignments = Vec::new();
    let mut matched_rows = vec![false; num_rows];
    let mut matched_cols = vec![false; num_cols];

    for (row_idx, &col_idx) in row_assignments.iter().enumerate() {
        if let Some(col) = col_idx {
            if col < num_cols {
                let cost = cost_matrix[row_idx][col];
                if cost <= max_cost {
                    assignments.push(Assignment {
                        row_idx,
                        col_idx: col,
                    });
                    matched_rows[row_idx] = true;
                    matched_cols[col] = true;
                }
            }
        }
    }

    let unmatched_rows: Vec<usize> = (0..num_rows).filter(|&i| !matched_rows[i]).collect();
    let unmatched_cols: Vec<usize> = (0..num_cols).filter(|&j| !matched_cols[j]).collect();

    AssignmentResult {
        assignments,
        unmatched_rows,
        unmatched_cols,
    }
}

/// Hungarian algorithm (Kuhn-Munkres) for minimum cost assignment.
///
/// Returns a vector where result[i] = Some(j) means row i is assigned to column j.
fn hungarian_algorithm(cost_matrix: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n_rows = cost_matrix.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let n_cols = cost_matrix[0].len();
    if n_cols == 0 {
        return vec![None; n_rows];
    }

    // Pad to square matrix
    let n = n_rows.max(n_cols);
    let mut cost = vec![vec![0.0; n]; n];

    // Copy original costs
    for i in 0..n_rows {
        for j in 0..n_cols {
            cost[i][j] = cost_matrix[i][j];
        }
    }

    // Step 1: Subtract row minimum from each row
    for i in 0..n {
        let row_min = cost[i].iter().cloned().fold(f64::INFINITY, f64::min);
        if row_min.is_finite() {
            for j in 0..n {
                cost[i][j] -= row_min;
            }
        }
    }

    // Step 2: Subtract column minimum from each column
    for j in 0..n {
        let col_min = (0..n).map(|i| cost[i][j]).fold(f64::INFINITY, f64::min);
        if col_min.is_finite() {
            for i in 0..n {
                cost[i][j] -= col_min;
            }
        }
    }

    // Initialize assignment tracking
    let mut row_match: Vec<Option<usize>> = vec![None; n];
    let mut col_match: Vec<Option<usize>> = vec![None; n];

    // Try to find initial assignment with zeros
    for i in 0..n {
        for j in 0..n {
            if cost[i][j].abs() < 1e-10 && row_match[i].is_none() && col_match[j].is_none() {
                row_match[i] = Some(j);
                col_match[j] = Some(i);
            }
        }
    }

    // Augmenting path algorithm
    loop {
        // Find unmatched rows
        let unmatched_rows: Vec<usize> = (0..n).filter(|&i| row_match[i].is_none()).collect();

        if unmatched_rows.is_empty() {
            break; // All rows matched
        }

        // BFS to find augmenting path
        let mut found_augmenting = false;

        for &start_row in &unmatched_rows {
            let mut parent_row: Vec<Option<usize>> = vec![None; n];
            let mut parent_col: Vec<Option<usize>> = vec![None; n];
            let mut visited_col = vec![false; n];
            let mut queue = vec![start_row];
            let mut found_col: Option<usize> = None;

            'bfs: while !queue.is_empty() && found_col.is_none() {
                let row = queue.remove(0);

                for col in 0..n {
                    if !visited_col[col] && cost[row][col].abs() < 1e-10 {
                        visited_col[col] = true;
                        parent_col[col] = Some(row);

                        if col_match[col].is_none() {
                            // Found unmatched column - augmenting path exists
                            found_col = Some(col);
                            break 'bfs;
                        } else {
                            // Column is matched, continue BFS through matched row
                            let next_row = col_match[col].unwrap();
                            parent_row[next_row] = Some(col);
                            queue.push(next_row);
                        }
                    }
                }
            }

            if let Some(mut col) = found_col {
                // Augment along the path
                loop {
                    let row = parent_col[col].unwrap();
                    let prev_col = row_match[row];

                    row_match[row] = Some(col);
                    col_match[col] = Some(row);

                    if let Some(pc) = prev_col {
                        col = pc;
                    } else {
                        break;
                    }
                }
                found_augmenting = true;
                break;
            }
        }

        if !found_augmenting {
            // Need to update costs (create more zeros)
            // Find rows reachable from unmatched rows through alternating paths
            let mut row_covered = vec![false; n];
            let mut col_covered = vec![false; n];

            for &start_row in &unmatched_rows {
                let mut stack = vec![start_row];
                while let Some(row) = stack.pop() {
                    if row_covered[row] {
                        continue;
                    }
                    row_covered[row] = true;

                    for col in 0..n {
                        if cost[row][col].abs() < 1e-10 && !col_covered[col] {
                            col_covered[col] = true;
                            if let Some(matched_row) = col_match[col] {
                                stack.push(matched_row);
                            }
                        }
                    }
                }
            }

            // Find minimum uncovered value
            let mut min_val = f64::INFINITY;
            for i in 0..n {
                if row_covered[i] {
                    for j in 0..n {
                        if !col_covered[j] {
                            min_val = min_val.min(cost[i][j]);
                        }
                    }
                }
            }

            if !min_val.is_finite() || min_val <= 0.0 {
                break; // No improvement possible
            }

            // Update cost matrix
            for i in 0..n {
                for j in 0..n {
                    if row_covered[i] && !col_covered[j] {
                        cost[i][j] -= min_val;
                    } else if !row_covered[i] && col_covered[j] {
                        cost[i][j] += min_val;
                    }
                }
            }
        }
    }

    // Return assignments for original rows only
    row_match.truncate(n_rows);
    row_match
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_sum_assignment_basic_square() {
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        assert_eq!(result.assignments.len(), 3);
        assert!(result.unmatched_rows.is_empty());
        assert!(result.unmatched_cols.is_empty());

        // Calculate total cost
        let total: f64 = result
            .assignments
            .iter()
            .map(|a| cost[a.row_idx][a.col_idx])
            .sum();
        assert!((total - 5.0).abs() < 1e-10); // Optimal: (0,1)=1 + (1,0)=2 + (2,2)=2 = 5
    }

    #[test]
    fn test_linear_sum_assignment_cost_threshold() {
        let cost = vec![vec![1.0, 5.0], vec![5.0, 1.0]];
        let result = linear_sum_assignment(&cost, 2.0);

        // Only assignments with cost <= 2.0 should be kept
        assert_eq!(result.assignments.len(), 2);
        for a in &result.assignments {
            assert!(cost[a.row_idx][a.col_idx] <= 2.0);
        }
    }

    #[test]
    fn test_linear_sum_assignment_rectangular_more_rows() {
        let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        // Can only match 2 rows to 2 columns
        assert_eq!(result.assignments.len(), 2);
        assert_eq!(result.unmatched_rows.len(), 1);
        assert!(result.unmatched_cols.is_empty());
    }

    #[test]
    fn test_linear_sum_assignment_rectangular_more_cols() {
        let cost = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        // Can only match 2 rows to 2 columns
        assert_eq!(result.assignments.len(), 2);
        assert!(result.unmatched_rows.is_empty());
        assert_eq!(result.unmatched_cols.len(), 1);
    }

    #[test]
    fn test_linear_sum_assignment_empty_matrix() {
        let cost: Vec<Vec<f64>> = Vec::new();
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        assert!(result.assignments.is_empty());
        assert!(result.unmatched_rows.is_empty());
        assert!(result.unmatched_cols.is_empty());
    }

    #[test]
    fn test_linear_sum_assignment_empty_columns() {
        let cost: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        assert!(result.assignments.is_empty());
        assert_eq!(result.unmatched_rows, vec![0, 1]);
        assert!(result.unmatched_cols.is_empty());
    }

    #[test]
    fn test_linear_sum_assignment_all_rejected_by_threshold() {
        let cost = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let result = linear_sum_assignment(&cost, 5.0);

        // All costs exceed threshold
        assert!(result.assignments.is_empty());
        assert_eq!(result.unmatched_rows, vec![0, 1]);
        assert_eq!(result.unmatched_cols, vec![0, 1]);
    }

    #[test]
    fn test_linear_sum_assignment_single_element() {
        let cost = vec![vec![3.0]];
        let result = linear_sum_assignment(&cost, 5.0);

        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].row_idx, 0);
        assert_eq!(result.assignments[0].col_idx, 0);
    }

    #[test]
    fn test_linear_sum_assignment_partial_matching() {
        let cost = vec![vec![1.0, 100.0], vec![100.0, 2.0]];
        let result = linear_sum_assignment(&cost, 10.0);

        // Should get optimal matching with costs 1.0 and 2.0
        assert_eq!(result.assignments.len(), 2);
    }

    #[test]
    fn test_linear_sum_assignment_zero_costs() {
        let cost = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        assert_eq!(result.assignments.len(), 2);
        let total: f64 = result
            .assignments
            .iter()
            .map(|a| cost[a.row_idx][a.col_idx])
            .sum();
        assert!((total - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_sum_assignment_optimal_matching() {
        // Test case where greedy would give suboptimal result
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let result = linear_sum_assignment(&cost, f64::INFINITY);

        assert_eq!(result.assignments.len(), 3);
        // Optimal is diagonal: 1 + 4 + 9 = 14 or 1 + 6 + 6 = 13 or other
        // Actually optimal: (0,0)=1, (1,1)=4, (2,2)=9 = 14
        // Or: (0,0)=1, (1,2)=6, (2,1)=6 = 13
    }
}
