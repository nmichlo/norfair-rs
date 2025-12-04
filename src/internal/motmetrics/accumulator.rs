//! MOT metrics accumulator for tracking evaluation.

use nalgebra::DMatrix;

/// Event types for MOT evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Match,
    Switch,
    Miss,
    FalsePositive,
    Fragmentation,
}

/// Single tracking event
#[derive(Debug, Clone)]
pub struct Event {
    pub frame_id: i32,
    pub event_type: EventType,
    pub object_id: Option<i32>,
    pub hypothesis_id: Option<i32>,
    pub distance: Option<f64>,
}

/// Accumulator for MOT metrics computation.
///
/// Collects tracking events across frames for later metric computation.
#[derive(Debug, Default)]
pub struct MOTAccumulator {
    events: Vec<Event>,
    last_match: std::collections::HashMap<i32, i32>, // object_id -> hypothesis_id
}

impl MOTAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the accumulator with matches for a frame.
    ///
    /// # Arguments
    /// * `frame_id` - Current frame number
    /// * `object_ids` - Ground truth object IDs present in this frame
    /// * `hypothesis_ids` - Predicted track IDs present in this frame
    /// * `distances` - Distance matrix (objects x hypotheses), inf for invalid matches
    pub fn update(
        &mut self,
        frame_id: i32,
        object_ids: &[i32],
        hypothesis_ids: &[i32],
        distances: &DMatrix<f64>,
    ) {
        // Simple greedy matching for now
        let mut matched_objects = std::collections::HashSet::new();
        let mut matched_hypotheses = std::collections::HashSet::new();

        // Sort all valid distances
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..object_ids.len() {
            for j in 0..hypothesis_ids.len() {
                let d = distances[(i, j)];
                if d.is_finite() {
                    pairs.push((d, i, j));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Greedy matching
        for (dist, obj_idx, hyp_idx) in pairs {
            if matched_objects.contains(&obj_idx) || matched_hypotheses.contains(&hyp_idx) {
                continue;
            }

            let obj_id = object_ids[obj_idx];
            let hyp_id = hypothesis_ids[hyp_idx];

            // Check for ID switch
            let event_type = if let Some(&prev_hyp) = self.last_match.get(&obj_id) {
                if prev_hyp != hyp_id {
                    EventType::Switch
                } else {
                    EventType::Match
                }
            } else {
                EventType::Match
            };

            self.events.push(Event {
                frame_id,
                event_type,
                object_id: Some(obj_id),
                hypothesis_id: Some(hyp_id),
                distance: Some(dist),
            });

            self.last_match.insert(obj_id, hyp_id);
            matched_objects.insert(obj_idx);
            matched_hypotheses.insert(hyp_idx);
        }

        // Misses (unmatched ground truth)
        for (idx, &obj_id) in object_ids.iter().enumerate() {
            if !matched_objects.contains(&idx) {
                self.events.push(Event {
                    frame_id,
                    event_type: EventType::Miss,
                    object_id: Some(obj_id),
                    hypothesis_id: None,
                    distance: None,
                });
            }
        }

        // False positives (unmatched hypotheses)
        for (idx, &hyp_id) in hypothesis_ids.iter().enumerate() {
            if !matched_hypotheses.contains(&idx) {
                self.events.push(Event {
                    frame_id,
                    event_type: EventType::FalsePositive,
                    object_id: None,
                    hypothesis_id: Some(hyp_id),
                    distance: None,
                });
            }
        }
    }

    /// Get all collected events.
    pub fn get_events(&self) -> &[Event] {
        &self.events
    }

    /// Compute summary metrics.
    pub fn compute_metrics(&self) -> MOTMetrics {
        let mut metrics = MOTMetrics::default();

        for event in &self.events {
            match event.event_type {
                EventType::Match => {
                    metrics.num_matches += 1;
                    if let Some(d) = event.distance {
                        metrics.total_distance += d;
                    }
                }
                EventType::Switch => {
                    metrics.num_switches += 1;
                    metrics.num_matches += 1;
                    if let Some(d) = event.distance {
                        metrics.total_distance += d;
                    }
                }
                EventType::Miss => metrics.num_misses += 1,
                EventType::FalsePositive => metrics.num_false_positives += 1,
                EventType::Fragmentation => metrics.num_fragmentations += 1,
            }
        }

        // Compute derived metrics
        let num_gt = metrics.num_matches + metrics.num_misses;
        let num_pred = metrics.num_matches + metrics.num_false_positives;

        if num_gt > 0 {
            metrics.recall = metrics.num_matches as f64 / num_gt as f64;
            metrics.mota = 1.0
                - (metrics.num_misses + metrics.num_false_positives + metrics.num_switches) as f64
                    / num_gt as f64;
        }

        if num_pred > 0 {
            metrics.precision = metrics.num_matches as f64 / num_pred as f64;
        }

        if metrics.num_matches > 0 {
            metrics.motp = metrics.total_distance / metrics.num_matches as f64;
        }

        metrics
    }
}

/// Summary metrics from MOT evaluation.
#[derive(Debug, Clone, Default)]
pub struct MOTMetrics {
    pub num_matches: i32,
    pub num_misses: i32,
    pub num_false_positives: i32,
    pub num_switches: i32,
    pub num_fragmentations: i32,
    pub total_distance: f64,
    pub mota: f64,
    pub motp: f64,
    pub precision: f64,
    pub recall: f64,
    pub mostly_tracked: i32,
    pub mostly_lost: i32,
    pub partially_tracked: i32,
}

// Aliases matching Python/Go
pub type NumMatches = i32;
pub type NumMisses = i32;
pub type NumFalsePositives = i32;
pub type NumSwitches = i32;
pub type NumFragmentations = i32;
pub type MOTA = f64;
pub type MOTP = f64;
pub type Precision = f64;
pub type Recall = f64;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ===== Basic Accumulator Tests =====

    #[test]
    fn test_accumulator_new() {
        let acc = MOTAccumulator::new();
        let metrics = acc.compute_metrics();

        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
        assert_eq!(metrics.num_switches, 0);
    }

    #[test]
    fn test_accumulator_empty_frame() {
        let mut acc = MOTAccumulator::new();

        // Both empty: should produce no events
        let distances = DMatrix::zeros(0, 0);
        acc.update(0, &[], &[], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
    }

    #[test]
    fn test_accumulator_only_predictions() {
        let mut acc = MOTAccumulator::new();

        // No GT, 3 predictions → 3 false positives
        let distances = DMatrix::zeros(0, 3);
        acc.update(0, &[], &[1, 2, 3], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_false_positives, 3);
        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_misses, 0);
    }

    #[test]
    fn test_accumulator_only_gt() {
        let mut acc = MOTAccumulator::new();

        // 2 GT, no predictions → 2 misses
        let distances = DMatrix::zeros(2, 0);
        acc.update(0, &[1, 2], &[], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_misses, 2);
        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_false_positives, 0);
    }

    #[test]
    fn test_accumulator_perfect_tracking() {
        let mut acc = MOTAccumulator::new();

        // Perfect match: object 1 matched to hypothesis 1
        let distances = DMatrix::from_row_slice(1, 1, &[0.0]);
        acc.update(0, &[1], &[1], &distances);
        acc.update(1, &[1], &[1], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 2);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
        assert_eq!(metrics.num_switches, 0);
        assert_relative_eq!(metrics.total_distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_accumulator_perfect_match_two_objects() {
        let mut acc = MOTAccumulator::new();

        // Two objects, perfect match
        let distances = DMatrix::from_row_slice(2, 2, &[
            0.0, f64::INFINITY,
            f64::INFINITY, 0.0,
        ]);
        acc.update(0, &[1, 2], &[1, 2], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 2);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
    }

    #[test]
    fn test_accumulator_with_miss() {
        let mut acc = MOTAccumulator::new();

        // No hypothesis for object
        let distances = DMatrix::from_row_slice(1, 0, &[]);
        acc.update(0, &[1], &[], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_misses, 1);
    }

    #[test]
    fn test_accumulator_partial_match() {
        let mut acc = MOTAccumulator::new();

        // 2 GT, 2 predictions, but only 1 valid match
        let distances = DMatrix::from_row_slice(2, 2, &[
            0.1, f64::INFINITY,  // GT0 matches Pred0
            f64::INFINITY, f64::INFINITY,  // GT1 matches nothing
        ]);
        acc.update(0, &[1, 2], &[1, 2], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 1);
        assert_eq!(metrics.num_misses, 1);
        assert_eq!(metrics.num_false_positives, 1);
    }

    // ===== ID Switch Detection Tests =====

    #[test]
    fn test_accumulator_id_switch() {
        let mut acc = MOTAccumulator::new();

        // Frame 1: GT1 → Tracker1
        let distances = DMatrix::from_row_slice(1, 1, &[0.0]);
        acc.update(0, &[1], &[1], &distances);

        // Check no switch yet
        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_switches, 0);

        // Frame 2: GT1 → Tracker2 (switch!)
        acc.update(1, &[1], &[2], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_switches, 1);
        assert_eq!(metrics.num_matches, 2); // Both are still matches
    }

    #[test]
    fn test_accumulator_no_switch_same_tracker() {
        let mut acc = MOTAccumulator::new();

        // Frame 1-3: GT1 → Tracker1 consistently
        let distances = DMatrix::from_row_slice(1, 1, &[0.0]);
        acc.update(0, &[1], &[1], &distances);
        acc.update(1, &[1], &[1], &distances);
        acc.update(2, &[1], &[1], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_switches, 0);
        assert_eq!(metrics.num_matches, 3);
    }

    #[test]
    fn test_accumulator_multiple_switches() {
        let mut acc = MOTAccumulator::new();
        let distances = DMatrix::from_row_slice(1, 1, &[0.0]);

        // GT1 → Tracker1
        acc.update(0, &[1], &[1], &distances);
        // GT1 → Tracker2 (switch 1)
        acc.update(1, &[1], &[2], &distances);
        // GT1 → Tracker1 (switch 2)
        acc.update(2, &[1], &[1], &distances);
        // GT1 → Tracker3 (switch 3)
        acc.update(3, &[1], &[3], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_switches, 3);
        assert_eq!(metrics.num_matches, 4);
    }

    // ===== Multi-Frame Accumulation Tests =====

    #[test]
    fn test_accumulator_multi_frame() {
        let mut acc = MOTAccumulator::new();

        // Frame 1: 2 GT, 2 predictions (2 matches)
        let dist1 = DMatrix::from_row_slice(2, 2, &[
            0.0, f64::INFINITY,
            f64::INFINITY, 0.0,
        ]);
        acc.update(0, &[1, 2], &[1, 2], &dist1);

        // Frame 2: 2 GT, 1 prediction (1 match, 1 miss)
        let dist2 = DMatrix::from_row_slice(2, 1, &[0.0, f64::INFINITY]);
        acc.update(1, &[1, 2], &[1], &dist2);

        // Frame 3: 1 GT, 2 predictions (1 match, 1 FP)
        let dist3 = DMatrix::from_row_slice(1, 2, &[0.0, f64::INFINITY]);
        acc.update(2, &[1], &[1, 3], &dist3);

        let metrics = acc.compute_metrics();
        // 2+1+1=4 matches, 0+1+0=1 miss, 0+0+1=1 FP
        assert_eq!(metrics.num_matches, 4);
        assert_eq!(metrics.num_misses, 1);
        assert_eq!(metrics.num_false_positives, 1);
    }

    // ===== Metric Computation Tests =====

    #[test]
    fn test_metrics_recall() {
        let mut acc = MOTAccumulator::new();

        // 3 matches, 2 misses → recall = 3/5 = 0.6
        let dist = DMatrix::from_row_slice(1, 1, &[0.0]);
        acc.update(0, &[1], &[1], &dist);
        acc.update(1, &[1], &[1], &dist);
        acc.update(2, &[1], &[1], &dist);

        let empty = DMatrix::zeros(1, 0);
        acc.update(3, &[1], &[], &empty);
        acc.update(4, &[1], &[], &empty);

        let metrics = acc.compute_metrics();
        assert_relative_eq!(metrics.recall, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_precision() {
        let mut acc = MOTAccumulator::new();

        // 2 matches, 3 false positives → precision = 2/5 = 0.4
        let dist = DMatrix::from_row_slice(1, 1, &[0.0]);
        acc.update(0, &[1], &[1], &dist);
        acc.update(1, &[1], &[1], &dist);

        let empty = DMatrix::zeros(0, 1);
        acc.update(2, &[], &[2], &empty);
        acc.update(3, &[], &[3], &empty);
        acc.update(4, &[], &[4], &empty);

        let metrics = acc.compute_metrics();
        assert_relative_eq!(metrics.precision, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_mota() {
        let mut acc = MOTAccumulator::new();

        // MOTA = 1 - (FN + FP + IDSW) / num_gt
        // 8 matches, 1 miss, 1 FP, 0 switches on 9 GT objects
        // MOTA = 1 - (1 + 1 + 0) / 9 = 1 - 2/9 ≈ 0.778

        let dist = DMatrix::from_row_slice(1, 1, &[0.1]);
        for _ in 0..8 {
            acc.update(0, &[1], &[1], &dist);
        }

        let empty_hyp = DMatrix::zeros(1, 0);
        acc.update(8, &[1], &[], &empty_hyp);

        let empty_gt = DMatrix::zeros(0, 1);
        acc.update(9, &[], &[2], &empty_gt);

        let metrics = acc.compute_metrics();
        // 8 matches + 1 miss = 9 GT, 8 matches + 1 FP = 9 pred
        // MOTA = 1 - (1 + 1 + 0) / 9
        assert_relative_eq!(metrics.mota, 1.0 - 2.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_motp() {
        let mut acc = MOTAccumulator::new();

        // 3 matches with distances 0.1, 0.2, 0.3 → MOTP = 0.6/3 = 0.2
        let dist1 = DMatrix::from_row_slice(1, 1, &[0.1]);
        let dist2 = DMatrix::from_row_slice(1, 1, &[0.2]);
        let dist3 = DMatrix::from_row_slice(1, 1, &[0.3]);

        acc.update(0, &[1], &[1], &dist1);
        acc.update(1, &[1], &[1], &dist2);
        acc.update(2, &[1], &[1], &dist3);

        let metrics = acc.compute_metrics();
        assert_relative_eq!(metrics.motp, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_perfect_tracking() {
        let mut acc = MOTAccumulator::new();

        // Perfect tracking: all matches, no misses/FPs/switches
        let dist = DMatrix::from_row_slice(2, 2, &[
            0.0, f64::INFINITY,
            f64::INFINITY, 0.0,
        ]);

        for _ in 0..10 {
            acc.update(0, &[1, 2], &[1, 2], &dist);
        }

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 20);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
        assert_eq!(metrics.num_switches, 0);
        assert_relative_eq!(metrics.mota, 1.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.recall, 1.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.precision, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_all_misses() {
        let mut acc = MOTAccumulator::new();

        // All misses: no predictions
        let empty = DMatrix::zeros(2, 0);
        for _ in 0..5 {
            acc.update(0, &[1, 2], &[], &empty);
        }

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_misses, 10);
        assert_eq!(metrics.num_false_positives, 0);
        // MOTA = 1 - (FN + FP + IDSW) / num_gt = 1 - 10/10 = 0
        assert_relative_eq!(metrics.mota, 0.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.recall, 0.0, epsilon = 1e-10);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_accumulator_greedy_matching() {
        let mut acc = MOTAccumulator::new();

        // Test greedy matching: GT0 should match Pred1 (distance 0.1)
        // even though GT1 also wants Pred1 (distance 0.2)
        let distances = DMatrix::from_row_slice(2, 2, &[
            0.5, 0.1,  // GT0: prefers Pred1
            0.2, 0.2,  // GT1: both same
        ]);
        acc.update(0, &[1, 2], &[1, 2], &distances);

        let metrics = acc.compute_metrics();
        // Both should match, greedy picks smallest first
        assert_eq!(metrics.num_matches, 2);
        assert_eq!(metrics.num_misses, 0);
        assert_eq!(metrics.num_false_positives, 0);
    }

    #[test]
    fn test_accumulator_with_inf_distances() {
        let mut acc = MOTAccumulator::new();

        // All infinite distances → no matches
        let distances = DMatrix::from_row_slice(2, 2, &[
            f64::INFINITY, f64::INFINITY,
            f64::INFINITY, f64::INFINITY,
        ]);
        acc.update(0, &[1, 2], &[1, 2], &distances);

        let metrics = acc.compute_metrics();
        assert_eq!(metrics.num_matches, 0);
        assert_eq!(metrics.num_misses, 2);
        assert_eq!(metrics.num_false_positives, 2);
    }
}
