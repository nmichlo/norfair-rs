//! MOT metrics accumulator for tracking evaluation.
//!
//! Ported from py-motmetrics.
//! License: MIT (Christoph Heindl, Jack Valmadre)

use nalgebra::DMatrix;
use std::collections::HashMap;

/// Track lifecycle for computing MT/ML/PT metrics.
///
/// Tracks the lifecycle of a ground truth object across frames.
#[derive(Debug, Clone)]
pub struct TrackLifecycle {
    /// Ground truth object ID
    pub gt_id: i32,
    /// First frame where this object appears
    pub first_frame: i32,
    /// Last frame where this object appears
    pub last_frame: i32,
    /// Number of frames where the object was tracked (matched)
    pub tracked_frames: i32,
    /// Number of frames where the object was detected (present in GT)
    pub detected_frames: i32,
    /// Number of fragmentations (miss → match transitions)
    pub fragmentations: i32,
    /// Was the object matched in the previous frame?
    pub was_matched: bool,
}

impl TrackLifecycle {
    /// Create a new track lifecycle.
    pub fn new(gt_id: i32, first_frame: i32) -> Self {
        Self {
            gt_id,
            first_frame,
            last_frame: first_frame,
            tracked_frames: 0,
            detected_frames: 0,
            fragmentations: 0,
            was_matched: false,
        }
    }

    /// Update lifecycle when the object is matched in a frame.
    pub fn update_matched(&mut self, frame: i32) {
        self.last_frame = frame;
        self.tracked_frames += 1;
        self.detected_frames += 1;

        // Fragmentation: transition from miss to match
        if self.detected_frames > 1 && !self.was_matched {
            self.fragmentations += 1;
        }

        self.was_matched = true;
    }

    /// Update lifecycle when the object is missed in a frame.
    pub fn update_missed(&mut self, frame: i32) {
        self.last_frame = frame;
        self.detected_frames += 1;
        self.was_matched = false;
    }

    /// Calculate coverage ratio (tracked_frames / detected_frames).
    pub fn coverage(&self) -> f64 {
        if self.detected_frames == 0 {
            0.0
        } else {
            self.tracked_frames as f64 / self.detected_frames as f64
        }
    }
}

/// Extended MOT accumulator matching Go's API.
///
/// Uses a Hungarian assignment callback for flexible matching.
#[derive(Debug)]
pub struct ExtendedMOTAccumulator {
    /// Video name
    pub video_name: String,
    /// Current frame ID
    pub frame_id: i32,
    /// Number of matches
    pub num_matches: i32,
    /// Number of misses
    pub num_misses: i32,
    /// Number of false positives
    pub num_false_positives: i32,
    /// Number of ID switches
    pub num_switches: i32,
    /// Number of GT objects seen
    pub num_objects: i32,
    /// Total distance for MOTP
    pub total_distance: f64,
    /// Previous frame's GT-to-tracker mapping
    pub previous_mapping: HashMap<i32, i32>,
    /// Track lifecycles for MT/ML/PT metrics
    pub track_lifecycles: HashMap<i32, TrackLifecycle>,
}

impl ExtendedMOTAccumulator {
    /// Create a new accumulator.
    pub fn new(video_name: &str) -> Self {
        Self {
            video_name: video_name.to_string(),
            frame_id: 0,
            num_matches: 0,
            num_misses: 0,
            num_false_positives: 0,
            num_switches: 0,
            num_objects: 0,
            total_distance: 0.0,
            previous_mapping: HashMap::new(),
            track_lifecycles: HashMap::new(),
        }
    }

    /// Update the accumulator with a frame's data.
    ///
    /// # Arguments
    /// * `gt_bboxes` - Ground truth bounding boxes
    /// * `gt_ids` - Ground truth object IDs
    /// * `pred_bboxes` - Predicted bounding boxes
    /// * `pred_ids` - Predicted track IDs
    /// * `threshold` - Maximum distance threshold
    /// * `hungarian_fn` - Assignment function returning (matches, unmatched_gt, unmatched_pred)
    pub fn update<F>(
        &mut self,
        gt_bboxes: &[Vec<f64>],
        gt_ids: &[i32],
        pred_bboxes: &[Vec<f64>],
        pred_ids: &[i32],
        threshold: f64,
        hungarian_fn: F,
    ) where
        F: Fn(&[Vec<f64>], f64) -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>),
    {
        self.frame_id += 1;

        // Edge case: no GT, no predictions
        if gt_bboxes.is_empty() && pred_bboxes.is_empty() {
            return;
        }

        // Edge case: no GT, only predictions → all false positives
        if gt_bboxes.is_empty() {
            self.num_false_positives += pred_bboxes.len() as i32;
            return;
        }

        // Edge case: no predictions, only GT → all misses
        if pred_bboxes.is_empty() {
            self.num_misses += gt_bboxes.len() as i32;
            self.num_objects += gt_bboxes.len() as i32;

            // Update lifecycles: all GT objects are missed
            for &gt_id in gt_ids {
                let lifecycle = self
                    .track_lifecycles
                    .entry(gt_id)
                    .or_insert_with(|| TrackLifecycle::new(gt_id, self.frame_id));
                lifecycle.update_missed(self.frame_id);
            }
            return;
        }

        self.num_objects += gt_ids.len() as i32;

        // Compute IoU distance matrix
        let distances = super::iou::compute_iou_distance_matrix(gt_bboxes, pred_bboxes);

        // Get assignment
        let (matches, unmatched_gt, unmatched_pred) = hungarian_fn(&distances, threshold);

        // Accumulate events
        self.num_matches += matches.len() as i32;
        self.num_misses += unmatched_gt.len() as i32;
        self.num_false_positives += unmatched_pred.len() as i32;

        // Accumulate distances for MOTP
        for &[gt_idx, pred_idx] in &matches {
            self.total_distance += distances[gt_idx][pred_idx];
        }

        // Update lifecycles for matched GT objects
        for &[gt_idx, pred_idx] in &matches {
            let gt_id = gt_ids[gt_idx];
            let pred_id = pred_ids[pred_idx];

            let lifecycle = self
                .track_lifecycles
                .entry(gt_id)
                .or_insert_with(|| TrackLifecycle::new(gt_id, self.frame_id));
            lifecycle.update_matched(self.frame_id);

            // Check for ID switch
            if let Some(&prev_pred_id) = self.previous_mapping.get(&gt_id) {
                if prev_pred_id != pred_id {
                    self.num_switches += 1;
                }
            }

            self.previous_mapping.insert(gt_id, pred_id);
        }

        // Update lifecycles for missed GT objects
        for &gt_idx in &unmatched_gt {
            let gt_id = gt_ids[gt_idx];

            let lifecycle = self
                .track_lifecycles
                .entry(gt_id)
                .or_insert_with(|| TrackLifecycle::new(gt_id, self.frame_id));
            lifecycle.update_missed(self.frame_id);
        }
    }

    /// Compute extended metrics (MT, ML, PT, total fragmentations).
    pub fn compute_extended_metrics(&self) -> (i32, i32, i32, i32) {
        let mut mostly_tracked = 0;
        let mut mostly_lost = 0;
        let mut partially_tracked = 0;
        let mut total_fragmentations = 0;

        for lifecycle in self.track_lifecycles.values() {
            let coverage = lifecycle.coverage();
            total_fragmentations += lifecycle.fragmentations;

            if coverage >= 0.8 {
                mostly_tracked += 1;
            } else if coverage < 0.2 {
                mostly_lost += 1;
            } else {
                partially_tracked += 1;
            }
        }

        (mostly_tracked, mostly_lost, partially_tracked, total_fragmentations)
    }
}

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

    // ===== TrackLifecycle Tests (ported from Go) =====

    #[test]
    fn test_new_track_lifecycle() {
        let lifecycle = TrackLifecycle::new(42, 10);

        assert_eq!(lifecycle.gt_id, 42);
        assert_eq!(lifecycle.first_frame, 10);
        assert_eq!(lifecycle.last_frame, 10);
        assert_eq!(lifecycle.tracked_frames, 0);
        assert_eq!(lifecycle.detected_frames, 0);
        assert_eq!(lifecycle.fragmentations, 0);
        assert!(!lifecycle.was_matched);
    }

    #[test]
    fn test_track_lifecycle_update_matched() {
        let mut lifecycle = TrackLifecycle::new(1, 1);

        // Frame 1: First match
        lifecycle.update_matched(1);
        assert_eq!(lifecycle.tracked_frames, 1);
        assert_eq!(lifecycle.detected_frames, 1);
        assert_eq!(lifecycle.fragmentations, 0, "Expected no fragmentation on first match");
        assert!(lifecycle.was_matched);

        // Frame 2: Consecutive match
        lifecycle.update_matched(2);
        assert_eq!(lifecycle.tracked_frames, 2);
        assert_eq!(lifecycle.fragmentations, 0, "Expected no fragmentation on consecutive match");
    }

    #[test]
    fn test_track_lifecycle_update_missed() {
        let mut lifecycle = TrackLifecycle::new(1, 1);

        lifecycle.update_missed(1);
        assert_eq!(lifecycle.tracked_frames, 0, "Expected TrackedFrames=0 after miss");
        assert_eq!(lifecycle.detected_frames, 1);
        assert!(!lifecycle.was_matched, "Expected WasMatched=false after miss");
    }

    #[test]
    fn test_track_lifecycle_fragmentation() {
        let mut lifecycle = TrackLifecycle::new(1, 1);

        // Frame 1: Match
        lifecycle.update_matched(1);
        assert_eq!(lifecycle.fragmentations, 0, "Frame 1: Expected no fragmentation");

        // Frame 2: Miss (track break)
        lifecycle.update_missed(2);
        assert_eq!(lifecycle.fragmentations, 0, "Frame 2: Expected no fragmentation on miss");

        // Frame 3: Match (fragmentation: miss → match)
        lifecycle.update_matched(3);
        assert_eq!(lifecycle.fragmentations, 1, "Frame 3: Expected 1 fragmentation (miss → match)");

        // Frame 4: Miss
        lifecycle.update_missed(4);

        // Frame 5: Match (second fragmentation)
        lifecycle.update_matched(5);
        assert_eq!(lifecycle.fragmentations, 2, "Frame 5: Expected 2 fragmentations");
    }

    #[test]
    fn test_track_lifecycle_coverage() {
        let mut lifecycle = TrackLifecycle::new(1, 1);

        // No detections: coverage = 0
        let coverage = lifecycle.coverage();
        assert_relative_eq!(coverage, 0.0, epsilon = 1e-10);

        // 3 tracked out of 5 detected: coverage = 0.6
        lifecycle.update_matched(1); // Tracked
        lifecycle.update_matched(2); // Tracked
        lifecycle.update_missed(3);  // Missed
        lifecycle.update_matched(4); // Tracked
        lifecycle.update_missed(5);  // Missed

        let coverage = lifecycle.coverage();
        assert_relative_eq!(coverage, 0.6, epsilon = 1e-10);
    }

    // ===== ExtendedMOTAccumulator Tests (ported from Go) =====

    #[test]
    fn test_new_extended_mot_accumulator() {
        let acc = ExtendedMOTAccumulator::new("video1");

        assert_eq!(acc.video_name, "video1");
        assert_eq!(acc.frame_id, 0);
        assert_eq!(acc.num_matches, 0);
        assert!(acc.previous_mapping.is_empty());
        assert!(acc.track_lifecycles.is_empty());
    }

    /// Mock Hungarian that returns no matches (all unmatched)
    fn mock_hungarian_no_matches(distances: &[Vec<f64>], _threshold: f64) -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
        let num_gt = distances.len();
        let num_pred = if num_gt > 0 { distances[0].len() } else { 0 };

        let unmatched_gt: Vec<usize> = (0..num_gt).collect();
        let unmatched_pred: Vec<usize> = (0..num_pred).collect();

        (vec![], unmatched_gt, unmatched_pred)
    }

    #[test]
    fn test_extended_accumulator_update_empty_frame() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Both empty: should increment frame but no events
        acc.update(&[], &[], &[], &[], 0.5, mock_hungarian_no_matches);

        assert_eq!(acc.frame_id, 1);
        assert_eq!(acc.num_matches, 0);
        assert_eq!(acc.num_misses, 0);
        assert_eq!(acc.num_false_positives, 0);
    }

    #[test]
    fn test_extended_accumulator_update_only_predictions() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // No GT, 3 predictions → 3 false positives
        let pred_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],
            vec![20.0, 20.0, 30.0, 30.0],
            vec![40.0, 40.0, 50.0, 50.0],
        ];
        let pred_ids = vec![1, 2, 3];

        acc.update(&[], &[], &pred_bboxes, &pred_ids, 0.5, mock_hungarian_no_matches);

        assert_eq!(acc.num_false_positives, 3);
        assert_eq!(acc.num_matches, 0);
        assert_eq!(acc.num_misses, 0);
    }

    #[test]
    fn test_extended_accumulator_update_only_gt() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // 2 GT, no predictions → 2 misses
        let gt_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],
            vec![20.0, 20.0, 30.0, 30.0],
        ];
        let gt_ids = vec![1, 2];

        acc.update(&gt_bboxes, &gt_ids, &[], &[], 0.5, mock_hungarian_no_matches);

        assert_eq!(acc.num_misses, 2);
        assert_eq!(acc.num_objects, 2);

        // Verify lifecycles were created and updated
        assert_eq!(acc.track_lifecycles.len(), 2);
        for &gt_id in &gt_ids {
            let lifecycle = acc.track_lifecycles.get(&gt_id).unwrap();
            assert_eq!(lifecycle.detected_frames, 1, "GT {}: Expected 1 detected frame", gt_id);
            assert_eq!(lifecycle.tracked_frames, 0, "GT {}: Expected 0 tracked frames", gt_id);
        }
    }

    #[test]
    fn test_extended_accumulator_update_perfect_match() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Perfect match: same GT and predictions
        let boxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],
            vec![20.0, 20.0, 30.0, 30.0],
        ];
        let ids = vec![1, 2];

        // Mock Hungarian returns all matches
        let hungarian_fn = |_distances: &[Vec<f64>], _threshold: f64| -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
            (vec![[0, 0], [1, 1]], vec![], vec![])
        };

        acc.update(&boxes, &ids, &boxes, &ids, 0.5, hungarian_fn);

        assert_eq!(acc.num_matches, 2);
        assert_eq!(acc.num_misses, 0);
        assert_eq!(acc.num_false_positives, 0);

        // TotalDistance should be 0 (perfect overlap)
        assert_relative_eq!(acc.total_distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extended_accumulator_update_partial_match() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        let gt_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],   // Will match
            vec![20.0, 20.0, 30.0, 30.0], // Will miss
        ];
        let gt_ids = vec![1, 2];

        let pred_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],   // Matches GT 0
            vec![50.0, 50.0, 60.0, 60.0], // False positive
        ];
        let pred_ids = vec![1, 2];

        // Mock Hungarian: GT0↔Pred0 match, GT1 unmatched, Pred1 unmatched
        let hungarian_fn = |_distances: &[Vec<f64>], _threshold: f64| -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
            (vec![[0, 0]], vec![1], vec![1])
        };

        acc.update(&gt_bboxes, &gt_ids, &pred_bboxes, &pred_ids, 0.5, hungarian_fn);

        assert_eq!(acc.num_matches, 1);
        assert_eq!(acc.num_misses, 1);
        assert_eq!(acc.num_false_positives, 1);
        assert_eq!(acc.num_objects, 2);
    }

    #[test]
    fn test_extended_accumulator_detect_switches() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        let boxes = vec![vec![0.0, 0.0, 10.0, 10.0]];

        // Frame 1: GT1 → Tracker1
        let hungarian_fn = |_distances: &[Vec<f64>], _threshold: f64| -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
            (vec![[0, 0]], vec![], vec![])
        };
        acc.update(&boxes, &[1], &boxes, &[1], 0.5, hungarian_fn);

        assert_eq!(acc.num_switches, 0, "Frame 1: Expected 0 switches");

        // Frame 2: GT1 → Tracker2 (switch!)
        acc.update(&boxes, &[1], &boxes, &[2], 0.5, hungarian_fn);

        assert_eq!(acc.num_switches, 1, "Frame 2: Expected 1 switch");

        // Frame 3: GT1 → Tracker2 (no switch, same as previous)
        acc.update(&boxes, &[1], &boxes, &[2], 0.5, hungarian_fn);

        assert_eq!(acc.num_switches, 1, "Frame 3: Expected 1 switch total");
    }

    #[test]
    fn test_extended_accumulator_multi_frame() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Perfect Hungarian matcher
        let hungarian_fn = |distances: &[Vec<f64>], _threshold: f64| -> (Vec<[usize; 2]>, Vec<usize>, Vec<usize>) {
            let num_gt = distances.len();
            let num_pred = if num_gt > 0 { distances[0].len() } else { 0 };
            let num_matches = num_gt.min(num_pred);

            let matches: Vec<[usize; 2]> = (0..num_matches).map(|i| [i, i]).collect();
            let unmatched_gt: Vec<usize> = (num_matches..num_gt).collect();
            let unmatched_pred: Vec<usize> = (num_matches..num_pred).collect();

            (matches, unmatched_gt, unmatched_pred)
        };

        // Frame 1: 2 GT, 2 predictions (2 matches)
        acc.update(
            &[vec![0.0, 0.0, 10.0, 10.0], vec![20.0, 20.0, 30.0, 30.0]],
            &[1, 2],
            &[vec![0.0, 0.0, 10.0, 10.0], vec![20.0, 20.0, 30.0, 30.0]],
            &[1, 2],
            0.5,
            hungarian_fn,
        );

        // Frame 2: 2 GT, 1 prediction (1 match, 1 miss)
        acc.update(
            &[vec![0.0, 0.0, 10.0, 10.0], vec![20.0, 20.0, 30.0, 30.0]],
            &[1, 2],
            &[vec![0.0, 0.0, 10.0, 10.0]],
            &[1],
            0.5,
            hungarian_fn,
        );

        // Frame 3: 1 GT, 2 predictions (1 match, 1 FP)
        acc.update(
            &[vec![0.0, 0.0, 10.0, 10.0]],
            &[1],
            &[vec![0.0, 0.0, 10.0, 10.0], vec![50.0, 50.0, 60.0, 60.0]],
            &[1, 3],
            0.5,
            hungarian_fn,
        );

        // Verify totals: 2+1+1=4 matches, 0+1+0=1 miss, 0+0+1=1 FP
        assert_eq!(acc.num_matches, 4, "Expected 4 total matches");
        assert_eq!(acc.num_misses, 1, "Expected 1 total miss");
        assert_eq!(acc.num_false_positives, 1, "Expected 1 total FP");
        assert_eq!(acc.num_objects, 5, "Expected 5 total objects (2+2+1)");
    }

    // ===== Extended Metrics Tests (ported from Go) =====

    #[test]
    fn test_compute_extended_metrics_mostly_tracked() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Create lifecycle with 80% coverage (MT threshold)
        let mut lifecycle1 = TrackLifecycle::new(1, 1);
        for i in 0..8 {
            lifecycle1.update_matched(i + 1);
        }
        for i in 0..2 {
            lifecycle1.update_missed(i + 9);
        }

        // Create lifecycle with 100% coverage
        let mut lifecycle2 = TrackLifecycle::new(2, 1);
        for i in 0..10 {
            lifecycle2.update_matched(i + 1);
        }

        acc.track_lifecycles.insert(1, lifecycle1);
        acc.track_lifecycles.insert(2, lifecycle2);

        let (mt, ml, pt, _) = acc.compute_extended_metrics();

        assert_eq!(mt, 2, "Expected 2 MT tracks");
        assert_eq!(ml, 0, "Expected 0 ML tracks");
        assert_eq!(pt, 0, "Expected 0 PT tracks");
    }

    #[test]
    fn test_compute_extended_metrics_mostly_lost() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Create lifecycle with 0% coverage
        let mut lifecycle1 = TrackLifecycle::new(1, 1);
        for i in 0..10 {
            lifecycle1.update_missed(i + 1);
        }

        // Create lifecycle with 16.67% coverage (1/6, just below 20%)
        let mut lifecycle2 = TrackLifecycle::new(2, 1);
        lifecycle2.update_matched(1);
        for i in 0..5 {
            lifecycle2.update_missed(i + 2);
        }

        acc.track_lifecycles.insert(1, lifecycle1);
        acc.track_lifecycles.insert(2, lifecycle2);

        let (mt, ml, pt, _) = acc.compute_extended_metrics();

        assert_eq!(mt, 0, "Expected 0 MT tracks");
        assert_eq!(ml, 2, "Expected 2 ML tracks");
        assert_eq!(pt, 0, "Expected 0 PT tracks");
    }

    #[test]
    fn test_compute_extended_metrics_partially_tracked() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Create lifecycle with 50% coverage
        let mut lifecycle = TrackLifecycle::new(1, 1);
        for i in 0..5 {
            lifecycle.update_matched(i * 2 + 1);
            lifecycle.update_missed(i * 2 + 2);
        }

        acc.track_lifecycles.insert(1, lifecycle);

        let (mt, ml, pt, _) = acc.compute_extended_metrics();

        assert_eq!(mt, 0, "Expected 0 MT tracks");
        assert_eq!(ml, 0, "Expected 0 ML tracks");
        assert_eq!(pt, 1, "Expected 1 PT track");
    }

    #[test]
    fn test_compute_extended_metrics_fragmentations() {
        let mut acc = ExtendedMOTAccumulator::new("test");

        // Track 1: 2 fragmentations
        let mut lifecycle1 = TrackLifecycle::new(1, 1);
        lifecycle1.update_matched(1);
        lifecycle1.update_missed(2);
        lifecycle1.update_matched(3); // Frag 1
        lifecycle1.update_missed(4);
        lifecycle1.update_matched(5); // Frag 2

        // Track 2: 0 fragmentations
        let mut lifecycle2 = TrackLifecycle::new(2, 1);
        lifecycle2.update_matched(1);
        lifecycle2.update_matched(2);

        acc.track_lifecycles.insert(1, lifecycle1);
        acc.track_lifecycles.insert(2, lifecycle2);

        let (_, _, _, total_frag) = acc.compute_extended_metrics();

        assert_eq!(total_frag, 2, "Expected 2 total fragmentations");
    }
}
