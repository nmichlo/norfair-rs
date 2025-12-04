//! MOT metrics accumulator.

use std::collections::{HashMap, HashSet};
use nalgebra::DMatrix;
use crate::internal::motmetrics::iou_matrix;

/// Accumulator for MOT (Multi-Object Tracking) metrics.
///
/// Collects frame-by-frame tracking results and computes metrics like
/// MOTA, MOTP, IDF1, etc.
#[derive(Debug, Default)]
pub struct MOTAccumulator {
    /// Frame-level events: (frame, event_type, object_id, hypothesis_id, distance)
    events: Vec<MOTEvent>,

    /// Mapping of ground truth IDs to hypothesis IDs
    matches: HashMap<i32, i32>,

    /// Ground truth IDs seen
    gt_ids_seen: HashSet<i32>,

    /// Hypothesis IDs seen
    hyp_ids_seen: HashSet<i32>,
}

/// A single MOT event.
#[derive(Debug, Clone)]
pub struct MOTEvent {
    /// Frame number
    pub frame: i32,
    /// Event type
    pub event_type: MOTEventType,
    /// Ground truth object ID (if applicable)
    pub gt_id: Option<i32>,
    /// Hypothesis (predicted) object ID (if applicable)
    pub hyp_id: Option<i32>,
    /// Distance/IoU (if applicable)
    pub distance: Option<f64>,
}

/// Types of MOT events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MOTEventType {
    /// True positive match
    Match,
    /// False positive (hypothesis without ground truth)
    FalsePositive,
    /// False negative (ground truth without hypothesis)
    Miss,
    /// ID switch (correct detection but wrong ID)
    Switch,
}

impl MOTAccumulator {
    /// Create a new accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the accumulator with frame results.
    ///
    /// # Arguments
    /// * `frame` - Frame number
    /// * `gt_ids` - Ground truth object IDs for this frame
    /// * `hyp_ids` - Hypothesis (predicted) object IDs for this frame
    /// * `gt_boxes` - Ground truth bounding boxes (N x 4 matrix: x1, y1, x2, y2)
    /// * `hyp_boxes` - Hypothesis bounding boxes (M x 4 matrix)
    /// * `iou_threshold` - IoU threshold for matching (typically 0.5)
    pub fn update(
        &mut self,
        frame: i32,
        gt_ids: &[i32],
        hyp_ids: &[i32],
        gt_boxes: &DMatrix<f64>,
        hyp_boxes: &DMatrix<f64>,
        iou_threshold: f64,
    ) {
        // Record seen IDs
        for &id in gt_ids {
            self.gt_ids_seen.insert(id);
        }
        for &id in hyp_ids {
            self.hyp_ids_seen.insert(id);
        }

        // Handle empty cases
        if gt_ids.is_empty() {
            // All hypotheses are false positives
            for &hyp_id in hyp_ids {
                self.events.push(MOTEvent {
                    frame,
                    event_type: MOTEventType::FalsePositive,
                    gt_id: None,
                    hyp_id: Some(hyp_id),
                    distance: None,
                });
            }
            return;
        }

        if hyp_ids.is_empty() {
            // All ground truths are misses
            for &gt_id in gt_ids {
                self.events.push(MOTEvent {
                    frame,
                    event_type: MOTEventType::Miss,
                    gt_id: Some(gt_id),
                    hyp_id: None,
                    distance: None,
                });
            }
            return;
        }

        // Compute IoU matrix (actual IoU values, not distances)
        let iou_mat = iou_matrix(gt_boxes, hyp_boxes);

        // Greedy matching based on IoU
        let mut matched_gt: HashSet<usize> = HashSet::new();
        let mut matched_hyp: HashSet<usize> = HashSet::new();

        // Collect valid matches (iou, gt_idx, hyp_idx)
        let mut valid_matches: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..gt_ids.len() {
            for j in 0..hyp_ids.len() {
                let iou = iou_mat[(i, j)];
                if iou >= iou_threshold {
                    valid_matches.push((iou, i, j));
                }
            }
        }

        // Sort by IoU (descending - highest IoU first)
        valid_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Greedy matching
        for (iou, gt_idx, hyp_idx) in valid_matches {
            if matched_gt.contains(&gt_idx) || matched_hyp.contains(&hyp_idx) {
                continue;
            }

            matched_gt.insert(gt_idx);
            matched_hyp.insert(hyp_idx);

            let gt_id = gt_ids[gt_idx];
            let hyp_id = hyp_ids[hyp_idx];

            // Check for ID switch
            let event_type = if let Some(&prev_hyp) = self.matches.get(&gt_id) {
                if prev_hyp != hyp_id {
                    MOTEventType::Switch
                } else {
                    MOTEventType::Match
                }
            } else {
                MOTEventType::Match
            };

            self.matches.insert(gt_id, hyp_id);
            // Store as distance (1 - IoU) for consistency with MOTP calculation
            self.events.push(MOTEvent {
                frame,
                event_type,
                gt_id: Some(gt_id),
                hyp_id: Some(hyp_id),
                distance: Some(1.0 - iou),
            });
        }

        // Unmatched ground truths are misses
        for (idx, &gt_id) in gt_ids.iter().enumerate() {
            if !matched_gt.contains(&idx) {
                self.events.push(MOTEvent {
                    frame,
                    event_type: MOTEventType::Miss,
                    gt_id: Some(gt_id),
                    hyp_id: None,
                    distance: None,
                });
            }
        }

        // Unmatched hypotheses are false positives
        for (idx, &hyp_id) in hyp_ids.iter().enumerate() {
            if !matched_hyp.contains(&idx) {
                self.events.push(MOTEvent {
                    frame,
                    event_type: MOTEventType::FalsePositive,
                    gt_id: None,
                    hyp_id: Some(hyp_id),
                    distance: None,
                });
            }
        }
    }

    /// Get all events.
    pub fn events(&self) -> &[MOTEvent] {
        &self.events
    }

    /// Count events by type.
    pub fn count_events(&self, event_type: MOTEventType) -> usize {
        self.events.iter().filter(|e| e.event_type == event_type).count()
    }

    /// Get the number of matches.
    pub fn num_matches(&self) -> usize {
        self.count_events(MOTEventType::Match)
    }

    /// Get the number of false positives.
    pub fn num_false_positives(&self) -> usize {
        self.count_events(MOTEventType::FalsePositive)
    }

    /// Get the number of misses.
    pub fn num_misses(&self) -> usize {
        self.count_events(MOTEventType::Miss)
    }

    /// Get the number of ID switches.
    pub fn num_switches(&self) -> usize {
        self.count_events(MOTEventType::Switch)
    }

    /// Get the number of unique ground truth IDs.
    pub fn num_gt_ids(&self) -> usize {
        self.gt_ids_seen.len()
    }

    /// Get the number of unique hypothesis IDs.
    pub fn num_hyp_ids(&self) -> usize {
        self.hyp_ids_seen.len()
    }

    /// Compute MOTA (Multi-Object Tracking Accuracy).
    ///
    /// MOTA = 1 - (FN + FP + IDSW) / num_gt
    pub fn mota(&self) -> f64 {
        let num_gt: usize = self.events.iter()
            .filter(|e| e.gt_id.is_some())
            .count();

        if num_gt == 0 {
            return 0.0;
        }

        let fn_count = self.num_misses();
        let fp = self.num_false_positives();
        let idsw = self.num_switches();

        1.0 - (fn_count + fp + idsw) as f64 / num_gt as f64
    }

    /// Compute MOTP (Multi-Object Tracking Precision).
    ///
    /// MOTP = sum(IoU) / num_matches
    pub fn motp(&self) -> f64 {
        let matches: Vec<_> = self.events.iter()
            .filter(|e| e.event_type == MOTEventType::Match || e.event_type == MOTEventType::Switch)
            .filter_map(|e| e.distance)
            .collect();

        if matches.is_empty() {
            return 0.0;
        }

        // Convert distance back to IoU (1 - distance)
        let total_iou: f64 = matches.iter().map(|d| 1.0 - d).sum();
        total_iou / matches.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_empty() {
        let acc = MOTAccumulator::new();
        assert_eq!(acc.num_matches(), 0);
        assert_eq!(acc.num_false_positives(), 0);
        assert_eq!(acc.num_misses(), 0);
    }

    #[test]
    fn test_perfect_tracking() {
        let mut acc = MOTAccumulator::new();

        // Perfect overlap
        let gt_boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let hyp_boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);

        acc.update(1, &[1], &[1], &gt_boxes, &hyp_boxes, 0.5);

        assert_eq!(acc.num_matches(), 1);
        assert_eq!(acc.num_false_positives(), 0);
        assert_eq!(acc.num_misses(), 0);
        assert!((acc.mota() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_misses() {
        let mut acc = MOTAccumulator::new();

        let gt_boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let hyp_boxes = DMatrix::zeros(0, 4);

        acc.update(1, &[1], &[], &gt_boxes, &hyp_boxes, 0.5);

        assert_eq!(acc.num_matches(), 0);
        assert_eq!(acc.num_misses(), 1);
        assert!((acc.mota() - 0.0).abs() < 1e-10);
    }

    // ===== Ported from Go: TestEvalMotChallenge_* (accumulator-level equivalents) =====

    /// Ported from Go: TestEvalMotChallenge_Perfect
    /// Tests perfect tracking over multiple frames
    #[test]
    fn test_accumulator_perfect_multiframe() {
        let mut acc = MOTAccumulator::new();

        // Simulate 10 frames with 2 objects tracked perfectly
        // Same positions as gt1.txt/pred1.txt (though IDs differ, positions match)
        for frame in 1..=10 {
            // Two objects at (100,100,50,50) and (200,200,50,50)
            // In MOT format: x,y,w,h -> convert to x1,y1,x2,y2
            let gt_boxes = DMatrix::from_row_slice(2, 4, &[
                100.0, 100.0, 150.0, 150.0,  // Object 1
                200.0, 200.0, 250.0, 250.0,  // Object 2
            ]);
            let hyp_boxes = DMatrix::from_row_slice(2, 4, &[
                100.0, 100.0, 150.0, 150.0,  // Object 1
                200.0, 200.0, 250.0, 250.0,  // Object 2
            ]);

            acc.update(frame, &[1, 2], &[1, 2], &gt_boxes, &hyp_boxes, 0.5);
        }

        // Perfect tracking should have MOTA ≈ 1.0
        assert!(acc.mota() > 0.99, "Perfect tracking should have MOTA ≈ 1.0, got {}", acc.mota());

        // Should have no false positives or misses
        assert_eq!(acc.num_false_positives(), 0, "Expected 0 false positives");
        assert_eq!(acc.num_misses(), 0, "Expected 0 misses");
        assert_eq!(acc.num_switches(), 0, "Expected 0 switches");

        // Should have 20 matches (2 per frame * 10 frames)
        assert_eq!(acc.num_matches(), 20, "Expected 20 matches");
    }

    /// Ported from Go: TestEvalMotChallenge_MostlyLost
    /// Tests scenario with many misses
    #[test]
    fn test_accumulator_mostly_lost() {
        let mut acc = MOTAccumulator::new();

        // Simulate tracking where predictions only cover first 2 frames
        for frame in 1..=10 {
            let gt_boxes = DMatrix::from_row_slice(2, 4, &[
                100.0, 100.0, 150.0, 150.0,
                200.0, 200.0, 250.0, 250.0,
            ]);

            if frame <= 2 {
                // Only predict in first 2 frames
                let hyp_boxes = DMatrix::from_row_slice(2, 4, &[
                    100.0, 100.0, 150.0, 150.0,
                    200.0, 200.0, 250.0, 250.0,
                ]);
                acc.update(frame, &[1, 2], &[1, 2], &gt_boxes, &hyp_boxes, 0.5);
            } else {
                // No predictions - all misses
                let hyp_boxes = DMatrix::zeros(0, 4);
                acc.update(frame, &[1, 2], &[], &gt_boxes, &hyp_boxes, 0.5);
            }
        }

        // Mostly lost should have poor MOTA
        assert!(acc.mota() < 0.5, "Mostly lost tracking should have low MOTA, got {}", acc.mota());

        // Should have many misses (16 = 2 objects * 8 frames without predictions)
        assert_eq!(acc.num_misses(), 16, "Expected 16 misses");
    }

    /// Ported from Go: TestEvalMotChallenge_Fragmented
    /// Tests scenario with ID switches
    #[test]
    fn test_accumulator_fragmented() {
        let mut acc = MOTAccumulator::new();

        // Frame 1: Object 1 matched with hypothesis 1
        let gt_boxes = DMatrix::from_row_slice(1, 4, &[100.0, 100.0, 150.0, 150.0]);
        let hyp_boxes = DMatrix::from_row_slice(1, 4, &[100.0, 100.0, 150.0, 150.0]);
        acc.update(1, &[1], &[1], &gt_boxes, &hyp_boxes, 0.5);

        // Frame 2: Same object now matched with hypothesis 2 (ID switch)
        acc.update(2, &[1], &[2], &gt_boxes, &hyp_boxes, 0.5);

        // Frame 3: Back to hypothesis 1 (another ID switch)
        acc.update(3, &[1], &[1], &gt_boxes, &hyp_boxes, 0.5);

        // Should have ID switches
        assert!(acc.num_switches() > 0, "Expected ID switches for fragmented tracking");

        // All detections should still be matches (just with switches)
        assert_eq!(acc.num_matches() + acc.num_switches(), 3, "Expected 3 total matched events");
    }

    /// Ported from Go: TestEvalMotChallenge_Mixed
    /// Tests mixed scenario with some matches, misses, and false positives
    #[test]
    fn test_accumulator_mixed() {
        let mut acc = MOTAccumulator::new();

        // Frame 1: 2 GT objects, 2 hypotheses - perfect match
        let gt_boxes = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);
        let hyp_boxes = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);
        acc.update(1, &[1, 2], &[1, 2], &gt_boxes, &hyp_boxes, 0.5);

        // Frame 2: 2 GT objects, 1 hypothesis - 1 match, 1 miss
        let gt_boxes2 = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);
        let hyp_boxes2 = DMatrix::from_row_slice(1, 4, &[
            100.0, 100.0, 150.0, 150.0,
        ]);
        acc.update(2, &[1, 2], &[1], &gt_boxes2, &hyp_boxes2, 0.5);

        // Frame 3: 1 GT object, 2 hypotheses - 1 match, 1 false positive
        let gt_boxes3 = DMatrix::from_row_slice(1, 4, &[100.0, 100.0, 150.0, 150.0]);
        let hyp_boxes3 = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            300.0, 300.0, 350.0, 350.0,  // FP - no overlapping GT
        ]);
        acc.update(3, &[1], &[1, 3], &gt_boxes3, &hyp_boxes3, 0.5);

        // Should have some matches
        assert!(acc.num_matches() > 0, "Expected some matches in mixed scenario");

        // MOTA should be between 0 and 1
        let mota = acc.mota();
        assert!(mota >= 0.0 && mota <= 1.0, "MOTA should be in [0, 1], got {}", mota);

        // Should have at least 1 miss and 1 false positive
        assert!(acc.num_misses() >= 1, "Expected at least 1 miss");
        assert!(acc.num_false_positives() >= 1, "Expected at least 1 false positive");
    }

    /// Test MOTP calculation
    #[test]
    fn test_accumulator_motp() {
        let mut acc = MOTAccumulator::new();

        // Perfect overlap should have MOTP = 1.0
        let gt_boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let hyp_boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);

        acc.update(1, &[1], &[1], &gt_boxes, &hyp_boxes, 0.5);

        assert!((acc.motp() - 1.0).abs() < 1e-10, "MOTP should be 1.0 for perfect overlap");
    }

    /// Test all false positives scenario
    #[test]
    fn test_accumulator_all_false_positives() {
        let mut acc = MOTAccumulator::new();

        let gt_boxes = DMatrix::zeros(0, 4);
        let hyp_boxes = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);

        acc.update(1, &[], &[1, 2], &gt_boxes, &hyp_boxes, 0.5);

        assert_eq!(acc.num_matches(), 0);
        assert_eq!(acc.num_misses(), 0);
        assert_eq!(acc.num_false_positives(), 2);
    }

    /// Test GT/hypothesis ID counting
    #[test]
    fn test_accumulator_id_counts() {
        let mut acc = MOTAccumulator::new();

        let gt_boxes = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);
        let hyp_boxes = DMatrix::from_row_slice(2, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
        ]);

        // Frame 1
        acc.update(1, &[1, 2], &[10, 20], &gt_boxes, &hyp_boxes, 0.5);
        // Frame 2 - same IDs
        acc.update(2, &[1, 2], &[10, 20], &gt_boxes, &hyp_boxes, 0.5);
        // Frame 3 - add new GT ID
        acc.update(3, &[1, 2, 3], &[10, 20, 30], &DMatrix::from_row_slice(3, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
            300.0, 300.0, 350.0, 350.0,
        ]), &DMatrix::from_row_slice(3, 4, &[
            100.0, 100.0, 150.0, 150.0,
            200.0, 200.0, 250.0, 250.0,
            300.0, 300.0, 350.0, 350.0,
        ]), 0.5);

        assert_eq!(acc.num_gt_ids(), 3, "Expected 3 unique GT IDs");
        assert_eq!(acc.num_hyp_ids(), 3, "Expected 3 unique hypothesis IDs");
    }
}
