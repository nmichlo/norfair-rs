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
}
