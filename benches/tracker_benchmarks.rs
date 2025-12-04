//! Tracker benchmarks using Criterion.
//!
//! Ported from Go: pkg/norfairgo/benchmark_test.go
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;

use norfair_rs::distances::distance_function_by_name;
use norfair_rs::filter::{FilterFactoryEnum, FilterPyKalmanFilterFactory, NoFilterFactory, OptimizedKalmanFilterFactory};
use norfair_rs::{Detection, Tracker, TrackerConfig};

/// Create test detections for benchmarking.
/// Ported from Go: createTestDetections
fn create_test_detections(n: usize) -> Vec<Detection> {
    (0..n)
        .map(|i| {
            let x = (i * 100) as f64;
            let y = (i * 50) as f64;
            // 2x2 matrix: two points (top-left, bottom-right style)
            let points = DMatrix::from_row_slice(2, 2, &[x, y, x + 50.0, y + 50.0]);
            Detection::new(points).expect("valid detection")
        })
        .collect()
}

/// Ported from Go: BenchmarkTrackerUpdate_10Objects
fn benchmark_tracker_update_10_objects(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("euclidean"), 50.0);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0));

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(10);

    c.bench_function("tracker_update_10_objects", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

/// Ported from Go: BenchmarkTrackerUpdate_50Objects
fn benchmark_tracker_update_50_objects(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("euclidean"), 50.0);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0));

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(50);

    c.bench_function("tracker_update_50_objects", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

/// Ported from Go: BenchmarkTrackerUpdate_100Objects
fn benchmark_tracker_update_100_objects(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("euclidean"), 50.0);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0));

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(100);

    c.bench_function("tracker_update_100_objects", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

/// Ported from Go: BenchmarkTrackerUpdate_100Objects_FilterPyKalman
fn benchmark_tracker_update_100_objects_filterpy_kalman(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("euclidean"), 50.0);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::FilterPy(FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0));

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(100);

    c.bench_function("tracker_update_100_objects_filterpy_kalman", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

/// Ported from Go: BenchmarkTrackerUpdate_100Objects_NoFilter
fn benchmark_tracker_update_100_objects_no_filter(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("euclidean"), 50.0);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::None(NoFilterFactory);

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(100);

    c.bench_function("tracker_update_100_objects_no_filter", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

/// Ported from Go: BenchmarkTrackerUpdate_100Objects_IoU
fn benchmark_tracker_update_100_objects_iou(c: &mut Criterion) {
    let mut config = TrackerConfig::new(distance_function_by_name("iou"), 0.5);
    config.hit_counter_max = 30;
    config.initialization_delay = 3;
    config.filter_factory = FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0));

    let mut tracker = Tracker::new(config).expect("valid tracker");
    let detections = create_test_detections(100);

    c.bench_function("tracker_update_100_objects_iou", |b| {
        b.iter(|| {
            tracker.update(black_box(detections.clone()), 1, None);
        })
    });
}

criterion_group!(
    benches,
    benchmark_tracker_update_10_objects,
    benchmark_tracker_update_50_objects,
    benchmark_tracker_update_100_objects,
    benchmark_tracker_update_100_objects_filterpy_kalman,
    benchmark_tracker_update_100_objects_no_filter,
    benchmark_tracker_update_100_objects_iou,
);
criterion_main!(benches);
