// Benchmark for norfair-go tracking.
//
// Usage:
//
//	go run benchmark_go.go [scenario_name]
//
// Example:
//
//	go run benchmark_go.go medium
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/nmichlo/norfair-go/pkg/norfairgo"
	"gonum.org/v1/gonum/mat"
)

type Detection struct {
	BBox          []float64 `json:"bbox"`
	GroundTruthID int       `json:"ground_truth_id"`
}

type Frame struct {
	FrameID    int         `json:"frame_id"`
	Detections []Detection `json:"detections"`
}

type Scenario struct {
	Name          string  `json:"name,omitempty"`
	Seed          int     `json:"seed"`
	NumObjects    int     `json:"num_objects"`
	NumFrames     int     `json:"num_frames"`
	DetectionProb float64 `json:"detection_prob"`
	NoiseStd      float64 `json:"noise_std"`
	Frames        []Frame `json:"frames"`
}

type Results struct {
	Language            string  `json:"language"`
	Scenario            string  `json:"scenario"`
	NumFrames           int     `json:"num_frames"`
	TotalDetections     int     `json:"total_detections"`
	TotalTracked        int     `json:"total_tracked"`
	ElapsedSeconds      float64 `json:"elapsed_seconds"`
	FPS                 float64 `json:"fps"`
	DetectionsPerSecond float64 `json:"detections_per_second"`
}

func loadScenario(name string) (*Scenario, error) {
	// Get the directory of the executable
	execPath, err := os.Executable()
	if err != nil {
		execPath = "."
	}
	dataDir := filepath.Join(filepath.Dir(execPath), "data")

	// Also try relative to current directory
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		dataDir = filepath.Join(".", "data")
	}

	// Also try examples/benchmark/data
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		dataDir = filepath.Join("examples", "benchmark", "data")
	}

	path := filepath.Join(dataDir, name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read scenario: %w", err)
	}

	var scenario Scenario
	if err := json.Unmarshal(data, &scenario); err != nil {
		return nil, fmt.Errorf("failed to parse scenario: %w", err)
	}

	scenario.Name = name
	return &scenario, nil
}

func runBenchmark(scenario *Scenario) (*Results, error) {
	// Create tracker with standard settings
	tracker, err := norfairgo.NewTracker(&norfairgo.TrackerConfig{
		DistanceFunction:    norfairgo.DistanceByName("iou"),
		DistanceThreshold:   0.5,
		HitCounterMax:       15,
		InitializationDelay: 3,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create tracker: %w", err)
	}

	// Warm up
	for i := 0; i < 10; i++ {
		tracker.Update(nil, 1, nil)
	}

	// Reset tracker
	tracker, _ = norfairgo.NewTracker(&norfairgo.TrackerConfig{
		DistanceFunction:    norfairgo.DistanceByName("iou"),
		DistanceThreshold:   0.5,
		HitCounterMax:       15,
		InitializationDelay: 3,
	})

	// Run benchmark
	startTime := time.Now()
	totalTracked := 0
	totalDetections := 0

	for _, frame := range scenario.Frames {
		// Convert detections to norfair format
		var detections []*norfairgo.Detection
		for _, det := range frame.Detections {
			// Create 2x2 matrix for bounding box (top-left, bottom-right)
			points := mat.NewDense(2, 2, []float64{
				det.BBox[0], det.BBox[1], // top-left
				det.BBox[2], det.BBox[3], // bottom-right
			})
			detection, err := norfairgo.NewDetection(points, nil)
			if err != nil {
				continue
			}
			detections = append(detections, detection)
		}
		totalDetections += len(detections)

		// Update tracker
		trackedObjects := tracker.Update(detections, 1, nil)
		totalTracked += len(trackedObjects)
	}

	elapsed := time.Since(startTime).Seconds()
	numFrames := len(scenario.Frames)

	return &Results{
		Language:            "go",
		Scenario:            scenario.Name,
		NumFrames:           numFrames,
		TotalDetections:     totalDetections,
		TotalTracked:        totalTracked,
		ElapsedSeconds:      elapsed,
		FPS:                 float64(numFrames) / elapsed,
		DetectionsPerSecond: float64(totalDetections) / elapsed,
	}, nil
}

func main() {
	scenarioName := "medium"
	if len(os.Args) > 1 {
		scenarioName = os.Args[1]
	}

	// fmt.Printf("Loading scenario: %s\n", scenarioName)
	scenario, err := loadScenario(scenarioName)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("Run generate_data.py first to create test data.")
		os.Exit(1)
	}

	// fmt.Println("Running Go benchmark...")
	// fmt.Printf("  Objects: %d\n", scenario.NumObjects)
	// fmt.Printf("  Frames: %d\n", scenario.NumFrames)

	results, err := runBenchmark(scenario)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	// fmt.Println("\nResults:")
	// fmt.Printf("  Elapsed: %.3fs\n", results.ElapsedSeconds)
	// fmt.Printf("  FPS: %.1f\n", results.FPS)
	// fmt.Printf("  Detections/sec: %.0f\n", results.DetectionsPerSecond)

	// Output JSON for comparison
	jsonBytes, _ := json.Marshal(results)
	fmt.Printf("\n%s\n", string(jsonBytes))
}
