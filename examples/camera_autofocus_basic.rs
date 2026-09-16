//! 简单的相机自动追焦示例
//!
//! 本示例展示如何使用norfair-rs实现基本的自动追焦功能。
//! 不包含硬件集成，仅演示追踪和焦距计算逻辑。
//!
//! 使用方法:
//!   cargo run --example camera_autofocus_basic

use norfair_rs::{Detection, Tracker, TrackerConfig};

/// 简化的焦距控制器
pub struct SimpleFocusController {
    tracker: Tracker,
    frame_width: f64,
    frame_height: f64,
}

/// 焦距命令
#[derive(Debug, Clone)]
pub struct FocusCommand {
    pub pan: f64,           // [-1, 1]
    pub tilt: f64,          // [-1, 1]
    pub zoom_speed: f64,    // [-0.5, 0.5]
    pub focus_value: u16,   // [0, 1000]
    pub tracking_id: Option<i32>,
    pub confidence: f64,
}

impl SimpleFocusController {
    pub fn new(frame_width: f64, frame_height: f64) 
        -> Result<Self, Box<dyn std::error::Error>> 
    {
        let mut config = TrackerConfig::from_distance_name("iou", 0.5);
        config.hit_counter_max = 30;
        config.initialization_delay = 3;
        
        let tracker = Tracker::new(config)?;
        
        Ok(Self {
            tracker,
            frame_width,
            frame_height,
        })
    }
    
    pub fn update(&mut self, detections: Vec<Detection>) -> FocusCommand {
        let tracked_objects = self.tracker.update(detections, 1, None);
        
        // 找最大的活跃对象
        let best = tracked_objects
            .iter()
            .filter(|obj| obj.id.is_some())
            .max_by(|a, b| {
                let size_a = (a.estimate[(0, 2)] - a.estimate[(0, 0)])
                           * (a.estimate[(1, 3)] - a.estimate[(1, 1)]);
                let size_b = (b.estimate[(0, 2)] - b.estimate[(0, 0)])
                           * (b.estimate[(1, 3)] - b.estimate[(1, 1)]);
                size_a.partial_cmp(&size_b).unwrap_or(std::cmp::Ordering::Equal)
            });
        
        if let Some(obj) = best {
            let bbox = &obj.estimate;
            let cx = (bbox[(0, 0)] + bbox[(0, 2)]) / 2.0;
            let cy = (bbox[(0, 1)] + bbox[(0, 3)]) / 2.0;
            let w = (bbox[(0, 2)] - bbox[(0, 0)]).abs();
            let h = (bbox[(1, 3)] - bbox[(1, 1)]).abs();
            
            // 平移
            let center_x = self.frame_width / 2.0;
            let center_y = self.frame_height / 2.0;
            let pan = ((cx - center_x) * 0.02).clamp(-1.0, 1.0);
            let tilt = ((cy - center_y) * 0.02).clamp(-1.0, 1.0);
            
            // 变焦
            let target_size = self.frame_width * 0.3;
            let current_size = (w + h) / 2.0;
            let zoom_factor = if current_size > 0.0 {
                target_size / current_size
            } else {
                1.0
            };
            let zoom_speed = ((zoom_factor - 1.0) * 0.05).clamp(-0.5, 0.5);
            
            // 对焦
            let distance = (self.frame_width * self.frame_height) / (w * h).max(1.0);
            let focus_value = ((distance / 1000.0).min(1.0) * 1000.0) as u16;
            
            FocusCommand {
                pan,
                tilt,
                zoom_speed,
                focus_value,
                tracking_id: obj.id,
                confidence: (obj.hit_counter as f64 / 30.0).min(1.0),
            }
        } else {
            FocusCommand {
                pan: 0.0,
                tilt: 0.0,
                zoom_speed: 0.0,
                focus_value: 500,
                tracking_id: None,
                confidence: 0.0,
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("norfair-rs 自动追焦演示");
    println!("========================\n");
    
    // 创建控制器 (假设640x480分辨率)
    let mut controller = SimpleFocusController::new(640.0, 480.0)?;
    
    // 模拟3帧的检测数据
    let frames = vec![
        // 帧1: 一个检测
        vec![
            Detection::from_slice(&[100.0, 150.0, 200.0, 300.0], 1, 4)?,
        ],
        // 帧2: 同一对象，位置略微变化
        vec![
            Detection::from_slice(&[110.0, 155.0, 210.0, 305.0], 1, 4)?,
        ],
        // 帧3: 继续追踪
        vec![
            Detection::from_slice(&[120.0, 160.0, 220.0, 310.0], 1, 4)?,
        ],
    ];
    
    for (frame_idx, detections) in frames.iter().enumerate() {
        let cmd = controller.update(detections.clone());
        
        println!("帧 {}:", frame_idx);
        println!("  ID: {:?}", cmd.tracking_id);
        println!("  置信度: {:.2}%", cmd.confidence * 100.0);
        println!("  平移: Pan={:+.3}, Tilt={:+.3}", cmd.pan, cmd.tilt);
        println!("  变焦速度: {:.3}", cmd.zoom_speed);
        println!("  对焦值: {}", cmd.focus_value);
        println!();
    }
    
    Ok(())
}
