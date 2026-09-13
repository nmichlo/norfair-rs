# 实时相机自动追焦集成指南

**Real-Time Camera Auto-Focus Integration Guide for norfair-rs**

---

## 📖 目录

1. [概述](#概述)
2. [系统架构](#系统架构)
3. [核心概念](#核心概念)
4. [完整Rust实现](#完整rust实现)
5. [硬件集成](#硬件集成)
6. [配置和调试](#配置和调试)
7. [性能指标](#性能指标)
8. [常见问题](#常见问题)

---

## 概述

本指南展示如何将 **norfair-rs** 集成到实时相机软件中，实现：

✅ **实时多对象追踪** - 通过稳定的ID追踪人物/物体  
✅ **自动平移 (Pan)** - 保持主体在画面中心  
✅ **自动变焦 (Zoom)** - 维持对象大小恒定  
✅ **自动对焦 (Focus)** - 根据距离调整焦距  
✅ **相机运动补偿** - 补偿云台/手持抖动  
✅ **预测追踪** - 提前补偿对象运动  
✅ **性能优化** - 60-180倍快于原始Python版本  

### 性能对标

| 场景 | norfair (Python) | norfair-rs (Rust) | 加速比 |
|------|-----------------|-------------------|-------|
| 小规模 | 4,700 fps | 296,000 fps | **63x** |
| 中等 | 540 fps | 89,000 fps | **165x** |
| 大规模 | 101 fps | 41,000 fps | **406x** |

---

## 系统架构

### 数据流

```
┌────────────────────────────────────────────────────────────────────────┐
│                        摄像头输入                                 │
│                    (30/60 FPS视频流)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────────────────────────────┐
        │   对象检测 (YOLO/Faster R-CNN)   │
        │  检测人物/物体的边界框和置信度   │
        └────────────┬─────────────────────────┘
                     │ detections[]
        ┌────────────────────────────────────────┐
        │    追踪器 (norfair-rs)           │
        │  • 分配/维护稳定ID               │
        │  • 补偿相机运动                  │
        │  • 速度估计和预测                │
        │  • Kalman滤波状态维护            │
        └────────────┬─────────────────────────┘
                     │ tracked_objects[]
        ┌────────────────────────────────────────┐
        │  自动追焦控制器                  │
        │  (AutoFocusController)          │
        │  • PID平移控制                  │
        │  • 变焦反馈控制                  │
        │  • 深度估计对焦值                │
        │  • 预测性运动补偿                │
        └────────────┬─────────────────────────┘
                     │ FocusCommand
        ┌────────────────────────────────────────┐
        │  相机硬件控制接口                │
        │  (CameraHardware trait)         │
        └────────────┬─────────────────────────┘
                     │
        ┌────────────────────────────────────────┐
        │  相机硬件                        │
        │  • PTZ云台 (VISCA协议)          │
        │  • 变焦镜头 (USB/Serial)        │
        │  • 自动对焦马达                  │
        └────────────────────────────────────────┘
```

### 时序

```
[相机帧] → [YOLO检测] → [norfair追踪] → [焦距计算] → [硬件命令]
  0ms      50-100ms      5-10ms          2-5ms        1-2ms
  └─────────────────────────────────────────────────────────────┘
                    总延迟: 60-120ms (足以实时追踪)
```

---

## 核心概念

### 1. 相对坐标 vs 绝对坐标

```rust
// 相对坐标 (Relative) = 相机画面坐标系
// • (0,0) = 左上角
// • (w,h) = 右下角  
// • 随相机运动而变化
// • 用于距离计算、显示

// 绝对坐标 (Absolute) = 世界坐标系
// • (0,0) = 第一帧的左上角
// • 固定不变（补偿相机运动后）
// • 用于Kalman滤波器内部状态
// • 用于真实世界位置追踪

let relative = obj.estimate;           // 画面坐标
let absolute = obj.get_estimate(true); // 世界坐标
```

### 2. 追踪生命周期

```rust
TrackedObject状态转移:

创建 (Create)
  ↓
初始化阶段 (Initialization Phase)
  ├─ is_initializing = true
  ├─ id = None (无永久ID)
  ├─ initializing_id = Some(N) (临时ID)
  ├─ 需要initialization_delay次检测
  └─ 超过hit_counter则死亡
  ↓
活跃阶段 (Active Phase)
  ├─ is_initializing = false
  ├─ id = Some(N) (永久ID)
  ├─ 参与追踪和匹配
  └─ 无匹配时逐帧衰减
  ↓
衰减阶段 (Decay Phase)
  ├─ hit_counter > 0 (仍可见)
  ├─ 无新检测时hit_counter -= 1
  └─ 用于处理临时遮挡
  ↓
死亡/重识别 (Death/ReID)
  ├─ hit_counter < 0
  ├─ 如果启用ReID，尝试匹配past_detections
  └─ 失败则完全移除
```

### 3. 自动追焦控制

```rust
// PID平移控制
Pan控制:  误差 = 对象中心X - 画面中心X
          输出 = 误差 × 0.02 (P系数)
          范围 = [-1.0, +1.0] (左到右)

Tilt控制: 误差 = 对象中心Y - 画面中心Y
          输出 = 误差 × 0.02 (P系数)
          范围 = [-1.0, +1.0] (下到上)

// 变焦反馈控制
目标大小 = 画面宽度 × 30%
当前大小 = (对象宽 + 对象高) / 2
缩放因子 = 目标大小 / 当前大小
输出 = (缩放因子 - 1.0) × 0.05 (反馈增益)
范围 = [-0.5, +0.5] (缩小到放大)

// 深度估计对焦
估计距离 = (画面宽 × 画面高) / (对象宽 × 对象高)
焦距值 = (距离 / 1000.0).min(1.0) × 1000
范围 = [0, 1000] (焦距单位)

// 预测追踪
预测时间 = 100ms (100毫秒后的位置)
预测X = 当前X + 速度X × 预测时间
预测Y = 当前Y + 速度Y × 预测时间
预测平移 = (预测Y - 画面中心Y) × 0.02
```

---

## 完整Rust实现

### 关键数据结构

```rust
use norfair_rs::{Detection, Tracker, TrackerConfig};
use norfair_rs::camera_motion::TranslationTransformation;
use nalgebra::DMatrix;
use std::time::Instant;

/// 追踪的主体信息
pub struct SubjectTracker {
    /// 当前追踪的对象ID
    pub primary_id: Option<i32>,
    
    /// 对象在世界坐标系中的位置 (绝对坐标)
    pub world_position: (f64, f64),
    
    /// 对象在相机画面中的位置 (相对坐标)
    pub frame_position: (f64, f64),
    
    /// 对象的速度向量 [vx, vy]
    pub velocity: (f64, f64),
    
    /// 对象的宽度和高度
    pub size: (f64, f64),
    
    /// 追踪置信度 (0.0 = 刚创建, 1.0 = 完全稳定)
    pub confidence: f64,
    
    /// 最后一次更新的时间
    pub last_update: Instant,
}

impl SubjectTracker {
    pub fn new() -> Self {
        Self {
            primary_id: None,
            world_position: (0.0, 0.0),
            frame_position: (0.0, 0.0),
            velocity: (0.0, 0.0),
            size: (0.0, 0.0),
            confidence: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// 焦距控制命令
#[derive(Debug, Clone)]
pub struct FocusCommand {
    /// 平移速度 (-1 = 左, +1 = 右)
    pub pan: f64,
    
    /// 倾斜速度 (-1 = 下, +1 = 上)
    pub tilt: f64,
    
    /// 变焦速度 (-0.5 = 缩小, +0.5 = 放大)
    pub zoom_speed: f64,
    
    /// 对焦值 (0-1000)
    pub focus_value: u16,
    
    /// 追踪对象的ID
    pub tracking_id: Option<i32>,
    
    /// 追踪置信度 (0.0-1.0)
    pub confidence: f64,
    
    /// 主体在画面中的位置 (用于UI显示)
    pub subject_position: (f64, f64),
    
    /// 主体的大小 (用于UI显示)
    pub subject_size: (f64, f64),
}

impl Default for FocusCommand {
    fn default() -> Self {
        Self {
            pan: 0.0,
            tilt: 0.0,
            zoom_speed: 0.0,
            focus_value: 500,
            tracking_id: None,
            confidence: 0.0,
            subject_position: (0.0, 0.0),
            subject_size: (0.0, 0.0),
        }
    }
}
```

### 自动追焦控制器

```rust
/// 自动追焦控制器 - 核心逻辑
pub struct AutoFocusController {
    /// norfair追踪器
    tracker: Tracker,
    
    /// 当前主要追踪对象
    subject: SubjectTracker,
    
    /// 帧尺寸
    frame_width: f64,
    frame_height: f64,
}

impl AutoFocusController {
    /// 创建新的自动追焦控制器
    pub fn new(frame_width: f64, frame_height: f64) 
        -> Result<Self, Box<dyn std::error::Error>> 
    {
        // 配置追踪器：针对人物/物体追踪优化
        let mut config = TrackerConfig::from_distance_name("iou", 0.5);
        config.hit_counter_max = 30;           // 保持追踪30帧（无检测时）
        config.initialization_delay = 3;       // 需要3次检测才分配ID
        config.detection_threshold = 0.5;      // 最小检测置信度
        config.past_detections_length = 10;    // 存储历史用于速度估计
        
        let tracker = Tracker::new(config)?;
        
        Ok(Self {
            tracker,
            subject: SubjectTracker::new(),
            frame_width,
            frame_height,
        })
    }

    /// 更新追踪 + 计算自动追焦参数
    pub fn update(
        &mut self,
        detections: Vec<Detection>,
        optical_flow: Option<(f64, f64)>,  // 相机运动: (dx, dy)
    ) -> FocusCommand 
    {
        // ========== 步骤1: 补偿相机运动 ==========
        let coord_transform = optical_flow.map(|(dx, dy)| {
            TranslationTransformation::new([dx, dy])
        });

        // ========== 步骤2: 更新追踪器 ==========
        let tracked_objects = self.tracker.update(
            detections,
            1,  // period = 1帧
            coord_transform.as_ref()
                .map(|t| t as &dyn norfair_rs::camera_motion::CoordinateTransformation),
        );

        // ========== 步骤3: 选择主要追踪对象 ==========
        // 策略: 选择最大且最稳定的对象
        let best_object = tracked_objects
            .iter()
            .filter(|obj| obj.id.is_some() && !obj.is_initializing)
            .max_by(|a, b| {
                // 比较对象面积
                let size_a = (a.estimate[(0, 2)] - a.estimate[(0, 0)])
                           * (a.estimate[(1, 3)] - a.estimate[(1, 1)]);
                let size_b = (b.estimate[(0, 2)] - b.estimate[(0, 0)])
                           * (b.estimate[(1, 3)] - b.estimate[(1, 1)]);
                size_a.partial_cmp(&size_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(obj) = best_object {
            self.update_subject_from_object(obj, coord_transform.as_ref());
        }

        // ========== 步骤4: 计算焦距控制命令 ==========
        self.compute_focus_command()
    }

    /// 从追踪对象更新主体信息
    fn update_subject_from_object(
        &mut self,
        obj: &norfair_rs::TrackedObject,
        _transform: Option<&TranslationTransformation>,
    ) {
        self.subject.primary_id = obj.id;
        
        // 提取边界框坐标
        let bbox = &obj.estimate;
        let x1 = bbox[(0, 0)];
        let y1 = bbox[(0, 1)];
        let x2 = bbox[(0, 2)];
        let y2 = bbox[(0, 3)];
        
        // 相机画面坐标
        let frame_cx = (x1 + x2) / 2.0;
        let frame_cy = (y1 + y2) / 2.0;
        self.subject.frame_position = (frame_cx, frame_cy);
        
        // 对象大小
        let width = (x2 - x1).abs();
        let height = (y2 - y1).abs();
        self.subject.size = (width, height);
        
        // 速度估计 (基于过去检测历史)
        if obj.past_detections.len() >= 2 {
            if let (Some(latest), Some(oldest)) = 
                (obj.past_detections.back(), obj.past_detections.front()) 
            {
                let dt = (obj.age as f64) / 30.0;  // 假设30fps
                if dt > 0.0 {
                    self.subject.velocity = (
                        (latest.points[(0, 0)] - oldest.points[(0, 0)]) / dt,
                        (latest.points[(0, 1)] - oldest.points[(0, 1)]) / dt,
                    );
                }
            }
        }
        
        // 置信度 (基于hit_counter和age)
        self.subject.confidence = (obj.hit_counter as f64).min(obj.age as f64) / 30.0;
        self.subject.last_update = Instant::now();
    }

    /// 计算焦距控制命令
    fn compute_focus_command(&self) -> FocusCommand {
        // 如果没有追踪对象，返回空命令
        if self.subject.primary_id.is_none() {
            return FocusCommand::default();
        }

        let (cx, cy) = self.subject.frame_position;
        let (width, height) = self.subject.size;
        let (vx, vy) = self.subject.velocity;

        // ========== 控制1: 平移 (Pan/Tilt) ==========
        // 目标: 保持主体在画面中心
        // 方法: PID控制
        
        let center_x = self.frame_width / 2.0;
        let center_y = self.frame_height / 2.0;
        
        let pan_error = cx - center_x;      // 水平偏差
        let tilt_error = cy - center_y;     // 竖直偏差
        
        let pan_speed = (pan_error * 0.02).clamp(-1.0, 1.0);   // P=0.02
        let tilt_speed = (tilt_error * 0.02).clamp(-1.0, 1.0);

        // ========== 控制2: 变焦 (Zoom) ==========
        // 目标: 维持对象大小恒定
        // 方法: 反馈控制
        
        let target_size = self.frame_width * 0.3;  // 目标: 帧宽的30%
        let current_size = (width + height) / 2.0;  // 当前平均大小
        
        let zoom_factor = if current_size > 0.0 {
            target_size / current_size
        } else {
            1.0
        };
        
        // 限制变焦速度避免抖动
        let zoom_speed = ((zoom_factor - 1.0) * 0.05).clamp(-0.5, 0.5);

        // ========== 控制3: 对焦 (Focus) ==========
        // 目标: 根据物体距离调整焦距
        // 方法: 基于边界框大小估计深度
        
        // 粗略估计: 假设固定大小物体
        // 距离 ∝ (总像素面积) / (物体像素面积)
        let estimated_distance = (self.frame_width * self.frame_height) 
            / (width * height).max(1.0);
        
        // 映射到焦距值 (0-1000)
        let focus_value = ((estimated_distance / 1000.0).min(1.0) * 1000.0) as u16;

        // ========== 控制4: 预测追踪 ==========
        // 目标: 提前补偿对象运动
        // 方法: 预测100ms后的位置
        
        let prediction_time = 0.1;  // 秒
        let predicted_x = cx + vx * prediction_time;
        let predicted_y = cy + vy * prediction_time;
        
        // 预测的平移校正
        let predict_pan = (predicted_x - center_x) * 0.02;
        let predict_tilt = (predicted_y - center_y) * 0.02;
        
        // 合并基础控制和预测控制
        let final_pan = (pan_speed + predict_pan * 0.3).clamp(-1.0, 1.0);
        let final_tilt = (tilt_speed + predict_tilt * 0.3).clamp(-1.0, 1.0);

        FocusCommand {
            pan: final_pan,
            tilt: final_tilt,
            zoom_speed,
            focus_value,
            tracking_id: self.subject.primary_id,
            confidence: self.subject.confidence,
            subject_position: self.subject.frame_position,
            subject_size: self.subject.size,
        }
    }
}
```

---

## 硬件集成

### 硬件控制接口

```rust
/// 硬件抽象接口
pub trait CameraHardware {
    fn apply_focus_command(&mut self, cmd: &FocusCommand) 
        -> Result<(), Box<dyn std::error::Error>>;
}

/// PTZ相机实现 (VISCA协议)
/// 支持: Sony EVI, Panasonic等制造商
pub struct PTZCamera {
    serial_port: Box<dyn serialport::SerialPort>,
    current_zoom: f64,
    current_focus: u16,
}

impl PTZCamera {
    pub fn new(port_name: &str) 
        -> Result<Self, Box<dyn std::error::Error>> 
    {
        let settings = serialport::SerialPortSettings {
            baud_rate: 9600,
            data_bits: serialport::DataBits::Eight,
            flow_control: serialport::FlowControl::None,
            parity: serialport::Parity::None,
            stop_bits: serialport::StopBits::One,
            timeout: std::time::Duration::from_millis(100),
        };
        
        let port = serialport::open_with_settings(port_name, &settings)?;
        
        Ok(Self {
            serial_port: Box::new(port),
            current_zoom: 1.0,
            current_focus: 500,
        })
    }
}

impl CameraHardware for PTZCamera {
    fn apply_focus_command(&mut self, cmd: &FocusCommand) 
        -> Result<(), Box<dyn std::error::Error>> 
    {
        // 平移/倾斜 - 只在显著变化时发送
        if cmd.pan.abs() > 0.01 || cmd.tilt.abs() > 0.01 {
            self.send_pan_tilt_command(cmd.pan, cmd.tilt)?;
        }

        // 变焦 - 累积缩放
        if cmd.zoom_speed.abs() > 0.01 {
            self.current_zoom = (self.current_zoom + cmd.zoom_speed)
                .clamp(1.0, 10.0);
            self.send_zoom_command(self.current_zoom)?;
        }

        // 对焦 - 只在差值超过阈值时发送
        if (cmd.focus_value as i32 - self.current_focus as i32).abs() > 10 {
            self.current_focus = cmd.focus_value;
            self.send_focus_command(cmd.focus_value)?;
        }

        Ok(())
    }
}

// VISCA命令实现
impl PTZCamera {
    fn send_pan_tilt_command(&mut self, pan: f64, tilt: f64) 
        -> Result<(), Box<dyn std::error::Error>> 
    {
        // 速度: -1.0 (最左) 到 +1.0 (最右)
        let pan_speed = ((pan.abs() * 24.0) as u8).clamp(1, 24);
        let tilt_speed = ((tilt.abs() * 20.0) as u8).clamp(1, 20);
        
        // VISCA格式: $81 $01 $06 $01 [PanSpeed] [TiltSpeed] $FF
        let cmd = vec![0x81, 0x01, 0x06, 0x01, pan_speed, tilt_speed, 0xFF];
        self.serial_port.write_all(&cmd)?;
        
        Ok(())
    }

    fn send_zoom_command(&mut self, zoom: f64) 
        -> Result<(), Box<dyn std::error::Error>> 
    {
        // 缩放: 1.0 (无变焦) 到 10.0 (最大变焦)
        // 映射到12位值: 0 (无) 到 4095 (最大)
        let zoom_value = ((zoom - 1.0) * 4000.0 / 9.0) as u16;
        let high_byte = ((zoom_value >> 8) & 0xFF) as u8;
        let low_byte = (zoom_value & 0xFF) as u8;
        
        // VISCA格式: $81 $01 $04 $47 [H] [L] $FF
        let cmd = vec![0x81, 0x01, 0x04, 0x47, high_byte, low_byte, 0xFF];
        self.serial_port.write_all(&cmd)?;
        
        Ok(())
    }

    fn send_focus_command(&mut self, focus: u16) 
        -> Result<(), Box<dyn std::error::Error>> 
    {
        // 对焦: 0 (无穷远) 到 1000 (最近)
        // 映射到12位值: 0 到 4095
        let focus_value = (focus as f64 * 4095.0 / 1000.0) as u16;
        let high_byte = ((focus_value >> 8) & 0xFF) as u8;
        let low_byte = (focus_value & 0xFF) as u8;
        
        // VISCA格式: $81 $01 $04 $48 [H] [L] $FF
        let cmd = vec![0x81, 0x01, 0x04, 0x48, high_byte, low_byte, 0xFF];
        self.serial_port.write_all(&cmd)?;
        
        Ok(())
    }
}
```

---

## 配置和调试

### 推荐参数

```rust
// 追踪器配置
config.hit_counter_max = 30;              // 保持活跃帧数 (20-50)
config.initialization_delay = 3;          // 初始化延迟 (2-5)
config.distance_threshold = 0.5;          // IoU阈值 (0.3-0.7)
config.detection_threshold = 0.5;         // 检测置信度 (0.4-0.7)
config.past_detections_length = 10;       // 历史检测数 (5-15)

// 控制参数
const PAN_P: f64 = 0.02;                  // 平移P系数 (0.01-0.05)
const TILT_P: f64 = 0.02;                 // 倾斜P系数 (0.01-0.05)
const ZOOM_GAIN: f64 = 0.05;              // 变焦增益 (0.02-0.1)
const PREDICTION_TIME: f64 = 0.1;         // 预测时间 (50-200ms)
const TARGET_SIZE_RATIO: f64 = 0.3;       // 目标大小比例 (0.2-0.5)
```

### 调试输出

```rust
fn print_debug_info(cmd: &FocusCommand) {
    println!("=== 焦距命令 ===");
    println!("ID: {:?}", cmd.tracking_id);
    println!("置信度: {:.2}%", cmd.confidence * 100.0);
    println!("位置: ({:.0}, {:.0})", 
        cmd.subject_position.0, cmd.subject_position.1);
    println!("大小: {:.0} x {:.0}", 
        cmd.subject_size.0, cmd.subject_size.1);
    println!("平移: Pan={:+.3}, Tilt={:+.3}", cmd.pan, cmd.tilt);
    println!("变焦: {:.3}x", 1.0 + cmd.zoom_speed);
    println!("对焦: {}", cmd.focus_value);
}
```

---

## 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 检测延迟 | 50-100ms | YOLO推理 (取决于模型大小) |
| 追踪延迟 | 5-10ms | norfair更新 |
| 控制计算 | 2-5ms | PID + 预测 |
| 硬件通信 | 1-2ms | 串口发送 |
| **总延迟** | **60-120ms** | ~6-12帧 @60fps |
| **纯追踪吞吐** | **1000+ fps** | norfair-rs仅追踪 |

### 优化建议

1. **使用轻量级检测模型** (nano/small而非large)
2. **调整图像分辨率** (640x480而非1920x1080)
3. **启用GPU加速** (CUDA/CoreML)
4. **异步处理** 将检测和追踪放在不同线程
5. **降低帧率** (30fps而非60fps)
6. **减少历史长度** `past_detections_length = 5`

---

## 常见问题

### Q1: 追踪ID频繁变化怎么办？

**A:** 增加 `hit_counter_max` 和 `initialization_delay`
```rust
config.hit_counter_max = 50;           // 从30增加到50
config.initialization_delay = 5;       // 从3增加到5
```

### Q2: 追踪延迟太高？

**A:** 检查以下几点：
- YOLO延迟是否过高 (改用nano模型)
- 是否启用GPU (应该启用)
- 图像分辨率是否过高 (改用640x480)

### Q3: 自动追焦振荡/抖动？

**A:** 减少PID增益或添加死区
```rust
const PAN_P: f64 = 0.01;               // 从0.02减少到0.01
const DEAD_ZONE: f64 = 5.0;            // 仅在偏差>5像素时响应
```

### Q4: 无法连接相机硬件？

**A:** 检查串口设置
```bash
# Linux
ls /dev/ttyUSB*
screen /dev/ttyUSB0 9600

# Windows  
mode COM1 BAUD=9600 PARITY=N
```

### Q5: 多对象时如何选择追踪目标？

**A:** 当前实现选择最大对象。可改为：
```rust
// 选择最接近上次位置的
let best = tracked_objects
    .iter()
    .min_by_key(|obj| {
        let dist = (obj.estimate[(0, 0)] - last_x).powi(2)
                 + (obj.estimate[(0, 1)] - last_y).powi(2);
        dist as i32
    });
```

---

## 许可证

BSD 3-Clause License (同norfair-rs)

## 参考资源

- [norfair-rs GitHub](https://github.com/nmichlo/norfair-rs)
- [norfair原始项目](https://github.com/tryolabs/norfair)
- [VISCA协议文档](https://en.wikipedia.org/wiki/VISCA)
- [OpenCV光流教程](https://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html)

---

**最后更新**: 2024年  
**维护者**: norfair社区  
**贡献者**: 欢迎PR
