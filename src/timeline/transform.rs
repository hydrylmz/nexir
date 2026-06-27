// src/timeline/transform.rs

use bytemuck::{Pod, Zeroable};

/// 2D affine transform parameters for a clip.
/// Must stay exactly 32 bytes (8 × f32) for GPU upload.
/// repr(C) + Pod guarantees: safe to memcpy into a wgpu buffer.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct ClipTransform {
    /// Canvas-space translation of the anchor point, in pixels.
    /// (0,0) = top-left of canvas.
    pub position: [f32; 2],

    /// Scale multipliers. (1.0, 1.0) = original size.
    /// Negative values flip the axis (horizontal/vertical mirror).
    pub scale: [f32; 2],

    /// Rotation in radians, counterclockwise.
    /// Range: any f32, typically (-PI, PI].
    pub rotation: f32,

    /// Anchor point in normalised clip space: (0,0)=top-left, (1,1)=bottom-right.
    /// (0.5, 0.5) = centre — the default for most clips.
    pub anchor: [f32; 2],

    /// Explicit padding to reach 32 bytes and satisfy WGSL vec4 alignment.
    pub _pad: f32,
}

impl ClipTransform {
    /// Identity transform: no translation, no scale change, no rotation, centre anchor.
    ///
    /// # Values
    /// position = [0.0, 0.0]
    /// scale    = [1.0, 1.0]
    /// rotation = 0.0
    /// anchor   = [0.5, 0.5]  ← centre of clip
    /// _pad     = 0.0
    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            anchor: [0.5, 0.5],
            _pad: 0.0,
        }
    }

    /// Returns true if this transform is the identity (within f32 epsilon).
    pub fn is_identity(&self) -> bool {
        let id = Self::identity();
        (self.position[0] - id.position[0]).abs() < f32::EPSILON &&
        (self.position[1] - id.position[1]).abs() < f32::EPSILON &&
        (self.scale[0] - id.scale[0]).abs() < f32::EPSILON &&
        (self.scale[1] - id.scale[1]).abs() < f32::EPSILON &&
        (self.rotation - id.rotation).abs() < f32::EPSILON &&
        (self.anchor[0] - id.anchor[0]).abs() < f32::EPSILON &&
        (self.anchor[1] - id.anchor[1]).abs() < f32::EPSILON
    }

    /// Decompose into the 3×3 column-major affine matrix that the GPU vertex shader expects.
    /// The full transform is: T * R * S (scale first, then rotate, then translate).
    pub fn to_matrix(&self, clip_w: f32, clip_h: f32) -> [f32; 9] { 
        let (sin_r, cos_r) = self.rotation.sin_cos();

        // Scale factors
        let sx = self.scale[0];
        let sy = self.scale[1];

        // Anchor offsets in pixels
        let ax = self.anchor[0] * clip_w;
        let ay = self.anchor[1] * clip_h;

        // Translation components
        let tx = self.position[0];
        let ty = self.position[1];

        // Combined 2D affine transformation elements (M = T * R * S * pre_T)
        // Column 0
        let m00 = cos_r * sx;
        let m10 = sin_r * sx;
        let m20 = 0.0;

        // Column 1
        let m01 = -sin_r * sy;
        let m11 = cos_r * sy;
        let m21 = 0.0;

        // Column 2 (Translation column, accounts for the pre-translation anchor shift)
        let m02 = tx - (ax * m00 + ay * m01);
        let m12 = ty - (ax * m10 + ay * m11);
        let m22 = 1.0;

        // Return in column-major order as expected by WGSL mat3x3
        [
            m00, m10, m20, // Column 0
            m01, m11, m21, // Column 1
            m02, m12, m22, // Column 2
        ]
    }
    
}

/// Fixed-size GPU-uploadable effect parameter block.
/// 64 bytes = 16 × f32 = one cache line.
/// The interpretation of `data` depends on the `EffectKind` of the owning effect.
/// The WGSL shader reads this as a `uniform` block of 16 floats.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct EffectParams {
    /// Raw float parameters. Layout is per-effect:
    ///   ColorCorrection: data[0..3]=lift, data[4..6]=gamma, data[8..10]=gain, data[12]=saturation
    ///   GaussianBlur:    data[0]=radius_px, data[1]=sigma
    ///   ChromaKey:       data[0..2]=key_color(hue,sat), data[3]=tolerance, data[4]=softness
    ///   Transform2D:     (not used — ClipTransform handles this)
    ///   Custom:          user-defined layout
    pub data: [f32; 16],
}

impl EffectParams {
    /// All-zero params — the neutral/identity value for most effects.
    pub fn zero() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    /// Read a named float from the param block by slot index.
    pub fn get_f32(&self, index: usize) -> Result<f32, ParamError> {
        if index < 16 {
            Ok(self.data[index])
        } else {
            Err(ParamError::IndexOutOfRange { index, max: 15 })
        }
    }

    /// Write a float to a named slot.
    pub fn set_f32(&mut self, index: usize, value: f32) -> Result<(), ParamError> {
        if index < 16 {
            self.data[index] = value;
            Ok(())
        } else {
            Err(ParamError::IndexOutOfRange { index, max: 15 })
        }
    }
}

#[derive(Debug)]
pub enum ParamError {
    IndexOutOfRange { index: usize, max: usize },
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::IndexOutOfRange { index, max } => {
                write!(f, "Parameter index {} out of range (max {})", index, max)
            }
        }

    }
}