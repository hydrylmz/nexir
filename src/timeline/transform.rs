// src/timeline/transform.rs

use bytemuck::{Pod, Zeroable};

/// 2D affine transform parameters for a clip.
/// Must stay exactly 32 bytes (8 × f32) for GPU upload.
/// repr(C) + Pod guarantees: safe to memcpy into a wgpu buffer.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct ClipTransform {
    /// Canvas-space translation of the anchor point, in pixels.
    /// (0,0) = centre of canvas.
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

    /// Decompose into the 3x3 column-major affine matrix that the GPU vertex shader expects.
    /// The matrix maps from clip UV coordinates [0..1] to NDC coordinates [-1..1].
    pub fn to_matrix(&self, clip_w: f32, clip_h: f32, canvas_w: f32, canvas_h: f32) -> [f32; 9] { 
        // Negate rotation to make positive values rotate counter-clockwise (since Y is down)
        let (sin_r, cos_r) = (-self.rotation).sin_cos();

        // Scale factors
        let sx = self.scale[0];
        let sy = self.scale[1];

        // Anchor offsets in pixels
        let ax = self.anchor[0] * clip_w;
        let ay = self.anchor[1] * clip_h;

        // Translation in canvas pixels (0,0 is center of canvas)
        let tx = self.position[0] + canvas_w * 0.5;
        let ty = self.position[1] + canvas_h * 0.5;

        // Matrix mapping UV [0..1] to canvas pixels
        let m00 = sx * cos_r * clip_w;
        let m10 = sx * sin_r * clip_w;
        
        let m01 = -sy * sin_r * clip_h;
        let m11 =  sy * cos_r * clip_h;
        
        let m02 = tx - (ax * sx * cos_r - ay * sy * sin_r);
        let m12 = ty - (ax * sx * sin_r + ay * sy * cos_r);

        // Convert canvas pixels to NDC
        let to_ndc_x = 2.0 / canvas_w;
        let to_ndc_y = -2.0 / canvas_h;

        [
            m00 * to_ndc_x,       m10 * to_ndc_y,       0.0,
            m01 * to_ndc_x,       m11 * to_ndc_y,       0.0,
            m02 * to_ndc_x - 1.0, m12 * to_ndc_y + 1.0, 1.0,
        ]
    }
    
}

/// Compositing blend modes – GPU fragment shader interprets the `blend_mode` u32 field.
/// Values must match `BLEND_*` constants in the composite WGSL shader.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[repr(u32)]
pub enum BlendMode {
    /// Standard alpha compositing (Porter-Duff Over). Default.
    #[default]
    Normal = 0,
    /// dst + src
    Add = 1,
    /// src * dst
    Multiply = 2,
    /// 1 - (1-src)*(1-dst)
    Screen = 3,
    /// Multiply if dst < 0.5, Screen otherwise
    Overlay = 4,
    /// min(src, dst)
    Darken = 5,
    /// max(src, dst)
    Lighten = 6,
    /// Dodge: dst / (1 - src)
    ColorDodge = 7,
    /// Burn: 1 - (1 - dst) / src
    ColorBurn = 8,
    /// Hard Light: Multiply/Screen swap of Overlay
    HardLight = 9,
    /// Soft Light: Pegtop formula
    SoftLight = 10,
    /// |src - dst|
    Difference = 11,
    /// src + dst - 2*src*dst
    Exclusion = 12,
}

impl BlendMode {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => BlendMode::Normal,
            1 => BlendMode::Add,
            2 => BlendMode::Multiply,
            3 => BlendMode::Screen,
            4 => BlendMode::Overlay,
            5 => BlendMode::Darken,
            6 => BlendMode::Lighten,
            7 => BlendMode::ColorDodge,
            8 => BlendMode::ColorBurn,
            9 => BlendMode::HardLight,
            10 => BlendMode::SoftLight,
            11 => BlendMode::Difference,
            12 => BlendMode::Exclusion,
            _ => BlendMode::Normal,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Add => "Add",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::HardLight => "Hard Light",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::Difference => "Difference",
            BlendMode::Exclusion => "Exclusion",
        }
    }

    /// All blend modes in display order.
    pub fn all() -> &'static [BlendMode] {
        &[
            BlendMode::Normal, BlendMode::Add, BlendMode::Multiply,
            BlendMode::Screen, BlendMode::Overlay, BlendMode::Darken,
            BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
            BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference,
            BlendMode::Exclusion,
        ]
    }
}

/// Axis-aligned crop rectangle in normalised clip space [0..1].
/// (0,0)=top-left, (1,1)=bottom-right.
/// `feather` controls a soft gradient at the crop edge (0 = hard, 1 = full clip width feather).
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CropRect {
    pub left:    f32,
    pub top:     f32,
    pub right:   f32,
    pub bottom:  f32,
    pub feather: f32,
}

impl Default for CropRect {
    fn default() -> Self {
        Self { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0, feather: 0.0 }
    }
}

impl CropRect {
    /// No-op crop (full clip visible).
    pub fn full() -> Self {
        Self::default()
    }

    /// True if this is a full-clip no-op crop.
    pub fn is_identity(&self) -> bool {
        self.left < f32::EPSILON
            && self.top < f32::EPSILON
            && (self.right - 1.0).abs() < f32::EPSILON
            && (self.bottom - 1.0).abs() < f32::EPSILON
            && self.feather < f32::EPSILON
    }

    /// Clamps all values to [0..1] and ensures left < right, top < bottom.
    pub fn normalise(&mut self) {
        self.left = self.left.clamp(0.0, 1.0);
        self.right = self.right.clamp(0.0, 1.0);
        self.top = self.top.clamp(0.0, 1.0);
        self.bottom = self.bottom.clamp(0.0, 1.0);
        self.feather = self.feather.clamp(0.0, 1.0);
        if self.left > self.right { std::mem::swap(&mut self.left, &mut self.right); }
        if self.top > self.bottom { std::mem::swap(&mut self.top, &mut self.bottom); }
    }

    /// Returns the alpha multiplier at a given normalised clip UV, applying feather.
    pub fn alpha_at(&self, u: f32, v: f32) -> f32 {
        if self.feather < f32::EPSILON {
            // Hard crop
            if u >= self.left && u <= self.right && v >= self.top && v <= self.bottom {
                1.0
            } else {
                0.0
            }
        } else {
            let half = self.feather * 0.5;
            let al = ((u - self.left) / half).clamp(0.0, 1.0);
            let ar = ((self.right - u) / half).clamp(0.0, 1.0);
            let at = ((v - self.top) / half).clamp(0.0, 1.0);
            let ab = ((self.bottom - v) / half).clamp(0.0, 1.0);
            al.min(ar).min(at).min(ab)
        }
    }

    /// Pack into 4 GPU floats: [left, top, right, bottom] (feather handled separately).
    pub fn to_gpu(&self) -> [f32; 4] {
        [self.left, self.top, self.right, self.bottom]
    }
}

/// 4-point perspective warp (Corner Pin) in normalised clip space.
/// Each corner is (u, v) in [0..1] clip UV space.
/// Identity = corners at (0,0), (1,0), (0,1), (1,1).
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CornerPin {
    pub top_left:     [f32; 2],
    pub top_right:    [f32; 2],
    pub bottom_left:  [f32; 2],
    pub bottom_right: [f32; 2],
}

impl Default for CornerPin {
    fn default() -> Self {
        Self {
            top_left:     [0.0, 0.0],
            top_right:    [1.0, 0.0],
            bottom_left:  [0.0, 1.0],
            bottom_right: [1.0, 1.0],
        }
    }
}

impl CornerPin {
    pub fn identity() -> Self { Self::default() }

    pub fn is_identity(&self) -> bool {
        let id = Self::identity();
        self == &id
    }

    /// Pack corners into 8 floats for GPU upload (TL, TR, BL, BR each as [u,v]).
    pub fn to_gpu(&self) -> [f32; 8] {
        [
            self.top_left[0], self.top_left[1],
            self.top_right[0], self.top_right[1],
            self.bottom_left[0], self.bottom_left[1],
            self.bottom_right[0], self.bottom_right[1],
        ]
    }
}

/// Track-matte type for a clip.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum MatteMode {
    #[default]
    /// No matte applied.
    None,
    /// Use the alpha channel of the track above as the matte.
    AlphaMatte,
    /// Invert the alpha channel of the track above as the matte.
    AlphaMatteInverted,
    /// Use the luminance of the track above as the matte.
    LumaMatte,
    /// Invert the luminance of the track above as the matte.
    LumaMatteInverted,
}

impl MatteMode {
    pub fn label(self) -> &'static str {
        match self {
            MatteMode::None => "None",
            MatteMode::AlphaMatte => "Alpha Matte",
            MatteMode::AlphaMatteInverted => "Alpha Matte Inverted",
            MatteMode::LumaMatte => "Luma Matte",
            MatteMode::LumaMatteInverted => "Luma Matte Inverted",
        }
    }

    /// Evaluates the matte multiplier (in [0.0, 1.0]) given the RGBA color of the matte frame.
    pub fn evaluate_matte(&self, matte_rgba: [f32; 4]) -> f32 {
        match self {
            MatteMode::None => 1.0,
            MatteMode::AlphaMatte => matte_rgba[3].clamp(0.0, 1.0),
            MatteMode::AlphaMatteInverted => (1.0 - matte_rgba[3]).clamp(0.0, 1.0),
            MatteMode::LumaMatte => {
                let luma = 0.2126 * matte_rgba[0] + 0.7152 * matte_rgba[1] + 0.0722 * matte_rgba[2];
                luma.clamp(0.0, 1.0)
            }
            MatteMode::LumaMatteInverted => {
                let luma = 0.2126 * matte_rgba[0] + 0.7152 * matte_rgba[1] + 0.0722 * matte_rgba[2];
                (1.0 - luma).clamp(0.0, 1.0)
            }
        }
    }
}

/// Blends a single colour channel using the specified blend mode.
/// Both `src` and `dst` are expected in [0.0, 1.0].
pub fn blend_channel(mode: BlendMode, src: f32, dst: f32) -> f32 {
    match mode {
        BlendMode::Normal => src,
        BlendMode::Add => (src + dst).min(1.0),
        BlendMode::Multiply => src * dst,
        BlendMode::Screen => 1.0 - (1.0 - src) * (1.0 - dst),
        BlendMode::Overlay => {
            if dst < 0.5 {
                2.0 * src * dst
            } else {
                1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
            }
        }
        BlendMode::Darken => src.min(dst),
        BlendMode::Lighten => src.max(dst),
        BlendMode::ColorDodge => {
            if src >= 1.0 {
                1.0
            } else {
                (dst / (1.0 - src)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if src <= 0.0 {
                0.0
            } else {
                (1.0 - (1.0 - dst) / src).max(0.0)
            }
        }
        BlendMode::HardLight => {
            if src < 0.5 {
                2.0 * src * dst
            } else {
                1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
            }
        }
        BlendMode::SoftLight => {
            (1.0 - 2.0 * src) * dst * dst + 2.0 * src * dst
        }
        BlendMode::Difference => (src - dst).abs(),
        BlendMode::Exclusion => src + dst - 2.0 * src * dst,
    }
}

/// Blends an RGBA source pixel over an RGBA destination pixel.
pub fn blend_pixel(mode: BlendMode, src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let src_a = src[3];
    let dst_a = dst[3];
    let out_a = src_a + dst_a * (1.0 - src_a);

    if out_a <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut out_rgb = [0.0; 3];
    for i in 0..3 {
        let blended = blend_channel(mode, src[i], dst[i]);
        // Standard Porter-Duff compositing with non-separable blend modes
        out_rgb[i] = (src_a * (1.0 - dst_a) * src[i]
            + dst_a * (1.0 - src_a) * dst[i]
            + src_a * dst_a * blended)
            / out_a;
    }

    [out_rgb[0], out_rgb[1], out_rgb[2], out_a]
}

/// Geometric mask shapes for clip masking.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GeometricMask {
    Rectangle(CropRect),
    Ellipse {
        /// Center in normalized [0..1] UV coordinates.
        center: [f32; 2],
        /// Radius X and Y in normalized coordinates.
        radius: [f32; 2],
        /// Feather radius in normalized coordinates.
        feather: f32,
    },
    Polygon {
        /// Polygon vertices in normalized UV space.
        points: Vec<[f32; 2]>,
        /// Feather factor.
        feather: f32,
    },
}

impl GeometricMask {
    /// Computes the alpha mask value in [0.0, 1.0] at a normalized coordinate (u, v).
    pub fn alpha_at(&self, u: f32, v: f32) -> f32 {
        match self {
            GeometricMask::Rectangle(rect) => rect.alpha_at(u, v),
            GeometricMask::Ellipse { center, radius, feather } => {
                if radius[0] <= 0.0 || radius[1] <= 0.0 {
                    return 0.0;
                }
                let dx = (u - center[0]) / radius[0];
                let dy = (v - center[1]) / radius[1];
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt();
                if *feather < f32::EPSILON {
                    if dist <= 1.0 { 1.0 } else { 0.0 }
                } else {
                    let half_f = *feather * 0.5;
                    ((1.0 + half_f - dist) / *feather).clamp(0.0, 1.0)
                }
            }
            GeometricMask::Polygon { points, feather: _ } => {
                if points.len() < 3 {
                    return 1.0;
                }
                // Ray casting algorithm for point in polygon
                let mut inside = false;
                let mut j = points.len() - 1;
                for i in 0..points.len() {
                    let pi = points[i];
                    let pj = points[j];
                    if ((pi[1] > v) != (pj[1] > v))
                        && (u < (pj[0] - pi[0]) * (v - pi[1]) / (pj[1] - pi[1]) + pi[0])
                    {
                        inside = !inside;
                    }
                    j = i;
                }
                if inside { 1.0 } else { 0.0 }
            }
        }
    }
}

/// Fixed-size GPU-uploadable effect parameter block.
/// 64 bytes = 16 × f32 = one cache line.
/// The interpretation of `data` depends on the `EffectKind` of the owning effect.
/// The WGSL shader reads this as a `uniform` block of 16 floats.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable, serde::Serialize, serde::Deserialize)]
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

/// Parameters for all built-in GPU effects applied to a clip.
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClipEffects {
    #[serde(default)]
    pub color_enabled: bool,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,

    pub blur_enabled: bool,
    pub blur_radius: f32,
    pub blur_sigma: f32,

    pub sharpen_enabled: bool,
    pub sharpen_amount: f32,

    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
    pub vignette_roundness: f32,

    pub chroma_key_enabled: bool,
    pub chroma_key_color: [f32; 3],
    pub chroma_key_tolerance: f32,
    pub chroma_key_softness: f32,
}

impl Default for ClipEffects {
    fn default() -> Self {
        Self {
            color_enabled: false,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            blur_enabled: false,
            blur_radius: 10.0,
            blur_sigma: 5.0,
            sharpen_enabled: false,
            sharpen_amount: 0.5,
            vignette_enabled: false,
            vignette_intensity: 0.5,
            vignette_radius: 0.75,
            vignette_softness: 0.45,
            vignette_roundness: 1.0,
            chroma_key_enabled: false,
            chroma_key_color: [0.0, 1.0, 0.0],
            chroma_key_tolerance: 0.3,
            chroma_key_softness: 0.1,
        }
    }
}