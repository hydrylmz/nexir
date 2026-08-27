// src/render/shader/registry.rs

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::render::device::GpuDevice;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BuiltinShader {
    YuvToRgb,
    Composite,
    CompositeSingle,
    Blit,
    ColorCorrection,
    ChromaKey,
    Lut3D,
    ToneMap,
}

pub struct ShaderRegistry {
    modules: RwLock<HashMap<BuiltinShader, Arc<wgpu::ShaderModule>>>,
}

impl ShaderRegistry {
    pub fn compile_all(device: &GpuDevice) -> Result<Self, ShaderError> {
        let mut map = HashMap::new();

        let sources: &[(BuiltinShader, &str)] = &[
            (BuiltinShader::YuvToRgb,        include_str!("yuv_to_rgb.wgsl")),
            (BuiltinShader::Composite,        include_str!("composite.wgsl")),
            (BuiltinShader::CompositeSingle,  include_str!("composite_single.wgsl")),
            (BuiltinShader::Blit,             include_str!("blit.wgsl")),
            (BuiltinShader::ColorCorrection,  include_str!("color_correction.wgsl")),
            (BuiltinShader::ChromaKey,        include_str!("chroma_key.wgsl")),
            (BuiltinShader::Lut3D,            include_str!("lut.wgsl")),
            (BuiltinShader::ToneMap,          include_str!("tonemap.wgsl")),
        ];

        for &(shader_id, src) in sources {
            device.device.push_error_scope(wgpu::ErrorFilter::Validation);

            let label = match shader_id {
                BuiltinShader::YuvToRgb        => "yuv_to_rgb_shader",
                BuiltinShader::Composite       => "composite_shader",
                BuiltinShader::CompositeSingle => "composite_single_shader",
                BuiltinShader::Blit            => "blit_shader",
                BuiltinShader::ColorCorrection => "color_correction_shader",
                BuiltinShader::ChromaKey       => "chroma_key_shader",
                BuiltinShader::Lut3D           => "lut_3d_shader",
                BuiltinShader::ToneMap         => "tonemap_shader",
            };


            let module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
            });

            if let Some(err) = pollster::block_on(device.device.pop_error_scope()) {
                return Err(ShaderError::CompilationFailed(format!(
                    "Failed to compile {:?}: {}", shader_id, err
                )));
            }

            map.insert(shader_id, Arc::new(module));
        }

        Ok(Self {
            modules: RwLock::new(map),
        })
    }

    pub fn get(&self, shader: BuiltinShader) -> Arc<wgpu::ShaderModule> {
        self.modules.read().unwrap().get(&shader).cloned()
            .unwrap_or_else(|| panic!("shader {:?} not compiled", shader))
    }

    pub fn hot_reload(
        &self,
        device:   &GpuDevice,
        shader:   BuiltinShader,
        new_wgsl: &str,
    ) -> Result<(), ShaderError> {
        device.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hot_reloaded_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(new_wgsl)),
        });

        if let Some(err) = pollster::block_on(device.device.pop_error_scope()) {
            return Err(ShaderError::CompilationFailed(format!(
                "Failed to hot-reload {:?}: {}", shader, err
            )));
        }

        let mut map = self.modules.write().unwrap();
        map.insert(shader, Arc::new(module));
        Ok(())
    }
}

#[derive(Debug)]
pub enum ShaderError {
    CompilationFailed(String),
    #[allow(dead_code)]
    ShaderNotFound(BuiltinShader),
}
