use std::path::PathBuf;

use cargo_gpu_install::{
    install::Install,
    spirv_builder::{Capability, ShaderPanicStrategy, SpirvMetadata},
};

pub fn main() -> anyhow::Result<()> {
    let crate_path = PathBuf::from("./shaders");

    let install = Install::from_shader_crate(crate_path.clone())
        .within_build_script()
        .run()?;
    let mut builder = install
        .to_spirv_builder(crate_path, "spirv-unknown-vulkan1.3")
        .capability(Capability::ImageQuery)
        .capability(Capability::RuntimeDescriptorArray);
    builder.build_script.defaults = true;
    builder.shader_panic_strategy = ShaderPanicStrategy::SilentExit;
    builder.spirv_metadata = SpirvMetadata::Full;

    let compile_result = builder.build()?;
    let spv_path = compile_result.module.unwrap_single();
    println!("cargo::rustc-env=SHADERS={}", spv_path.display());
    Ok(())
}
