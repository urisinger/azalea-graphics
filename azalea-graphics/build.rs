use std::path::PathBuf;

use cargo_gpu::spirv_builder::{Capability, MetadataPrintout, SpirvMetadata};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shader_crate = PathBuf::from("./shaders");

    let backend = cargo_gpu::Install::from_shader_crate(shader_crate.clone()).run()?;

    let builder = backend
        .to_spirv_builder(shader_crate, "spirv-unknown-vulkan1.2")
        .capability(Capability::ImageQuery)
        .print_metadata(MetadataPrintout::DependencyOnly)
        .spirv_metadata(SpirvMetadata::Full);

    let spv_result = builder.build()?;
    let path_to_spv = spv_result.module.unwrap_single();

    println!("cargo::rustc-env=SHADERS={}", path_to_spv.display());

    Ok(())
}
