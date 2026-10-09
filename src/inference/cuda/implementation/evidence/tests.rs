use super::super::MEASURED_MODULES;
use crate::inference::cuda::kernels::LoadedArtifact;

/// A kernel rebuild changes the shipped cubin, and a binding still naming the old hash
/// refuses to load on its device; checked wherever this build embeds the bound tier
#[test]
fn measured_module_bindings_name_embedded_cubins() {
    let mut checked = 0;
    for binding in MEASURED_MODULES {
        let request = binding.module;
        if !request.tier().is_compiled_in() {
            continue;
        }

        let LoadedArtifact::Cubin { arch, sha256 } = request.artifact() else {
            panic!("an included measured cubin binding must name a cubin")
        };

        let ptx = request
            .area()
            .variants()
            .embedded(request.tier())
            .expect("an included bound tier must embed its PTX");
        let cubin = ptx
            .cubin(arch)
            .expect("an included bound tier must embed its bound cubin");
        checked += 1;

        assert_eq!(
            cubin.sha256(),
            sha256,
            "{:?} {:?} binding names a cubin this build does not embed",
            request.area(),
            arch
        );
    }

    if MEASURED_MODULES
        .iter()
        .any(|binding| binding.module.tier().is_compiled_in())
    {
        assert!(
            checked > 0,
            "an included bound tier must check at least one cubin"
        );
    }
}
