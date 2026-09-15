use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, ensure};

use speakrs::inference::EmbeddingModel;
use speakrs::pipeline::{EmbeddingStageSnapshot, ImportedDiarizationPipeline};

use super::domain::RecipeSpec;
use super::run::ValidatedRecording;

pub(crate) fn load_embedding_model(
    embedding_path: PathBuf,
    mode: speakrs::ExecutionMode,
) -> Result<EmbeddingModel> {
    Ok(EmbeddingModel::with_mode(embedding_path, mode)?)
}

pub(crate) fn run_embedding_stage<'a>(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    embedding_model: &'a mut EmbeddingModel,
    plda_dir: &Path,
) -> Result<(ImportedDiarizationPipeline<'a>, EmbeddingStageSnapshot)> {
    ensure_recipe_matches_bundle(recipe, recording)?;
    let mut pipeline =
        ImportedDiarizationPipeline::new(recording.bundle.clone(), embedding_model, plda_dir)?;
    let snapshot = pipeline.run_embedding_stage(&recording.samples)?;

    Ok((pipeline, snapshot))
}

pub(crate) fn ensure_recipe_matches_bundle(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
) -> Result<()> {
    let manifest = recording.bundle.manifest();
    ensure!(
        recipe.decoder.id == manifest.policy.decoder.id
            && recipe.decoder.revision == manifest.policy.decoder.revision,
        "recipe decoder does not match imported bundle policy"
    );
    ensure!(
        recipe.embedding.id == manifest.policy.embedding.id
            && recipe.embedding.revision == manifest.policy.embedding.revision,
        "recipe embedding does not match imported bundle policy"
    );
    ensure!(
        recipe.reconstruction.id == manifest.policy.reconstruction.id
            && recipe.reconstruction.revision == manifest.policy.reconstruction.revision,
        "recipe reconstruction does not match imported bundle policy"
    );

    Ok(())
}
