use rust_bert::distilbert::DistilBertModelResources;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use tch::kind::Kind;
use anyhow::Result;


fn main() -> Result<()> {
    println!("MPS Ready: {}", tch::utils::has_mps());

    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased).create_model()?;
    let root = "cat";
    let candidates = ["dog", "car", "kitten", "animal", "fish"];

    let mut all_words = vec![root.to_string()];
    all_words.extend(candidates.iter().cloned().map(String::from));

    let embeddings = model.encode(&all_words)?;

    dbg!(embeddings);

    println!("Hello, world!");
    Ok(())
}
