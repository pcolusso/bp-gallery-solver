use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use anyhow::Result;
use soa_derive::StructOfArray;
use std::borrow::Cow;
use std::io::{BufRead, BufReader};
use std::fs::File;

#[derive(StructOfArray)]
struct Word {
    len: usize,
    chars: String
}


fn main() -> Result<()> {
    #[cfg(target_os="macos")]
    {
        println!("MPS Ready: {}", tch::utils::has_mps());
    }
    

    #[cfg(target_os="windows")]
    println!("CUDA Ready: {}", tch::utils::has_cuda());

    let mut dict = WordVec::new();

    let file = File::open("words_alpha.txt")?;
    let reader = BufReader::new(file);

    for w in reader.lines() {
        let w = w?;
        dict.push(Word { len: w.len(), chars: w });
    }

    let matches: Vec<Cow<str>> = dict.iter()
        .filter( |w| *w.len == 6 )
        .filter( |w| ['b', 'c', 'p', 't', 'l', 's', 'm', 'w'].contains(&w.chars.chars().nth(0).unwrap()))
        .filter( |w| ['a', 'e', 'i', 'o', 'u', 'h', 'r', 't'].contains(&w.chars.chars().nth(1).unwrap()))
        .filter( |w| ['o', 'i', 'v', 'l', 'r', 'n', 't', 's'].contains(&w.chars.chars().nth(2).unwrap()))
        .filter( |w| ['i', 'k', 'v',    'a', 'd', 't', 'h', 's' ].contains(&w.chars.chars().nth(3).unwrap()))
        .filter( |w| ['a', 'e', 'u', 'w', 'l', 'r', 'c', 's' ].contains(&w.chars.chars().nth(4).unwrap()))
        .filter( |w| ['n', 'd', 'g', 'r', 'w', 't', 'e', 'y' ].contains(&w.chars.chars().nth(5).unwrap()))
        .map(|w| Cow::from(w.chars) ).collect();


    let seed = "thought";
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased).create_model()?;

    let mut all_words = vec![seed.to_string()];
    all_words.extend(matches.iter().cloned().map(String::from));

    let embeddings = model.encode(&all_words)?;

    let seed_embedding = embeddings.get(0).unwrap();

    let mut scores: Vec<(Cow<str>, f32)> = matches
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let sim = cosine_similarity(seed_embedding, &embeddings[i + 1]);
            (word.clone(), sim)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (word, score) in scores.iter().take(10) {
        println!("{}: {:.3}", word, score);
    }

    Ok(())
}

// chatgpt generated ahh garbage, prolly need to double check
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}