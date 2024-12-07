use anyhow::Result;
use std::fs::File;
use std::io::Write;

const DATASET_URL: &str =
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
const OUTPUT_FILE: &str = "./data/boston_housing.csv";

fn download_dataset() -> Result<()> {
    println!("Downloading Boston Housing dataset...");

    // Download the CSV data
    let response = reqwest::blocking::get(DATASET_URL)?;
    let data = response.text()?;

    // Save to file
    let mut file = File::create(OUTPUT_FILE)?;
    file.write_all(data.as_bytes())?;

    println!("Dataset downloaded successfully to {}", OUTPUT_FILE);
    Ok(())
}

fn main() -> Result<()> {
    download_dataset()?;
    Ok(())
}
