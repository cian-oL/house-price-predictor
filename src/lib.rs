use anyhow::Result;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;

pub fn download_dataset(dataset_url: &str, output_file: &str) -> Result<()> {
    println!("Downloading Boston Housing dataset...");

    let response = reqwest::blocking::get(dataset_url)?;
    let data = response.text()?;

    let mut file = File::create(output_file)?;
    file.write_all(data.as_bytes())?;

    println!("Dataset downloaded successfully to {}", output_file);
    Ok(())
}

pub fn load_csv_file(file_path: &str) -> Result<DataFrame> {
    let df = LazyCsvReader::new(file_path).finish()?.collect()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    print!("First 5 rows:\n{:#?}\n", df.head(Some(5)));

    Ok(df)
}
