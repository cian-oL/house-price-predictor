use anyhow::Result;
use polars::prelude::*;
use rand::prelude::*;
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

pub fn split_train_test(df: &DataFrame, test_size_percent: f64) -> Result<(DataFrame, DataFrame)> {
    if test_size_percent <= 0.0 || test_size_percent >= 1.0 {
        return Err(anyhow::anyhow!(PolarsError::ComputeError(
            "test_size_percent must be between 0 and 1".into(),
        )));
    }

    // Create random index range to split data
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..df.height()).collect();
    indices.shuffle(&mut rng);

    // Split data based on desired test percentage of df
    let test_size = (test_size_percent * df.height() as f64) as usize;
    let (test_indices, train_indices) = indices.split_at(test_size);

    // Convert to ChunkedArray<Int32Type>
    let train_indices_ca = UInt32Chunked::from_vec(
        "training",
        train_indices.iter().map(|&x| x as u32).collect(),
    );
    let test_indices_ca =
        UInt32Chunked::from_vec("testing", test_indices.iter().map(|&x| x as u32).collect());

    let train_df = df.take(&train_indices_ca)?;
    let test_df = df.take(&test_indices_ca)?;

    println!("Training set size: {}", train_df.height());
    println!("Testing set size: {}", test_df.height());

    Ok((train_df, test_df))
}

pub fn split_features_target(df: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let features = df.select([
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b",
        "lstat",
    ])?;

    let target = df.select(["medv"])?;

    Ok((features, target))
}
