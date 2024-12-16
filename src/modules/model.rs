use anyhow::Result;
use polars::prelude::*;
use xgboost::{parameters, Booster, DMatrix};

pub fn train_model(
    x_train_df: &DataFrame,
    y_train_df: &DataFrame,
    x_test_df: &DataFrame,
    y_test_df: &DataFrame,
) -> Result<String> {
    // Transform Polars DataFrames into 2D arrays in row-major order
    let x_train = x_train_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_train = y_train_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let x_test = x_test_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_test = y_test_df.to_ndarray::<Float32Type>(IndexOrder::C)?;

    // Convert training and testing sets to XGBoost DMatrix objects for evaluation
    let mut dtrain = DMatrix::from_dense(&x_train.clone().into_raw_vec(), x_train.nrows())?;
    let mut dtest = DMatrix::from_dense(&x_test.clone().into_raw_vec(), x_test.nrows())?;

    dtrain.set_labels(
        y_train
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Training array not contiguous"))?,
    )?;

    dtest.set_labels(
        y_test
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Testing array not contiguous"))?,
    )?;

    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // Specify overall training setup
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .unwrap();

    // Train model, and print evaluation data
    let bst = Booster::train(&training_params).unwrap();
    println!("Test {:?}", bst.predict(&dtest).unwrap());

    // Save model to disk
    let model_path = "./output/models/model.bin";
    bst.save(model_path)?;
    println!("Model saved to {}", model_path);

    Ok(model_path.to_string())
}
