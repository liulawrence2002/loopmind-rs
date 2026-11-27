//! Burn tensors for batches produced by `loopmind-data`.
//!
//! The data crate stays free of Burn. Training code calls these helpers with
//! the same device the model uses.

use burn::tensor::{Device, Int, Tensor, TensorData};
use loopmind_data::{PretrainBatch, SftBatch};

pub fn pretrain_batch(batch: &PretrainBatch, device: &Device) -> Tensor<2, Int> {
    let values: Vec<i32> = batch.tokens.iter().copied().map(i32::from).collect();
    Tensor::<2, Int>::from_data(
        TensorData::new(values, [batch.batch, batch.seq_len]),
        device,
    )
}

pub fn sft_batch(batch: &SftBatch, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let tokens: Vec<i32> = batch.tokens.iter().copied().map(i32::from).collect();
    let mask: Vec<i32> = batch.mask.iter().copied().map(i32::from).collect();
    let shape = [batch.batch, batch.seq_len];
    (
        Tensor::<2, Int>::from_data(TensorData::new(tokens, shape), device),
        Tensor::<2, Int>::from_data(TensorData::new(mask, shape), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopmind_data::{write_pretrain_shard, write_sft_shard, PretrainLoader, SftLoader, SftRow};

    #[test]
    fn fixture_shards_become_burn_tensors() {
        let dir = std::env::temp_dir().join(format!("loopmind-burn-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let pretrain = dir.join("pretrain");
        let sft = dir.join("sft");
        std::fs::create_dir_all(&pretrain).unwrap();
        std::fs::create_dir_all(&sft).unwrap();
        write_pretrain_shard(
            &pretrain.join("a.bin"),
            4,
            &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        )
        .unwrap();
        write_sft_shard(
            &sft.join("sft-00000.bin"),
            &[SftRow {
                tokens: vec![9, 8, 7],
                mask: vec![0, 1, 1],
            }],
        )
        .unwrap();
        let device = Device::flex();
        let mut pretrain_loader = PretrainLoader::from_dir(&pretrain, 1).unwrap();
        let tokens = pretrain_batch(&pretrain_loader.sample(2).unwrap(), &device);
        assert_eq!(tokens.dims(), [2, 4]);
        let mut sft_loader = SftLoader::from_dir(&sft, 0, 1).unwrap();
        let (ids, mask) = sft_batch(&sft_loader.sample(1).unwrap(), &device);
        assert_eq!(ids.dims(), [1, 64]);
        assert_eq!(mask.dims(), [1, 64]);
    }
}
