//! Exact trunk batch classes; the dense head composes its own compiled classes

/// A graph shape supported by the embedding trunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingBatchClass {
    One,
    Four,
    Eight,
    Sixteen,
    ThirtyTwo,
}

impl EmbeddingBatchClass {
    pub(crate) const ALL: [Self; 5] = [
        Self::One,
        Self::Four,
        Self::Eight,
        Self::Sixteen,
        Self::ThirtyTwo,
    ];

    pub(crate) const fn chunks(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::ThirtyTwo => 32,
        }
    }

    pub(crate) const fn slot(self) -> usize {
        match self {
            Self::One => 0,
            Self::Four => 1,
            Self::Eight => 2,
            Self::Sixteen => 3,
            Self::ThirtyTwo => 4,
        }
    }

    /// Largest exact class that fits, without adding padded trunk work
    pub(crate) fn fitting(chunks: usize) -> Option<Self> {
        Self::ALL
            .into_iter()
            .rev()
            .find(|class| class.chunks() <= chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingBatchClass;

    #[test]
    fn exact_classes_partition_every_pipeline_batch() {
        assert_eq!(EmbeddingBatchClass::fitting(0), None);
        for chunks in 1..=32 {
            let mut remaining = chunks;
            let mut sizes = Vec::new();
            while let Some(class) = EmbeddingBatchClass::fitting(remaining) {
                sizes.push(class.chunks());
                remaining -= class.chunks();
            }
            assert_eq!(remaining, 0);
            assert_eq!(sizes.iter().sum::<usize>(), chunks);
            assert!(sizes.windows(2).all(|pair| pair[0] >= pair[1]));
        }
        let mut remaining = 31;
        let mut sizes = Vec::new();
        while let Some(class) = EmbeddingBatchClass::fitting(remaining) {
            sizes.push(class.chunks());
            remaining -= class.chunks();
        }
        assert_eq!(sizes, [16, 8, 4, 1, 1, 1]);
    }
}
