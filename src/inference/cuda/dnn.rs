//! Optional cuDNN convolution plans

use super::geometry::{Conv2d, Residual};

mod library;
pub(crate) use library::{ConvPlan, ConvPlanner};
