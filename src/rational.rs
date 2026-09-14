#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RationalError {
    NonPositiveDenominator,
    NonPositiveValue,
    NotReduced,
    NonPositiveDivisor,
    Negative,
    NotExact,
    Overflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckedRational {
    numerator: i128,
    denominator: i128,
}

impl CheckedRational {
    pub(crate) const fn integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    pub(crate) fn new(numerator: i128, denominator: i128) -> Result<Self, RationalError> {
        if denominator <= 0 {
            return Err(RationalError::NonPositiveDenominator);
        }
        let divisor = i128::try_from(gcd_u128(numerator.unsigned_abs(), denominator as u128))
            .map_err(|_| RationalError::Overflow)?;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub(crate) fn validate_parts(
        numerator: i128,
        denominator: i128,
        positive: bool,
    ) -> Result<(), RationalError> {
        if denominator <= 0 {
            return Err(RationalError::NonPositiveDenominator);
        }
        if positive && numerator <= 0 {
            return Err(RationalError::NonPositiveValue);
        }
        if gcd_u128(numerator.unsigned_abs(), denominator as u128) != 1 {
            return Err(RationalError::NotReduced);
        }
        Ok(())
    }

    pub(crate) fn checked_add(self, other: Self) -> Result<Self, RationalError> {
        let divisor = i128::try_from(gcd_u128(
            self.denominator as u128,
            other.denominator as u128,
        ))
        .map_err(|_| RationalError::Overflow)?;
        let left_factor = other.denominator / divisor;
        let right_factor = self.denominator / divisor;
        let numerator = self
            .numerator
            .checked_mul(left_factor)
            .and_then(|value| {
                other
                    .numerator
                    .checked_mul(right_factor)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .ok_or(RationalError::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or(RationalError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub(crate) fn checked_sub(self, other: Self) -> Result<Self, RationalError> {
        let numerator = other
            .numerator
            .checked_neg()
            .ok_or(RationalError::Overflow)?;
        self.checked_add(Self {
            numerator,
            denominator: other.denominator,
        })
    }

    pub(crate) fn checked_mul(self, multiplier: i128) -> Result<Self, RationalError> {
        let divisor = i128::try_from(gcd_u128(
            multiplier.unsigned_abs(),
            self.denominator as u128,
        ))
        .map_err(|_| RationalError::Overflow)?;
        let numerator = self
            .numerator
            .checked_mul(multiplier / divisor)
            .ok_or(RationalError::Overflow)?;
        let denominator = self.denominator / divisor;
        Self::new(numerator, denominator)
    }

    pub(crate) fn checked_div(self, divisor: Self) -> Result<Self, RationalError> {
        if divisor.numerator <= 0 {
            return Err(RationalError::NonPositiveDivisor);
        }

        let numerator_divisor = i128::try_from(gcd_u128(
            self.numerator.unsigned_abs(),
            divisor.numerator as u128,
        ))
        .map_err(|_| RationalError::Overflow)?;
        let denominator_divisor = i128::try_from(gcd_u128(
            self.denominator as u128,
            divisor.denominator as u128,
        ))
        .map_err(|_| RationalError::Overflow)?;
        let numerator = (self.numerator / numerator_divisor)
            .checked_mul(divisor.denominator / denominator_divisor)
            .ok_or(RationalError::Overflow)?;
        let denominator = (self.denominator / denominator_divisor)
            .checked_mul(divisor.numerator / numerator_divisor)
            .ok_or(RationalError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub(crate) const fn is_negative(self) -> bool {
        self.numerator < 0
    }

    pub(crate) fn floor_nonnegative(self) -> Result<i128, RationalError> {
        if self.is_negative() {
            return Err(RationalError::Negative);
        }
        Ok(self.numerator / self.denominator)
    }

    pub(crate) fn ceil_nonnegative(self) -> Result<i128, RationalError> {
        let floor = self.floor_nonnegative()?;
        if self.numerator % self.denominator == 0 {
            Ok(floor)
        } else {
            floor.checked_add(1).ok_or(RationalError::Overflow)
        }
    }

    pub(crate) fn round_ties_even(self) -> Result<i128, RationalError> {
        let floor = self.numerator.div_euclid(self.denominator);
        let remainder = self.numerator.rem_euclid(self.denominator);
        let complement = self.denominator - remainder;
        if remainder < complement {
            return Ok(floor);
        }
        if remainder > complement || floor % 2 != 0 {
            return floor.checked_add(1).ok_or(RationalError::Overflow);
        }
        Ok(floor)
    }

    pub(crate) fn to_usize_exact(self) -> Result<usize, RationalError> {
        if self.numerator < 0 || self.numerator % self.denominator != 0 {
            return Err(RationalError::NotExact);
        }
        usize::try_from(self.numerator / self.denominator).map_err(|_| RationalError::Overflow)
    }

    pub(crate) fn to_u64_exact(self) -> Result<u64, RationalError> {
        if self.numerator < 0 || self.numerator % self.denominator != 0 {
            return Err(RationalError::NotExact);
        }
        u64::try_from(self.numerator / self.denominator).map_err(|_| RationalError::Overflow)
    }

    pub(crate) fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}
