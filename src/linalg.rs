//! Dense f64 linear algebra for the one-time PLDA setup
//!
//! PLDA needs two inverses and one generalized symmetric-definite eigensolve on 128x128
//! matrices, once per pipeline. That is small enough that plain loops beat pulling in a
//! LAPACK backend or a linear algebra crate with its own dependency tree

use ndarray::{Array1, Array2, ArrayView2};

// implicit QL usually needs one to three iterations per eigenvalue
const MAX_QL_ITERATIONS_PER_VALUE: usize = 30;

/// Failure while computing a PLDA inverse or generalized eigensystem
#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    /// A matrix that must be symmetric positive definite was not
    #[error("PLDA matrix is not positive definite")]
    NotPositiveDefinite,
    /// A matrix was singular
    #[error("PLDA matrix is singular")]
    Singular,
    /// The symmetric eigensolver did not converge
    #[error("PLDA eigendecomposition did not converge")]
    NoConvergence,
    /// A calculation produced a non-finite value
    #[error("PLDA linear algebra produced a non-finite value")]
    NonFinite,
    /// A scale would lose nonzero values in the subnormal range
    #[error("PLDA matrix scale is outside the supported normal f64 range")]
    UnsupportedScale,
}

/// Inverse of a symmetric positive-definite matrix, reading only its lower triangle
pub(crate) fn inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>, LinalgError> {
    let lower = cholesky_lower(matrix.view())?;
    let n = matrix.nrows();
    let mut inverse = Array2::<f64>::eye(n);
    for mut column in inverse.columns_mut() {
        let mut values = column.to_vec();
        solve_lower_in_place(&lower, &mut values);
        solve_lower_transpose_in_place(&lower, &mut values);
        column.assign(&Array1::from(values));
    }

    finite(inverse)
}

/// Inverse of a general square matrix
pub(crate) fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, LinalgError> {
    // use pivoted LU because the caller does not promise a symmetric matrix
    let n = matrix.nrows();
    let mut lu = matrix.to_owned();
    let mut permutation: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&a, &b| lu[(a, k)].abs().total_cmp(&lu[(b, k)].abs()))
            .unwrap_or(k);
        if lu[(pivot, k)] == 0.0 {
            return Err(LinalgError::Singular);
        }

        if pivot != k {
            permutation.swap(pivot, k);
            for col in 0..n {
                lu.swap((pivot, col), (k, col));
            }
        }

        for row in k + 1..n {
            let factor = lu[(row, k)] / lu[(k, k)];
            lu[(row, k)] = factor;
            for col in k + 1..n {
                lu[(row, col)] -= factor * lu[(k, col)];
            }
        }
    }

    let mut inverse = Array2::<f64>::zeros((n, n));
    for (col, mut column) in inverse.columns_mut().into_iter().enumerate() {
        // solve L U x = P e_col
        let mut values: Vec<f64> = permutation
            .iter()
            .map(|&source| f64::from(source == col))
            .collect();
        for row in 0..n {
            let sum: f64 = (0..row).map(|k| lu[(row, k)] * values[k]).sum();
            values[row] -= sum;
        }
        for row in (0..n).rev() {
            let sum: f64 = (row + 1..n).map(|k| lu[(row, k)] * values[k]).sum();
            values[row] = (values[row] - sum) / lu[(row, row)];
        }
        column.assign(&Array1::from(values));
    }

    finite(inverse)
}

/// Solve `A x = lambda B x` for symmetric `A` and symmetric positive-definite `B`
///
/// Matches LAPACK `dsygv` (itype 1, lower triangles): eigenvalues come back ascending, and
/// eigenvector `i` is column `i`, normalized so that `X^T B X = I`. Eigenvector signs are
/// arbitrary, as in LAPACK
pub(crate) fn generalized_eigh(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), LinalgError> {
    let n = a.nrows();
    let a_scale = MatrixScale::for_lower_triangle(a)?;
    let b_scale = MatrixScale::for_lower_triangle(b)?;
    let scaled_b = if b_scale.exponent == 0 {
        b.clone()
    } else {
        let mirrored = Array2::from_shape_fn(b.dim(), |(row, col)| b[(row.max(col), row.min(col))]);
        b_scale.apply(mirrored)?
    };

    let lower = cholesky_lower(scaled_b.view())?;

    // reduce to the standard problem C y = lambda y with C = L^-1 A L^-T and x = L^-T y,
    // mirroring the lower triangle of A to keep LAPACK's UPLO=Lower semantics
    let mirrored = Array2::from_shape_fn((n, n), |(row, col)| a[(row.max(col), row.min(col))]);
    let reduced = reduce_symmetric(a_scale.apply(mirrored)?, &lower)?;
    let reduced_scale = MatrixScale::for_lower_triangle(&reduced)?;
    let (mut values, vectors) = symmetric_eigh(reduced_scale.apply(reduced)?)?;
    let restore_exponent = b_scale.exponent - a_scale.exponent - reduced_scale.exponent;
    if restore_exponent != 0 {
        for value in &mut values {
            *value = scale_value(*value, restore_exponent)?;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));

    let sorted_values = Array1::from_iter(order.iter().map(|&idx| values[idx]));
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));
    for (target, &source) in order.iter().enumerate() {
        let mut values = vectors.column(source).to_vec();
        solve_lower_transpose_in_place(&lower, &mut values);
        if b_scale.exponent != 0 {
            // restore X^T B X = I after factoring the scaled B
            let normalization = b_scale.factor().sqrt();
            for value in &mut values {
                *value *= normalization;
            }
        }

        sorted_vectors
            .column_mut(target)
            .assign(&Array1::from(values));
    }

    if !sorted_values.iter().all(|value| value.is_finite()) {
        return Err(LinalgError::NonFinite);
    }

    Ok((sorted_values, finite(sorted_vectors)?))
}

/// Exact power-of-two scaling, used only outside the range safe for QL products
#[derive(Clone, Copy)]
struct MatrixScale {
    exponent: i32,
}

impl MatrixScale {
    fn for_lower_triangle(matrix: &Array2<f64>) -> Result<Self, LinalgError> {
        let mut maximum = 0.0f64;
        for ((row, col), &value) in matrix.indexed_iter() {
            if row < col {
                continue;
            }

            if !value.is_finite() {
                return Err(LinalgError::NonFinite);
            }

            if value.is_subnormal() {
                return Err(LinalgError::UnsupportedScale);
            }

            maximum = maximum.max(value.abs());
        }

        // leave shipped PLDA arithmetic unchanged, with ample headroom for products
        if maximum == 0.0 || (2.0f64.powi(-400)..=2.0f64.powi(400)).contains(&maximum) {
            return Ok(Self { exponent: 0 });
        }

        let binary_exponent = ((maximum.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        Ok(Self {
            exponent: -binary_exponent,
        })
    }

    fn factor(self) -> f64 {
        // construct the exact factor, including 2^-1023 for the largest inputs
        if self.exponent == -1023 {
            return f64::MIN_POSITIVE / 2.0;
        }

        f64::from_bits(((self.exponent + 1023) as u64) << 52)
    }

    fn apply(self, mut matrix: Array2<f64>) -> Result<Array2<f64>, LinalgError> {
        if self.exponent == 0 {
            return Ok(matrix);
        }

        for value in &mut matrix {
            let original = *value;
            *value *= self.factor();
            check_scaled_value(original, *value)?;
        }
        Ok(matrix)
    }
}

fn check_scaled_value(original: f64, scaled: f64) -> Result<(), LinalgError> {
    if !scaled.is_finite() {
        return Err(LinalgError::NonFinite);
    }

    if scaled.is_subnormal() || (original != 0.0 && scaled == 0.0) {
        return Err(LinalgError::UnsupportedScale);
    }
    Ok(())
}

fn scale_value(mut value: f64, mut exponent: i32) -> Result<f64, LinalgError> {
    // staged factors avoid overflowing the factor when A and B have opposite scales
    while exponent != 0 {
        let step = exponent.clamp(-1022, 1023);
        let original = value;
        value *= MatrixScale { exponent: step }.factor();
        check_scaled_value(original, value)?;
        exponent -= step;
    }
    Ok(value)
}

fn reduce_symmetric(
    mut reduced: Array2<f64>,
    lower: &Array2<f64>,
) -> Result<Array2<f64>, LinalgError> {
    let n = reduced.nrows();
    for mut column in reduced.columns_mut() {
        let mut values = column.to_vec();
        solve_lower_in_place(lower, &mut values);
        column.assign(&Array1::from(values));
    }
    for mut row in reduced.rows_mut() {
        let mut values = row.to_vec();
        solve_lower_in_place(lower, &mut values);
        row.assign(&Array1::from(values));
    }

    // keep normal-scale rounding unchanged; halve first only if the sum overflows
    finite(Array2::from_shape_fn((n, n), |(row, col)| {
        let a = reduced[(row, col)];
        let b = reduced[(col, row)];
        let sum = a + b;
        if sum.is_finite() {
            0.5 * sum
        } else {
            a / 2.0 + b / 2.0
        }
    }))
}

/// Lower Cholesky factor of a symmetric positive-definite matrix, reading its lower triangle
fn cholesky_lower(matrix: ArrayView2<f64>) -> Result<Array2<f64>, LinalgError> {
    let n = matrix.nrows();
    let mut lower = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let diagonal = matrix[(col, col)] - (0..col).map(|k| lower[(col, k)].powi(2)).sum::<f64>();
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return Err(LinalgError::NotPositiveDefinite);
        }

        let pivot = diagonal.sqrt();
        lower[(col, col)] = pivot;
        for row in col + 1..n {
            let sum: f64 = (0..col).map(|k| lower[(row, k)] * lower[(col, k)]).sum();
            lower[(row, col)] = (matrix[(row, col)] - sum) / pivot;
        }
    }

    Ok(lower)
}

/// Overwrite `values` with `L^-1 values`
fn solve_lower_in_place(lower: &Array2<f64>, values: &mut [f64]) {
    for row in 0..values.len() {
        let sum: f64 = (0..row).map(|k| lower[(row, k)] * values[k]).sum();
        values[row] = (values[row] - sum) / lower[(row, row)];
    }
}

/// Overwrite `values` with `L^-T values`
fn solve_lower_transpose_in_place(lower: &Array2<f64>, values: &mut [f64]) {
    let n = values.len();
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| lower[(k, row)] * values[k]).sum();
        values[row] = (values[row] - sum) / lower[(row, row)];
    }
}

/// Eigenvalues (unsorted) and column eigenvectors of a symmetric matrix
///
/// Householder tridiagonalization followed by implicit QL, ported from the public-domain
/// JAMA `EigenvalueDecomposition` (EISPACK `tred2` and `tql2`), the same algorithm family
/// LAPACK uses for symmetric matrices
fn symmetric_eigh(matrix: Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), LinalgError> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(LinalgError::NonFinite);
    }

    let mut eigen = SymmetricEigen::new(matrix);
    eigen.tridiagonalize();
    eigen.diagonalize()?;
    Ok(eigen.into_eigensystem())
}

/// Working state for `tred2` and `tql2` on a flat row-major eigenvector buffer
struct SymmetricEigen {
    n: usize,
    /// eigenvectors as columns, row-major
    vectors: Vec<f64>,
    /// diagonal, then eigenvalues
    diagonal: Vec<f64>,
    /// subdiagonal
    off: Vec<f64>,
}

impl SymmetricEigen {
    fn new(matrix: Array2<f64>) -> Self {
        let n = matrix.nrows();
        Self {
            n,
            vectors: matrix.iter().copied().collect(),
            diagonal: vec![0.0; n],
            off: vec![0.0; n],
        }
    }

    fn at(&self, row: usize, col: usize) -> f64 {
        self.vectors[row * self.n + col]
    }

    fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        &mut self.vectors[row * self.n + col]
    }

    /// Householder reduction to tridiagonal form (`tred2`)
    fn tridiagonalize(&mut self) {
        let n = self.n;
        for j in 0..n {
            self.diagonal[j] = self.at(n - 1, j);
        }

        for i in (1..n).rev() {
            let scale: f64 = self.diagonal[..i].iter().map(|value| value.abs()).sum();
            let mut h = 0.0;
            if scale == 0.0 {
                self.off[i] = self.diagonal[i - 1];
                for j in 0..i {
                    self.diagonal[j] = self.at(i - 1, j);
                    *self.at_mut(i, j) = 0.0;
                    *self.at_mut(j, i) = 0.0;
                }
                self.diagonal[i] = h;
                continue;
            }

            for k in 0..i {
                self.diagonal[k] /= scale;
                h += self.diagonal[k] * self.diagonal[k];
            }
            let mut f = self.diagonal[i - 1];
            let mut g = if f > 0.0 { -h.sqrt() } else { h.sqrt() };
            self.off[i] = scale * g;
            h -= f * g;
            self.diagonal[i - 1] = f - g;
            self.off[..i].fill(0.0);

            for j in 0..i {
                f = self.diagonal[j];
                *self.at_mut(j, i) = f;
                g = self.off[j] + self.at(j, j) * f;
                for k in j + 1..i {
                    g += self.at(k, j) * self.diagonal[k];
                    self.off[k] += self.at(k, j) * f;
                }
                self.off[j] = g;
            }

            f = 0.0;
            for j in 0..i {
                self.off[j] /= h;
                f += self.off[j] * self.diagonal[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                self.off[j] -= hh * self.diagonal[j];
            }

            for j in 0..i {
                f = self.diagonal[j];
                g = self.off[j];
                for k in j..i {
                    let update = f * self.off[k] + g * self.diagonal[k];
                    *self.at_mut(k, j) -= update;
                }
                self.diagonal[j] = self.at(i - 1, j);
                *self.at_mut(i, j) = 0.0;
            }
            self.diagonal[i] = h;
        }

        self.accumulate_householder();
    }

    /// Turn the stored Householder vectors into the orthogonal transformation
    fn accumulate_householder(&mut self) {
        let n = self.n;
        for i in 0..n - 1 {
            let diagonal_entry = self.at(i, i);
            *self.at_mut(n - 1, i) = diagonal_entry;
            *self.at_mut(i, i) = 1.0;
            let h = self.diagonal[i + 1];
            if h != 0.0 {
                for k in 0..=i {
                    self.diagonal[k] = self.at(k, i + 1) / h;
                }
                for j in 0..=i {
                    let g: f64 = (0..=i).map(|k| self.at(k, i + 1) * self.at(k, j)).sum();
                    for k in 0..=i {
                        let update = g * self.diagonal[k];
                        *self.at_mut(k, j) -= update;
                    }
                }
            }
            for k in 0..=i {
                *self.at_mut(k, i + 1) = 0.0;
            }
        }

        for j in 0..n {
            self.diagonal[j] = self.at(n - 1, j);
            *self.at_mut(n - 1, j) = 0.0;
        }
        *self.at_mut(n - 1, n - 1) = 1.0;
        self.off[0] = 0.0;
    }

    /// Implicit QL iteration on the tridiagonal matrix (`tql2`)
    fn diagonalize(&mut self) -> Result<(), LinalgError> {
        let n = self.n;
        self.off.copy_within(1.., 0);
        self.off[n - 1] = 0.0;

        let mut shift_total = 0.0;
        let mut tolerance_base = 0.0f64;
        for l in 0..n {
            tolerance_base = tolerance_base.max(self.diagonal[l].abs() + self.off[l].abs());
            let tolerance = f64::EPSILON * tolerance_base;
            let m = (l..n)
                .find(|&m| self.off[m].abs() <= tolerance)
                .unwrap_or(n - 1);

            let mut iterations = 0;
            while m > l && self.off[l].abs() > tolerance {
                iterations += 1;
                if iterations > MAX_QL_ITERATIONS_PER_VALUE {
                    return Err(LinalgError::NoConvergence);
                }
                shift_total += self.ql_step(l, m);
            }

            self.diagonal[l] += shift_total;
            self.off[l] = 0.0;
        }

        Ok(())
    }

    /// One implicitly shifted QL step on the unreduced block `l..=m`, returning its shift
    fn ql_step(&mut self, l: usize, m: usize) -> f64 {
        let n = self.n;

        // wilkinson-style shift from the leading 2x2 block
        let g = self.diagonal[l];
        let mut p = (self.diagonal[l + 1] - g) / (2.0 * self.off[l]);
        let mut r = p.hypot(1.0);
        if p < 0.0 {
            r = -r;
        }
        self.diagonal[l] = self.off[l] / (p + r);
        self.diagonal[l + 1] = self.off[l] * (p + r);
        let dl1 = self.diagonal[l + 1];
        let shift = g - self.diagonal[l];
        for value in &mut self.diagonal[l + 2..] {
            *value -= shift;
        }

        p = self.diagonal[m];
        let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
        let el1 = self.off[l + 1];
        let (mut s, mut s2) = (0.0, 0.0);
        for i in (l..m).rev() {
            c3 = c2;
            c2 = c;
            s2 = s;
            let g = c * self.off[i];
            let h = c * p;
            r = p.hypot(self.off[i]);
            self.off[i + 1] = s * r;
            s = self.off[i] / r;
            c = p / r;
            p = c * self.diagonal[i] - s * g;
            self.diagonal[i + 1] = h + s * (c * g + s * self.diagonal[i]);

            for k in 0..n {
                let right = self.at(k, i + 1);
                let left = self.at(k, i);
                *self.at_mut(k, i + 1) = s * left + c * right;
                *self.at_mut(k, i) = c * left - s * right;
            }
        }

        p = -s * s2 * c3 * el1 * self.off[l] / dl1;
        self.off[l] = s * p;
        self.diagonal[l] = c * p;
        shift
    }

    fn into_eigensystem(self) -> (Array1<f64>, Array2<f64>) {
        let n = self.n;
        let vectors = Array2::from_shape_vec((n, n), self.vectors)
            .expect("eigenvector buffer holds n * n values");
        (Array1::from(self.diagonal), vectors)
    }
}

fn finite(matrix: Array2<f64>) -> Result<Array2<f64>, LinalgError> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(LinalgError::NonFinite);
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::{
        LinalgError, MatrixScale, cholesky_lower, generalized_eigh, inverse, inverse_spd,
        reduce_symmetric,
    };

    fn assert_eigenpairs(a: &Array2<f64>, b: &Array2<f64>, eigenvalue_scale: f64) {
        let (values, vectors) = generalized_eigh(a, b).unwrap();
        let expected = [2.0 - 2.0f64.sqrt(), 2.0, 2.0 + 2.0f64.sqrt()];
        let a_norm = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let b_norm = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let normalized_a = a / a_norm;
        let normalized_b = b / b_norm;
        let frobenius = normalized_a.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (idx, &value) in values.iter().enumerate() {
            assert!((value / eigenvalue_scale - expected[idx]).abs() < 1e-13);
            let vector = vectors.column(idx);
            let v_max = vector.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let vector = &vector / v_max;
            // normalize before products and norms so the residual cannot overflow or underflow
            let lambda = (value / a_norm) * b_norm;
            let residual = normalized_a.dot(&vector) - normalized_b.dot(&vector) * lambda;
            let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            let vector_norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
            let relative = residual_norm / (frobenius * vector_norm);
            assert!(relative < 1e-14, "scale-aware residual {relative:e}");
            let gram = vector.dot(&normalized_b.dot(&vector)) * b_norm * v_max * v_max;
            assert!((gram - 1.0).abs() < 1e-13, "B normalization {gram}");
        }
    }

    #[test]
    fn review_matrix_is_correct_at_extreme_scales() {
        let base = array![[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]];
        for scale in [1.0, 1e-200, 1e200] {
            assert_eigenpairs(&(&base * scale), &Array2::eye(3), scale);
        }

        let (values, _) = generalized_eigh(&base, &Array2::eye(3)).unwrap();
        // pin the unscaled solver's existing f64 results, not only a tolerance
        assert_eq!(
            values,
            array![0.5857864376269051, 1.9999999999999998, 3.414213562373095]
        );
    }

    #[test]
    fn generalized_problem_scales_a_b_and_eigenvectors() {
        let base = array![[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]];
        let lower = array![[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, -0.5, 3.0]];
        let a = lower.dot(&base).dot(&lower.t());
        let b = lower.dot(&lower.t());
        // these two cases leave A and B unscaled but force scaling of the reduced matrix
        for (a_scale, b_scale) in [(1e100, 1e-100), (1e-100, 1e100)] {
            assert_eigenpairs(&(&a * a_scale), &(&b * b_scale), a_scale / b_scale);
        }

        for scale in [1.0, 1e-200, 1e200] {
            for (a_scale, b_scale) in [(scale, 1.0), (1.0, scale), (scale, scale)] {
                assert_eigenpairs(&(&a * a_scale), &(&b * b_scale), a_scale / b_scale);
            }
        }
    }

    #[test]
    fn unsupported_scales_return_typed_errors() {
        for (a, b) in [
            (array![[f64::from_bits(1)]], Array2::eye(1)),
            (Array2::eye(1), array![[f64::from_bits(1)]]),
            (array![[1e-200]], array![[1e200]]),
            (array![[1e200]], array![[1e-200]]),
            (array![[1e200, 0.0], [1e-200, 1e200]], Array2::eye(2)),
        ] {
            assert!(matches!(
                generalized_eigh(&a, &b),
                Err(LinalgError::UnsupportedScale | LinalgError::NonFinite)
            ));
        }

        assert!(matches!(
            generalized_eigh(&array![[f64::INFINITY]], &Array2::eye(1)),
            Err(LinalgError::NonFinite)
        ));
    }

    #[test]
    fn symmetric_average_does_not_overflow() {
        let reduced = reduce_symmetric(array![[f64::MAX]], &Array2::eye(1)).unwrap();
        assert_eq!(reduced[(0, 0)], f64::MAX);
    }

    #[test]
    fn shipped_plda_inputs_and_reduced_matrix_need_no_scaling() {
        let models = crate::test_support::model_fixture_dir();
        let raw: Array2<f64> = ndarray_npy::read_npy(models.join("plda_tr.npy")).unwrap();
        let psi: ndarray::Array1<f64> = ndarray_npy::read_npy(models.join("plda_psi.npy")).unwrap();
        let b = inverse_spd(&raw.t().dot(&raw)).unwrap();
        let mut weighted = raw.t().to_owned();
        for (mut column, value) in weighted.columns_mut().into_iter().zip(&psi) {
            column /= *value;
        }

        let a = inverse(&weighted.dot(&raw)).unwrap();
        assert_eq!(MatrixScale::for_lower_triangle(&a).unwrap().factor(), 1.0);
        assert_eq!(MatrixScale::for_lower_triangle(&b).unwrap().factor(), 1.0);
        let mirrored = Array2::from_shape_fn(a.dim(), |(row, col)| a[(row.max(col), row.min(col))]);
        let reduced = reduce_symmetric(mirrored, &cholesky_lower(b.view()).unwrap()).unwrap();
        assert_eq!(
            MatrixScale::for_lower_triangle(&reduced).unwrap().factor(),
            1.0
        );
    }
}
