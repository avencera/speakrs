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
    let lower = cholesky_lower(b.view())?;

    // reduce to the standard problem C y = lambda y with C = L^-1 A L^-T and x = L^-T y,
    // mirroring the lower triangle of A to keep LAPACK's UPLO=Lower semantics
    let mut reduced = Array2::from_shape_fn((n, n), |(row, col)| a[(row.max(col), row.min(col))]);
    for mut column in reduced.columns_mut() {
        let mut values = column.to_vec();
        solve_lower_in_place(&lower, &mut values);
        column.assign(&Array1::from(values));
    }
    for mut row in reduced.rows_mut() {
        let mut values = row.to_vec();
        solve_lower_in_place(&lower, &mut values);
        row.assign(&Array1::from(values));
    }

    // the two triangular solves leave rounding-level asymmetry that Jacobi would otherwise keep
    let symmetric = Array2::from_shape_fn((n, n), |(row, col)| {
        0.5 * (reduced[(row, col)] + reduced[(col, row)])
    });
    let (values, vectors) = symmetric_eigh(symmetric)?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));

    let sorted_values = Array1::from_iter(order.iter().map(|&idx| values[idx]));
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));
    for (target, &source) in order.iter().enumerate() {
        let mut values = vectors.column(source).to_vec();
        solve_lower_transpose_in_place(&lower, &mut values);
        sorted_vectors
            .column_mut(target)
            .assign(&Array1::from(values));
    }

    if !sorted_values.iter().all(|value| value.is_finite()) {
        return Err(LinalgError::NonFinite);
    }

    Ok((sorted_values, finite(sorted_vectors)?))
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
