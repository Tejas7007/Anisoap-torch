mod ellip_expansion;

use ellip_expansion::compute_moments::{compute_moments_rust, compute_moments_batch_rust};
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;


#[pymodule]
fn anisoap_rust_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn compute_moments<'py>(
        py: Python<'py>,
        mat: PyReadonlyArray2<'_, f64>,
        g_vec: PyReadonlyArray1<'_, f64>,
        max_deg: i32,
    ) -> PyResult<&'py PyArray3<f64>> {
        Ok(compute_moments_rust(mat.as_array(), g_vec.as_array(), max_deg)?.into_pyarray(py))
    }

    #[pyfn(m)]
    fn compute_moments_batch<'py>(
        py: Python<'py>,
        mats: PyReadonlyArray2<'_, f64>,
        centers: PyReadonlyArray2<'_, f64>,
        max_deg: i32,
    ) -> PyResult<&'py PyArray1<f64>> {
        let n_pairs = centers.shape()[0];
        // Copy data so we can release GIL
        let mats_vec: Vec<f64> = mats.as_array().iter().copied().collect();
        let centers_vec: Vec<f64> = centers.as_array().iter().copied().collect();

        // Release the GIL so Rayon threads can run in parallel
        let result = py.allow_threads(|| {
            compute_moments_batch_rust(&mats_vec, &centers_vec, max_deg, n_pairs)
        })?;

        Ok(numpy::ndarray::Array1::from(result).into_pyarray(py))
    }

    Ok(())
}
