use pyo3::prelude::*;

mod common;
pub mod fly;

#[pymodule]
fn data_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Dataset>()?;

    Ok(())
}
