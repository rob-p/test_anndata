use anndata::{s, AnnData, AnnDataOp, ArrayData, ArrayElemOp};
use anndata_hdf5::H5;
use anyhow::Context;
use polars::io::prelude::*;
use polars::prelude::CsvReadOptions;
use std::env;

fn main() -> anyhow::Result<()> {
    let root_path = env::args().nth(1).expect("valid arg");

    let mut p = std::path::PathBuf::from(&root_path);
    p.set_file_name("quants_mat.mtx");

    let mut colpath = std::path::PathBuf::from(&root_path);
    colpath.set_file_name("quants_mat_cols.txt");

    let mut rowpath = std::path::PathBuf::from(&root_path);
    rowpath.set_file_name("quants_mat_rows.txt");

    let mut sw = libsw::Sw::new();
    sw.start()?;

    let r = anndata::reader::MMReader::from_path(&p)?;
    let mut col_df = CsvReadOptions::default()
        .with_has_header(false)
        .try_into_reader_with_file_path(Some(colpath))?
        .finish()?;
    col_df.set_column_names(["gene_symbols"])?;

    let mut row_df = CsvReadOptions::default()
        .with_has_header(false)
        .try_into_reader_with_file_path(Some(rowpath))?
        .finish()?;
    row_df.set_column_names(["barcodes"])?;
    // make  AnnData object
    let b = AnnData::<H5>::new("foo.anndata")?;
    r.finish(&b)?;
    eprintln!("Reading MM into AnnData took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let nr = b.n_obs();
    let nc = b.n_vars();
    let ngenes = nc / 3;

    // Get the unspliced, spliced and ambiguous slices
    let vars = col_df;
    eprintln!("vars : {:#?}", vars.shape());

    let slice1: ArrayData = b.get_x().slice(s![.., 0..ngenes])?.unwrap();
    let var1 = vars.slice(0_i64, ngenes);
    eprintln!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice2: ArrayData = b.get_x().slice(s![.., ngenes..2 * ngenes])?.unwrap();
    let var2 = vars.slice(ngenes as i64, ngenes);
    eprintln!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice3: ArrayData = b.get_x().slice(s![.., 2 * ngenes..3 * ngenes])?.unwrap();
    let var3 = vars.slice(2_i64 * ngenes as i64, ngenes);
    eprintln!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    // We must have an X, but don't want to waste space on it
    // so set it as and empty matrix
    let csr_zero = nalgebra_sparse::CsrMatrix::<f64>::zeros(nr, ngenes);
    let csr_zero = anndata::data::array::DynCsrMatrix::F64(csr_zero);
    // get rid of the old X
    b.del_x().context("unable to delete X")?;
    b.del_obs()?;
    b.set_n_obs(nr).context("unable to set n_obs")?;
    b.set_n_vars(ngenes).context("unable to set n_vars")?;
    // set the new X
    b.set_x(csr_zero).context("unable to set all 0s X")?;

    let layers = vec![
        ("spliced".to_owned(), slice1),
        ("unspliced".to_owned(), slice2),
        ("ambiguous".to_owned(), slice3),
    ];

    let varm = vec![
        ("spliced".to_owned(), var1),
        ("unspliced".to_owned(), var2),
        ("ambiguous".to_owned(), var3),
    ];
    b.set_layers(layers)
        .context("unable to set layers for AnnData object")?;
    eprintln!("setting layers took {:#?}", sw.elapsed());
    b.set_varm(varm)?;
    b.set_obs(row_df)?;
    Ok(())
}
