use anndata::data::Element;
use anndata::{s, AnnData, AnnDataOp, ArrayElemOp, HasShape};
use anndata_hdf5::H5;
use anyhow::Context;
use sprs::io::{self, read_matrix_market_from_bufread};
use std::env;
use std::io::BufRead;

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
    /*
    let a: sprs::CsMat<f64> = sprs::io::read_matrix_market(p).expect("read file").to_csr();
    println!("loading into CSR took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let nr = a.rows();
    let nc = a.cols();
    let ngenes = nc / 3;

    // convert to CSR
    let (iptr, istor, dat) = a.into_raw_storage();
    let ap = nalgebra_sparse::csr::CsrMatrix::try_from_csr_data(nr, nc, iptr, istor, dat)
        .expect("csr matrix");
    eprintln!("CSR to CSR took {:#?}", sw.elapsed());

    sw.reset();
    sw.start()?;
    */
    let r = anndata::reader::MMReader::from_path(&p)?
        .obs_names(&rowpath)?
        .var_names(&colpath)?;
    // make  AnnData object
    let b = anndata::AnnData::<H5>::new("foo.anndata")?;
    r.finish(&b)?;
    eprintln!("Reading MM into AnnData took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let nr = b.n_obs();
    let nc = b.n_vars();
    let ngenes = nc / 3;
    /*
    let c = anndata::data::array::DynCsrMatrix::F64(ap);
    eprintln!("converting to CSR took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    // set the AnnData content
    b.set_x(c)?;
    eprintln!("setting in anndata took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;
    */

    // Get the unspliced, spliced and ambiguous slices
    let vars = b.read_var().context("should exist")?;
    eprintln!("vars : {:#?}", vars);

    let slice1: anndata::ArrayData = b.get_x().slice(s![.., 0..ngenes])?.unwrap();
    let var1 = vars.select_by_range(0..ngenes)?;
    eprintln!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice2: anndata::ArrayData = b.get_x().slice(s![.., ngenes..2 * ngenes])?.unwrap();
    let var2 = vars.select_by_range(ngenes..2 * ngenes)?;
    eprintln!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice3: anndata::ArrayData = b.get_x().slice(s![.., 2 * ngenes..3 * ngenes])?.unwrap();
    let var3 = vars.select_by_range(2 * ngenes..3 * ngenes)?;
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
    Ok(())
}
