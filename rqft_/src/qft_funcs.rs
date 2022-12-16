use colored::Colorize;
use core::fmt::{Display, Formatter};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Array1, Array2};
use serde::Deserialize;
use std::{collections::HashMap, error::Error, f64::consts::PI, vec};

use crate::file_utils::init_file_writter;

/// Initialize a ProgressBar using 'indicatif' crate and
///
/// # Examples
///
/// ```rust
/// let msg: String = String::from("A message");
/// let iters: u64 = 1000;
/// let pb = init_progress_bar(&msg, iters);
/// ```
fn init_progress_bar(msg: String, iters: u64) -> ProgressBar {
    let pb = ProgressBar::new(iters);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "[{}] {{elapsed_precise}} {{bar:40.cyan/blue}} {{pos:>2}}/{{len:2}}",
            &msg
        ))
        .unwrap()
        .progress_chars("#--"),
    );
    pb
}

#[derive(Deserialize)]
pub struct Model {
    // Public properties
    pub eta: f64,
    pub omega: f64,
    pub t: f64,
    pub tp: f64,
    pub tpp: f64,
    pub mu: f64,
    pub beta: f64,
    pub peter_size: i32,
    pub peter_res: i32,

    // Private properties
    mu_min: f64,
    mu_max: f64,
    n_mu: usize,
    n_k: usize,
}

impl Default for Model {
    fn default() -> Model {
        Model {
            eta: 0.1,
            omega: 0.0,
            t: 1.0,
            tp: 0.0,
            tpp: 0.0,
            mu: 0.0,
            beta: 100.0,
            peter_size: 36,
            peter_res: 200,

            mu_min: -4.0,
            mu_max: 4.0,
            n_mu: 200,
            n_k: 200,
        }
    }
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        write!(
            f,
            "Model_object(
    {}: {},
    {}: {},
    {}: ({}, {}, {}),
    {}: {},
    {}: ({}, {}, {}),
    {}: {},
    {}: {},
    {}: {},
    {}: {}
)",
            "eta".cyan().bold(),
            self.eta,
            "omega".cyan().bold(),
            self.omega,
            "t, t', t''".cyan(),
            self.t,
            self.tp,
            self.tpp,
            "mu".cyan().bold(),
            self.mu,
            "mu_min, mu_max, n_mu".cyan().bold(),
            self.mu_min,
            self.mu_max,
            self.n_mu,
            "beta".cyan().bold(),
            self.beta,
            "n_k".cyan().bold(),
            self.n_k,
            "Peter cluster size".cyan().bold(),
            self.peter_size,
            "Peter momentum res.".cyan().bold(),
            self.peter_res
        )
    }
}

impl Model {
    /// Defines array for all values of chemical potential
    ///
    /// # Examples
    ///
    /// ```
    /// let model = Model {
    /// ...
    /// }
    /// let array = &model.mu_array();
    /// ```
    pub fn mu_array(&self) -> Array1<f64> {
        Array::linspace(self.mu_min, self.mu_max, self.n_mu)
    }

    /// Returns phase space 2D grids for further calculation
    /// such as energy and spectral weight.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = Model {
    /// ...
    /// }
    /// let k_x_array = &model.k_grid('x');
    /// ```
    pub fn k_grid(&self, axis: char) -> Array2<f64> {
        let mut k = Array2::zeros((self.n_k, self.n_k));
        if axis == 'x' {
            k = Array::from_shape_fn((self.n_k, self.n_k), |(_i, j)| {
                PI * (-1. + 2. * j as f64 / self.n_k as f64)
            });
        } else if axis == 'y' {
            k = Array::from_shape_fn((self.n_k, self.n_k), |(i, _j)| {
                PI * (-1. + 2. * i as f64 / self.n_k as f64)
            });
        }
        k
    }

    /// Reads spectral weight from file.
    ///
    /// # Examples
    ///
    /// ```
    /// let peter_data = read_spectral_weight();
    /// ```
    pub fn peter_data(&self) -> Result<HashMap<usize, Array2<f64>>, Box<dyn Error>> {
        let mut paths: Vec<String> = vec![String::from("")];
        let mut indexes: Vec<usize> = vec![];
        let mut elements: usize = 40000;
        let mut shape: usize = 200;

        if self.peter_size == 36 {
            paths = vec![
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N24.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N26.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N28.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N30.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N32.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N34.csv"),
                String::from("./src/Data/fermi_arc_data_1D_N36/Akw_N36.csv"),
            ];
            indexes = vec![0, 1, 2, 3, 4, 5, 6];
        } else if self.peter_size == 64 {
            if self.peter_res == 200 {
                paths = vec![
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_200/Akw_N48.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_200/Akw_N52.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_200/Akw_N56.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_200/Akw_N60.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_200/Akw_N64.csv"),
                ];
            } else if self.peter_res == 500 {
                elements = 250000;
                shape = 500;
                paths = vec![
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_500/Akw_N48.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_500/Akw_N52.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_500/Akw_N56.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_500/Akw_N60.csv"),
                    String::from("./src/Data/fermi_arc_data_1D_N64/nk_500/Akw_N64.csv"),
                ];
            }
            indexes = vec![0, 1, 2, 3, 4];
        }
        let mut arcs_data: HashMap<usize, Array2<f64>> = HashMap::new();

        for (idx, path) in indexes.iter().zip(paths.iter()) {
            // Init reader for each .csv file
            let mut rdr = csv::ReaderBuilder::new().from_path(path).unwrap();

            // Init empty array of size (dimension)
            let mut spectral_weight = Array1::zeros(elements);
            for (idx, rec) in rdr.records().enumerate() {
                let values = rec?;
                spectral_weight[idx] = values[0].parse::<f64>().unwrap();
            }
            // Insert converted 2D array inside HashMap
            let buff_array = spectral_weight.into_shape((shape, shape)).unwrap();
            arcs_data.insert(*idx, buff_array);
        }
        Ok(arcs_data)
    }

    /// Returns energy 2D array for given phase space.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = Model {
    /// ...
    /// }
    /// let k_x = &model.k_grid('x');
    /// let k_y = &model.k_grid('y');
    /// let energy = &model(k_x, k_y);
    /// ```
    pub fn energy(&self, kx: &Array2<f64>, ky: &Array2<f64>) -> Array2<f64> {
        let energy_1 = -2. * self.t * (kx.mapv(|kx| kx.cos()) + ky.mapv(|ky| ky.cos()));
        let energy_2 = -4. * self.tp * kx.map(|kx| kx.cos()) * ky.map(|ky| ky.cos());
        let energy_3 =
            -2. * self.tpp * (kx.map(|kx| (2. * kx).cos()) + ky.map(|ky| (2. * ky).cos()));
        energy_1 + energy_2 + energy_3
    }
}

/// Returns Model instance using numerical parameters
/// from properly formatted file.
///
/// # Examples
///
/// ```
/// let path: String = String::from("./path/to/file")
/// let params: Model = get_model_params(&path).unwrap();
/// ```
pub fn get_model_params(path: &String) -> Result<Model, csv::Error> {
    // Reader of model.dat
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b' ')
        .from_path(path)
        .unwrap();

    // Default Model instance initialization
    let mut params = Model {
        ..Default::default()
    };
    for row in reader.deserialize() {
        params = row?;
    }
    Ok(params)
}

/// Computes model's spectral functions (for all given chemical potentials)
/// using given parameters and writes it to .dat file while formatting it
/// to be 'gnuplot' friendly.
///
/// # Examples
///
/// ```
/// let eta = 1.0;
/// let omega = 0.0;
/// let mus = array![0, 0];
/// let energy: Array2<f64> = Array2::zeros((2, 2));
/// let s_w: Hashmap<usize, Array2<f64>> = get_spectral_weight(
///                                         &energy,
///                                         eta,
///                                         omega,
///                                         &mus,
///                                         (false, 0.0));
/// ```
pub fn get_spectral_weight(
    energies: &Array2<f64>,
    eta: f64,
    omega: f64,
    mus: &Array1<f64>,
    write: (bool, f64),
) -> HashMap<usize, Array2<f64>> {
    let mut a: HashMap<usize, Array2<f64>> = HashMap::new();
    let pb = init_progress_bar(String::from("Spectral functions"), mus.len() as u64);

    for (idx, mu) in mus.iter().enumerate() {
        // Compute spectral function for given chemical potential
        let energy_mu = energies - *mu;
        let a_1 = eta / PI;
        let a_2 = 1. / ((omega - energy_mu).mapv(|x| x.powi(2)) + eta.powi(2));
        let a_3 = a_1 * a_2;
        a.insert(idx, a_3);
        pb.inc(1);

        let mu_val = (*mu * 100.0).round() / 100.0;
        if write.0 && write.1 == mu_val {
            // Data file init
            let spectral_path = String::from("./examples/data/spectral.dat");

            let mut wtr_spectral = init_file_writter(&spectral_path, true);
            wtr_spectral.serialize(vec!["#k_x", "k_y", "A(k)"]).unwrap();

            let k_array = Array1::linspace(-PI, PI, energies.ncols());

            for i in 0..energies.nrows() {
                // Write empty line each kx to let know 'gnuplot'
                wtr_spectral.serialize(vec!["", "", ""]).unwrap();

                for j in 0..energies.ncols() {
                    wtr_spectral
                        .serialize(vec![k_array[i], k_array[j], a.get(&idx).unwrap()[[i, j]]])
                        .unwrap();
                }
            }
            wtr_spectral.flush().unwrap();
        }
    }
    pb.finish();
    a
}

/// Computes first and second energies derivatives, conductivities, densities
/// and hall number for given dispertion relation and spectral functions then writes
/// this data to .dat file.
///
/// # Examples
///
/// ```
/// let energy = Array2<f64>::zeros((2, 2));
/// let kx = Array2<f64>::zeros((2, 2));
/// let ky = Array2<f64>::zeros((2, 2));
/// let ak: Hashmap<usize, Array2<f64>>;
/// let mus = Array1<f64>::zero((2,));
/// let hops = array![1.0, 0.0, 0.0];
/// let beta = 100.0;
/// let (n, n_H) = get_conductivities(
///                 &energy,
///                 (&kx, &ky),
///                 &ak,
///                 &mus,
///                 &hops,
///                 beta
/// )
/// ```
pub fn get_conductivities(
    energy: &Array2<f64>,
    ks: (&Array2<f64>, &Array2<f64>),
    a_k: &HashMap<usize, Array2<f64>>,
    mus: &Array1<f64>,
    hop_amps: &Array1<f64>,
    beta: f64,
    peter_densities: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    // Data file init
    let conductivities_path = String::from("./examples/data/conductivities.dat");

    let mut wtr_conductivities = init_file_writter(&conductivities_path, true);
    wtr_conductivities
        .serialize(vec![
            "#mu",
            "n",
            "n_h",
            "sigma_xx",
            "sigma_yy",
            "sigma_xy",
            "Peter (p)",
        ])
        .unwrap();

    // Phase space grids unpacking
    let (k_x, k_y) = ks;

    // Constant parameters
    let t = hop_amps[0];
    let tp = hop_amps[1];
    let tpp = hop_amps[2];
    let v = 1. / (energy.ncols() as f64).powi(2);

    // de_dx
    let de_dx_1 = 2. * t * k_x.mapv(|kx| kx.sin());
    let de_dx_2 = 4. * tp * k_x.mapv(|kx| kx.sin()) * k_y.mapv(|ky| ky.cos());
    let de_dx_3 = 4. * tpp * k_x.mapv(|kx| (2. * kx).sin());
    let de_dx = de_dx_1 + de_dx_2 + de_dx_3;

    // dde_dxx
    let dde_dxx_1 = 2. * t * k_x.mapv(|kx| kx.cos());
    let dde_dxx_2 = 4. * tp * k_x.mapv(|kx| kx.cos()) * k_y.mapv(|ky| ky.cos());
    let dde_dxx_3 = 8. * tpp * k_x.mapv(|kx| (2. * kx).cos());
    let dde_dxx = dde_dxx_1 + dde_dxx_2 + dde_dxx_3;

    // de_dy
    let de_dy_1 = 2. * t * k_y.mapv(|ky| ky.sin());
    let de_dy_2 = 4. * tp * k_x.mapv(|kx| kx.cos()) * k_y.mapv(|ky| ky.sin());
    let de_dy_3 = 4. * tpp * k_y.mapv(|ky| (2. * ky).sin());
    let de_dy = de_dy_1 + de_dy_2 + de_dy_3;

    // dde_dyy
    let dde_dyy_1 = 2. * t * k_y.mapv(|ky| ky.cos());
    let dde_dyy_2 = 4. * tp * k_x.mapv(|kx| kx.cos()) * k_y.mapv(|ky| ky.cos());
    let dde_dyy_3 = 8. * tpp * k_y.mapv(|ky| (2. * ky).cos());
    let dde_dyy = dde_dyy_1 + dde_dyy_2 + dde_dyy_3;

    // dde_dxy
    let dde_dxdy = -4. * tp * k_x.mapv(|kx| kx.sin()) * k_y.mapv(|ky| ky.sin());

    // Energies derivative squared
    let de_dx_2 = de_dx.mapv(|dedx| dedx.powi(2));
    let de_dy_2 = de_dy.mapv(|dedy| dedy.powi(2));

    // Conductivities arrays init
    let mut sigma_xx = Array1::zeros(mus.len());
    let mut sigma_yy = Array1::zeros(mus.len());
    let mut sigma_xy = Array1::zeros(mus.len());
    let mut n_h = Array1::zeros(mus.len());
    let mut density = Array1::<f64>::zeros(mus.len());

    // ProgressBar init
    let pb = init_progress_bar(String::from("Hall, Density, Sigmas"), mus.len() as u64);

    for idx in 0..a_k.len() {
        // Access spectral function and energy
        let a = a_k.get(&idx).unwrap();
        let a_2 = a.mapv(|ak| ak.powi(2));
        let a_3 = a.mapv(|ak| ak.powi(3));
        let mu = mus[idx];
        let energy_mu = energy - mu;

        // Compute electron density
        let fermi_dirac = energy_mu.mapv(|e| 2. / (1. + ((e * beta).exp())));
        density[idx] = fermi_dirac.sum() / v;

        // Compute sigma_ii conductivities
        sigma_xx[idx] = -(&de_dx_2 * &a_2).sum();
        sigma_yy[idx] = -(&de_dy_2 * &a_2).sum();

        // Compute sigma_ij conductivity
        let c_xy_1 = -2. * &de_dx * &de_dy * &dde_dxdy;
        let c_xy_2 = &de_dx_2 * &dde_dyy;
        let c_xy_3 = &de_dy_2 * &dde_dxx;
        sigma_xy[idx] = -((c_xy_1 + c_xy_2 + c_xy_3) * a_3).sum();

        // Compute Hall number
        n_h[idx] = 6. * v * sigma_xx[idx] * sigma_yy[idx] / sigma_xy[idx];

        // Write to file
        let data_to_file = vec![
            mu,
            1. - density[idx],
            n_h[idx],
            sigma_xx[idx],
            sigma_yy[idx],
            sigma_xy[idx],
            1. - peter_densities[idx],
        ];
        wtr_conductivities.serialize(data_to_file).unwrap();

        // Increment ProgressBar
        pb.inc(1);
    }
    pb.finish();
    wtr_conductivities.flush().unwrap();
    (density, n_h)
}
