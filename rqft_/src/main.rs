mod file_utils;
mod qft_funcs;

use crate::qft_funcs::{get_conductivities, get_model_params, get_spectral_weight, Model};
use ndarray::array;

fn main() {
    // Defining path to model parameters and data files
    let model_path = String::from("./src/model.dat");

    // Floats parameters
    let params: Model = get_model_params(&model_path).unwrap();
    let _peters = params.peter_data().unwrap();

    println!("{}", params);

    let eta: f64 = params.eta;
    let omega: f64 = params.omega;
    let t: f64 = params.t;
    let tp: f64 = params.tp;
    let tpp: f64 = params.tpp;
    let _mu: f64 = params.mu;
    let beta: f64 = params.beta;

    // 1D arrays
    let mus = &params.mu_array();
    let _test_mus_36 = array![-1.3, -1.3, -1.0, -0.75, -0.4, -0.4, 0.0];
    let _test_mus_64 = array![-1.3, -0.8, -0.55, -0.75, -0.1, 0.0];

    // 2D arrays
    let k_x = &params.k_grid('x');
    let k_y = &params.k_grid('y');
    let energy = &params.energy(k_x, k_y);

    // Test 'get_spectral_weight' fn
    let _a_k = get_spectral_weight(energy, eta, omega, mus, (true, 0.0));

    // Test 'get_conductivities' fn
    let hop_amps = array![t, tp, tpp];
    let _peter_n_36 = array![0.667, 0.722, 0.778, 0.833, 0.889, 0.944, 1.0];
    let _peter_n_64 = array![0.75, 0.8125, 0.875, 0.9375, 1.0];

    let (_n, _n_h) = get_conductivities(
        energy,
        (k_x, k_y),
        &_peters,
        &_test_mus_64,
        &hop_amps,
        beta,
        &_peter_n_64,
    );
}
