#include <boost/numeric/odeint.hpp>
#include <boost/array.hpp>
#include <random>
#include <vector>
#include <math.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <boost/math/constants/constants.hpp>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_errno.h>
#include <mpi.h>
#include <string>

using namespace std;
using namespace boost::numeric::odeint;
using namespace std::chrono;
typedef vector<double> state_type;
typedef runge_kutta_fehlberg78<state_type> stepper;
//typedef runge_kutta_dopri5<state_type> stepper;
const int max_iter = 1000000;
const double gamma_n = -1.83247171e8;
const double gamma_3 = -2.037894569e8;
const double B0 = 3.0e-6;
const double pi_ = boost::math::constants::pi<double>();
//// Zero Phase
//const double theta_n = -0.99650;
//const double phi_n = 1.81020;
//const double theta_He = -1.00759;
//const double phi_He = 1.85511;
//// Pi/4 Phase
//const double theta_n = -0.59186;
//const double phi_n = 1.84162;
//const double theta_He = -1.40374;
//const double phi_He = 1.77373;
// Pi/2 Phase
//const double theta_n = -0.18485;
//const double phi_n = 1.83092;
//const double theta_He = -1.78485;
//const double phi_He = 1.66284;


void gen_rand_ints(const int upper, vector<double>& X) {
	random_device rd;
	mt19937_64 generator(rd());
	uniform_int_distribution<int> distribution(0, upper - 1);
	for (int i = 0; i < X.size(); i++)
		X[i] = distribution(generator);

}

void gen_rand_uniform(const int lower, const int upper, vector<double>& X) {
	random_device rd;
	mt19937_64 generator(rd());
	uniform_real_distribution<double> distribution(lower, upper);
	for (int i = 0; i < X.size(); i++)
		X[i] = distribution(generator);

}

double Hann(const vector<double>& params, const double tau, const double t) {
	return 1e-7 * cos(pi_ * t / tau) * cos(pi_ * t / tau) *
		(params[0] * cos(t * (B0 * gamma_n + params[1] * (t / tau + 4.0 / 3.0 / pi_ * sin(pi_ * t / tau) * (1.0 + 1.0 / 4.0 * cos(pi_ * t / tau)))))
			+ params[2] * cos(t * (B0 * gamma_3 + params[3] * (t / tau + 4.0 / 3.0 / pi_ * sin(pi_ * t / tau) * (1.0 + 1.0 / 4.0 * cos(pi_ * t / tau))))));
}

double Sech(const vector<double>& params, const double t) {
	return 1e-7 * params[0] * cos(pi_ * t / params.back()) * cos(pi_ * t / params.back()) * (1 / cosh(params[1] * t / params.back())) * cos(t * (B0 * gamma_n + params[2] * tanh(params[1] * t / params.back()) / tan(params[1])));
}

double Sine(const vector<double>& params, const double t) {
	double tau = t/params.back();
	return 1e-7 * params[0] * (cos(pi_ * tau) * cos(pi_ * tau)) * sin(2 * pi_ * params[1] * t);
}

double sincos(const vector<double>& params, const double tau, const double t) {
	return 1e-7 * sin(pi_ * t / tau) * sin(params[0] * t) *
		(params[1] * cos(B0 * gamma_n * t + params[2] * cos(params[3] * t)) + params[4] * cos(B0 * gamma_3 * t + params[5] * cos(params[6] * t)));
}

class bloch {

public:

	vector<double> params;
	double tau;

	bloch(double gamma, double B0, vector<double> params, double tau) : gamma(gamma), B0(B0), params(params), tau(tau) { }

	void operator() (const state_type& x, state_type& dxdt, const double t)
	{
		B1 =  sincos(params, tau, t);
		dxdt[0] = gamma * B0 * x[1];
		dxdt[1] = gamma * (B1 * x[2] - B0 * x[0]);
		dxdt[2] = -gamma * B1 * x[1];
	}
	void update_params(const vector<double>& new_params) {
		for (int ii = 0; ii < new_params.size(); ii++)
			params[ii] = new_params[ii];
	}
	void get_params(vector<double>& output_params) {
		for (int ii = 0; ii < params.size(); ii++)
			output_params[ii] = params[ii];
	}
	void print_params() {
		for (int ii = 0; ii < params.size(); ii++)
			if (ii < params.size() - 1 )
				cout << params[ii] << ", ";
			else
				cout << params[ii] << endl;
	}
	void write_params() {
		ofstream outfile;
		outfile.open("data.txt");
		for (int ii = 0; ii < params.size(); ii++)
			outfile << params[ii] << ", ";
		outfile << endl;
		outfile.close();
	}

	~bloch() {}
private:
	double B1 = 0;
	double gamma;
	double B0;
};

void obs(const state_type& x, const double t) {
	cout << x[0] << ", " << x[1] << ", " << x[2] << "\n";
}

state_type calc_M_final(const bloch l, const double B0, const double tol) {
	state_type x = { 0.0, 0.0, 1.0 };
	size_t steps = integrate_adaptive(make_controlled<stepper>(tol, tol), l, x, 0.0, l.tau, 0.001);

	return x;
}

double calc_error(const state_type& M_n, const state_type& M_3, const double phase) {
	//return M_n[2] * M_n[2] + M_3[2] * M_3[2];
	double phi = pi_;
	return M_n[2] * M_n[2] + M_3[2] * M_3[2] + (1 / (2 * pi_) * (phase - phi)) * (1 / (2 * pi_) * (phase - phi));
	//return (M_n[1] - 1) * (M_n[1] - 1) + (M_3[0] - 1) * (M_3[0] - 1);
	
}

double calc_phase(const double Mx_n, const double My_n, const double Mx_3, const double My_3)
{
	double norm = sqrt(Mx_n * Mx_n + My_n * My_n) * sqrt(Mx_3 * Mx_3 + My_3 * My_3);
	double dot = (Mx_n * Mx_3 + My_n * My_3) / norm;
	double angle = acos(dot);

	return angle;
}

double my_f(const gsl_vector* v, void* params)
{
	// Make sure to change number of parameters
	double* p = (double*)params;
	int n_params = (int)p[0];
	vector<double> pars(n_params);
	state_type S_n;
	state_type S_3;
	double E;

	for (int i = 0; i < n_params; i++) {
		pars[i] = gsl_vector_get(v, i);
	}

	bloch n(gamma_n, B0, pars, p[1]);
	bloch He(gamma_3, B0, pars, p[1]);

	state_type M_n = calc_M_final(n, B0, 1e-10);
	state_type M_3 = calc_M_final(He, B0, 1e-10);
	double phase = calc_phase(M_n[0], M_n[1], M_3[0], M_3[1]);

	E = calc_error(M_n, M_3, phase);

	return E;
}

int main() {

	//MPI_Init(NULL, NULL);
	//int my_rank;
	//int num_procs;
	//MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	//for (int i = 0; i < 250; i++) {

	//	string filename = "C:/Users/moran/Desktop/data/";
	//	filename.append(to_string(my_rank));

	//	if (i < 10)
	//		filename.append("000");		
	//	else if (i < 100)
	//		filename.append("00");
	//	else if (i < 1000)
	//		filename.append("0");


	//	filename.append(to_string(i));
	//	filename.append(".txt");

	auto start = high_resolution_clock::now();

		int N = 7;
		size_t iter = 0;
		int Na = 0;
		int Ns = 0;
		const int Na_max = (2 * (N + 1)) * 500;
		const int Ns_max = Na_max / 10;
		double T = 0.01;
		const double step_size = 0.1;
		double E1 = 0.0;
		double E2 = 0.0;
		double step = 0.0;
		double r = 0.0;
		double p = 0.0;
		double phase = 0.0;
		const double threshold = 1e-5;

		vector<double> index(max_iter);
		vector<double> randu1(max_iter);
		vector<double> randu2(max_iter);
		vector<double> E;
		vector<double> params(N, 1);

		double tau = 100e-3;
		//params[1] = 100.0;

		gen_rand_ints(N, index);
		gen_rand_uniform(-1, 1, randu1);
		gen_rand_uniform(0, 1, randu2);

		bloch n(gamma_n, B0, params, tau);
		bloch He(gamma_3, B0, params, tau);

		state_type M_n = calc_M_final(n, B0, 1e-10);
		state_type M_3 = calc_M_final(He, B0, 1e-10);
		phase = calc_phase(M_n[0], M_n[1], M_3[0], M_3[0]);

		E1 = calc_error(M_n, M_3, phase);

		while (true) {

			Na = 0;
			Ns = 0;

			if (iter >= max_iter || E1 < threshold)
				break;

			while (Na < Na_max && Ns < Ns_max) {

				if (iter >= max_iter || E1 < threshold) {
					cout << "\n";
					cout << "**********************************************************************" << "\n";
					cout << "Annealing finished at iteration number: " << iter << "\n";
					cout << "Error: " << E1 << endl;
					cout << "Parameters after annealing: ";
					n.print_params();
					//n.write_params();
					n.get_params(params);
					cout << "**********************************************************************" << "\n";
					cout << endl;
					break;
				}

				if (iter % 100 == 0) {
					cout << E1 << ", " << Na << ", " << Ns << endl;
				}

				step = step_size * randu1[iter];
				params[index[iter]] += step;
				n.update_params(params);
				He.update_params(params);

				state_type M_n = calc_M_final(n, B0, 1e-10);
				state_type M_3 = calc_M_final(He, B0, 1e-10);
				phase = calc_phase(M_n[0], M_n[1], M_3[0], M_3[0]);
				E2 = calc_error(M_n, M_3, phase);

				if (E2 >= E1) {
					r = randu2[iter];
					p = exp(-(E2 - E1) / T);
				}

				if (E2 < E1)
					E1 = E2;
				else if (p >= r) {
					Ns += 1;
					Na += 1;
					E1 = E2;
				}
				else if (p < r) {
					Na += 1;
					params[index[iter]] -= step;
				}
				else
					params[index[iter]] -= step;

				iter += 1;
				//E.push_back(E1);
			}

			T *= 0.5;

		}

		// Initialize some stuff for the final minimization
		double par[] = { N, tau };
		int status;
		int iter_status;
		double size;
		iter = 0;
		const gsl_multimin_fminimizer_type* M = gsl_multimin_fminimizer_nmsimplex2;
		gsl_multimin_fminimizer* s = NULL;
		gsl_vector* ss, * x;
		gsl_multimin_function minex_func;
		gsl_set_error_handler_off();

		// Set the starting parameters from the annealing
		x = gsl_vector_alloc(N);
		for (int i = 0; i < N; i++)
			gsl_vector_set(x, i, params[i]);

		// Set the initial step size for the minimization
		ss = gsl_vector_alloc(N);
		gsl_vector_set_all(ss, 0.001);

		// Initialize more stuff for the minimization
		minex_func.n = N;
		minex_func.f = my_f;
		minex_func.params = par;
		s = gsl_multimin_fminimizer_alloc(M, N);
		gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

		// Loop does the actual minimization
		do
		{
			iter++;
			iter_status = gsl_multimin_fminimizer_iterate(s);

			size = gsl_multimin_fminimizer_size(s);
			status = gsl_multimin_test_size(size, 1e-10);

			if (status == GSL_SUCCESS || iter_status != 0 || s->fval < 1e-16)
			{
				printf("converged to minimum at\n");
				break;
			}

			printf("%5d f() = %10.8e size = %10.8e\n", (int)iter, s->fval, size);

		} while (status == GSL_CONTINUE && iter < 1000);


		// Dump parameters to a text file
		ofstream outfile("C:/Users/moran/Desktop/params.txt");
		outfile << fixed;
		outfile << setprecision(16);
		// change N to N+1 if pulse duration is a paramter
		for (int i = 0; i < N; i++)
			outfile << setprecision(8) << gsl_vector_get(s->x, i) << ", ";
		outfile.close();

		for (int i = 0; i < N; i++)
			cout << setprecision(6) << gsl_vector_get(s->x, i) << "\t";
		cout << "\n";

		// Release memory used for minimization
		gsl_vector_free(x);
		gsl_vector_free(ss);
		gsl_multimin_fminimizer_free(s);

	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	cout << "Duration: " << duration.count() << "\n";

	//MPI_Finalize();

}
