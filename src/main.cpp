#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>
#include <iomanip>
#include <Eigen/Dense>

#include "solver/panel_method.h"

// Generador de perfiles NACA 4-digits
std::vector<Eigen::Vector2d> generateNACA4(double m, double p, double t, int N_panels, bool invert = false) {
    std::vector<Eigen::Vector2d> upper_pts;
    std::vector<Eigen::Vector2d> lower_pts;
    int N_half = N_panels / 2;

    for (int i = 0; i <= N_half; ++i) {
        double beta = std::numbers::pi * i / N_half;
        double x = 0.5 * (1.0 - std::cos(beta));

        double yt = 5.0 * t * (
            0.2969 * std::sqrt(x) - 0.1260 * x -
            0.3516 * std::pow(x, 2) + 0.2843 * std::pow(x, 3) - 0.1036 * std::pow(x, 4)
        );

        double yc = 0.0;
        double dyc_dx = 0.0;

        if (p > 0.0) {
            if (x < p) {
                yc = (m / (p * p)) * (2.0 * p * x - x * x);
                dyc_dx = (2.0 * m / (p * p)) * (p - x);
            } else {
                yc = (m / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x);
                dyc_dx = (2.0 * m / ((1.0 - p) * (1.0 - p))) * (p - x);
            }
        }

        double theta = std::atan(dyc_dx);

        double xu = x - yt * std::sin(theta);
        double yu = yc + yt * std::cos(theta);
        double xl = x + yt * std::sin(theta);
        double yl = yc - yt * std::cos(theta);

        if (invert) {
            yu = -yu;
            yl = -yl;
        }

        upper_pts.push_back(Eigen::Vector2d(xu, yu));
        lower_pts.push_back(Eigen::Vector2d(xl, yl));
    }

    std::vector<Eigen::Vector2d> points;
    for (int i = N_half; i >= 0; --i) points.push_back(upper_pts[i]);
    for (int i = 1; i <= N_half; ++i) points.push_back(lower_pts[i]);

    return points;
}

void runTest(const std::string& test_name, const std::vector<Eigen::Vector2d>& points, double alpha_deg) {
    FreestreamConditions conditions;
    conditions.v_inf = 10.0;
    conditions.alpha_deg = alpha_deg;
    conditions.mach = 0.0;

    PanelMethod solver(points, conditions);
    solver.solve();

    std::cout << "\n==================================================\n";
    std::cout << " TEST: " << test_name << " (alpha = " << alpha_deg << " deg)\n";
    std::cout << "==================================================\n";

    double cl = solver.getCl();
    double cm = solver.getCm();

    std::cout << " -> Cm final (ref 0.25c) : " << cm << "\n";
    (void)cl; // Evita el warning de variable no usada
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n==================================================\n";
    std::cout << "     SUITE DE PRUEBAS COMPLETA V-GENFOIL          \n";
    std::cout << "==================================================\n";

    // 1. NACA 0012 Simétrico a 0 deg (Debe dar Cl = 0, Cm = 0)
    auto naca0012 = generateNACA4(0.00, 0.0, 0.12, 100, false);
    runTest("NACA 0012 (Simetrico)", naca0012, 0.0);

    // 2. NACA 2412 Normal a 0 deg (Debe dar Cl positivo ~ +0.26)
    auto naca2412 = generateNACA4(0.02, 0.4, 0.12, 100, false);
    runTest("NACA 2412 (Normal)", naca2412, 0.0);

    // 3. NACA 2412 Invertido a 0 deg (Debe dar Cl negativo exacto ~ -0.26)
    auto naca2412_inv = generateNACA4(0.02, 0.4, 0.12, 100, true);
    runTest("NACA 2412 (Invertido)", naca2412_inv, 0.0);

    // 4. NACA 0012 a 5 deg (Comprueba sustentación por incidencia ~ +0.55)
    runTest("NACA 0012 (alpha = 5 deg)", naca0012, 5.0);

    std::cout << "\n==================================================\n";
    std::cout << "           TODOS LOS TESTS FINALIZADOS            \n";
    std::cout << "==================================================\n\n";

    return 0;
}