//
// Created by Carlos on 20/07/2026.
//
#include  <cassert>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>

#include "geometry/parsec.h"

#include <iomanip>

namespace {

    //Reparte y_te/dy_te/alpha_te/beta_te (bisectriz + cuña) en condicciones
    // de contorno independiente para cada superficie
    struct TrailingEdgeBC {
        double y_upper = 0.0, th_upper = 0.0;
        double y_lower = 0.0, th_lower = 0.0;
    };

    TrailingEdgeBC splitTrailingEdge(double y_te, double dy_th, double alpha_te, double beta_te) {
        TrailingEdgeBC bc;
        bc.y_upper = y_te + dy_th /2.0;
        bc.th_upper = alpha_te - beta_te /2.0;
        bc.y_lower = y_te - dy_th /2.0;
        bc.th_lower = alpha_te + beta_te /2.0;
        return bc;
    }

} //namespace

Parsec::Parsec(const ParsecParameters &parameters) : params(parameters) {
    TrailingEdgeBC bc = splitTrailingEdge(
        params.y_te, params.dy_te, params.alpha_te, params.beta_te);

    a_upper = calculateCoefficients(
        params.r_le, params.x_up, params.y_up, params.d2y_up,
        bc.y_upper, bc.th_upper, /*isUpper=*/true);

    a_lower = calculateCoefficients(
    params.r_le, params.x_lo, params.y_lo, params.d2y_lo,
    bc.y_lower, bc.th_lower, /*isUpper=*/false);

}

std::array<double, 6> Parsec::calculateCoefficients(
    double r_le, double x_m, double y_m, double d2y_m,
    double y_te_surf, double th_te_surf, bool isUpper) {

    assert(x_m > 0.0 && x_m < 1.0 && "x_m debe de estar en (0, 1)");

    // a1 no forma parte del sistema lineal: sale directo del radio LE
    // Signo positivo en el extradós, negativo en el intradós
    const double a1 = (isUpper ? 1.0 : -1.0)*std::sqrt(2.0*r_le);

    const double s05 = std::sqrt(x_m);        // x^0.5
    const double s15 = s05 * x_m;             // x^1.5
    const double s25 = s15 * x_m;             // x^2.5
    const double s35 = s25 * x_m;             // x^3.5
    const double s45 = s35 * x_m;             // x^4.5

    Eigen::Matrix<double, 5, 5> A;
    Eigen::Matrix<double, 5, 1> b;

    // Fila 1: z(x_m) = y_m  -> a2 x^1.5 + a3 x^2.5 + a4 x^3.5 + a5 x^4.5 + a6 x^5.5
    A(0,0)=s15; A(0,1)=s25; A(0,2)=s35; A(0,3)=s45; A(0,4)=s45*x_m;
    b(0) = y_m - a1 * s05;

    // Fila 2: z(1) = y_te_surf
    A(1,0)=1; A(1,1)=1; A(1,2)=1; A(1,3)=1; A(1,4)=1;
    b(1) = y_te_surf - a1;

    // Fila 3: z'(x_m) = 0 -> 1.5 a2 x^0.5 + 2.5 a3 x^1.5 + 3.5 a4 x^2.5 + 4.5 a5 x^3.5 + 5.5 a6 x^4.5
    A(2,0)=1.5*s05; A(2,1)=2.5*s15; A(2,2)=3.5*s25; A(2,3)=4.5*s35; A(2,4)=5.5*s45;
    b(2) = -0.5 * a1 / s05;

    // Fila 4: z'(1) = tan(th_te_surf)
    A(3,0)=1.5; A(3,1)=2.5; A(3,2)=3.5; A(3,3)=4.5; A(3,4)=5.5;
    b(3) = std::tan(th_te_surf) - 0.5 * a1;

    // Fila 5: z''(x_m) = d2y_m -> 0.75 a2 x^-0.5 + 3.75 a3 x^0.5 + 8.75 a4 x^1.5 + 15.75 a5 x^2.5 + 24.75 a6 x^3.5
    A(4,0)=0.75/s05; A(4,1)=3.75*s05; A(4,2)=8.75*s15; A(4,3)=15.75*s25; A(4,4)=24.75*s35;
    b(4) = d2y_m + 0.25 * a1 / s15;

    Eigen::Matrix<double, 5, 1> coeffs = A.partialPivLu().solve(b);
    return {a1, coeffs(0), coeffs(1), coeffs(2), coeffs(3), coeffs(4)};
}

double Parsec::getY(double x, bool isUpper) const {
    assert(x >= 0.0 && x <= 1.0);
    if (x == 0.0) return 0.0;

    const std::array<double, 6>& a = isUpper ? a_upper : a_lower;
    const double sx = std::sqrt(x);

    double x_pow = sx;
    double z = 0.0;
    for (int i = 0; i < 6; ++i) {
        z += a[i] * x_pow;
        x_pow *= x;                 // avanza el exponente 1.0: 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    }
    return z;
}

void Parsec::exportToDat(const std::string &filename, int num_points) const {
    if (num_points <= 1) return;

    struct Point { double x, z; };
    std::vector<Point> points;
    points.reserve(2 * num_points - 1);

    // 1. Generar la nube de puntos completa
    for (int i = num_points - 1; i >= 0; --i) {
        double x = static_cast<double>(i) / (num_points - 1);
        points.push_back({x, getY(x, true)});
    }
    for (int i = 1; i < num_points; ++i) {
        double x = static_cast<double>(i) / (num_points - 1);
        points.push_back({x, getY(x, false)});
    }

    // 2. Calcular área firmada (Shoelace formula)
    double area2 = 0.0;
    size_t n = points.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area2 += (points[i].x * points[j].z) - (points[j].x * points[i].z);
    }

    // 3. Si es horario (area2 < 0), invertir para forzar CCW
    if (area2 < 0.0) {
        std::reverse(points.begin(), points.end());
    }

    // 4. Escribir al archivo .dat
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "V-Genfoil_PARSEC_Geometria\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& pt : points) {
        file << std::setw(12) << pt.x << std::setw(12) << pt.z << "\n";
    }
}
