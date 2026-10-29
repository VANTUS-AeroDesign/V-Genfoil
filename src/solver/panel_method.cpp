//
// Created by carlo on 22/07/2026.
//

#include <cmath>
#include <numbers>
#include <utility>
#include <iostream>
#include <algorithm>
#include "solver/panel_method.h"

// ==============================================================================
// 1. CONSTRUCTOR Y GEOMETRÍA UNIFICADA
// ==============================================================================
PanelMethod::PanelMethod(std::vector<Eigen::Vector2d> points, FreestreamConditions conditions) {
    freestream = conditions;
    panel_points = std::move(points);
    int N = panel_points.size() - 1;

    // Calcular la orientación global (Área firmada) al instanciar
    double area2 = 0.0;
    for (int i = 0; i < N; ++i) {
        area2 += panel_points[i].x() * panel_points[i+1].y()
               - panel_points[i+1].x() * panel_points[i].y();
    }
    // Si viene en sentido horario (CW), lo invertimos a antihorario (CCW)
    if (area2 < 0.0) {
        std::reverse(panel_points.begin(), panel_points.end());
    }
    gamma = Eigen::VectorXd::Zero(N + 1);
    cp_distribution = Eigen::VectorXd::Zero(N);
}

Eigen::Vector2d PanelMethod::getNormal(int i) const {
    Eigen::Vector2d p1 = panel_points[i];
    Eigen::Vector2d p2 = panel_points[i+1];
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double length = std::hypot(dx, dy);
    // Garantiza que la normal siempre apunta hacia el exterior geométrico
    return Eigen::Vector2d( dy / length,-dx / length);
}

Eigen::Vector2d PanelMethod::getTangent(int i) const {
    Eigen::Vector2d p1 = panel_points[i];
    Eigen::Vector2d p2 = panel_points[i+1];
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double length = std::hypot(dx, dy);
    // Ortogonal a la normal exterior, formando un sistema dextrógiro consistente (t, n)
    return Eigen::Vector2d(dx / length, dy / length);
}

// ==============================================================================
// 2. ENSAMBLAJE DEL SISTEMA MATRICIAL
// ==============================================================================
std::pair<Eigen::MatrixXd, Eigen::VectorXd> PanelMethod::buildLinearSystem() const {
    int N = panel_points.size() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N+1, N+1);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N+1);

    double alpha_rad = freestream.alpha_deg * std::numbers::pi / 180.0;
    Eigen::Vector2d V_inf(std::cos(alpha_rad), std::sin(alpha_rad));
    V_inf *= freestream.v_inf;

    for (int i = 0; i < N; i++) {
        b(i) = -V_inf.dot(getNormal(i)); // Uso de geometría unificada

        for (int j = 0; j <= N; ++j) {
            A(i,j) = influenceCoefficient(i, j);
        }
    }

    A(N,0) = 1.0;
    A(N,N) = 1.0;
    b(N) = 0.0;

    return {A,b};
}

// ==============================================================================
// 3. COEFICIENTES DE INFLUENCIA
// ==============================================================================
double PanelMethod::influenceCoefficient(int control_panel_idx, int vortex_node_idx) const {
    int N = panel_points.size() - 1;
    double influence = 0.0;

    auto calcPanelInfluence = [&](int panel_idx, bool is_second_node) {
        if (control_panel_idx == panel_idx) return 0.0;

        Eigen::Vector2d p1 = panel_points[panel_idx];
        Eigen::Vector2d p2 = panel_points[panel_idx+1];
        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        double length = std::hypot(dx,dy);
        double cos_th = dx/length;
        double sin_th = dy/length;

        Eigen::Vector2d p_i1 = panel_points[control_panel_idx];
        Eigen::Vector2d p_i2 = panel_points[control_panel_idx+1];
        Eigen::Vector2d control = 0.5*(p_i1 + p_i2);

        // Uso de geometría unificada
        Eigen::Vector2d normal_i = getNormal(control_panel_idx);

        double xt = control.x() - p1.x();
        double yt = control.y() - p1.y();
        double x_star = xt*cos_th+yt*sin_th;
        double y_star = -xt*sin_th+yt*cos_th;

        double r1_sqr = x_star*x_star + y_star*y_star;
        double r2_sqr = (x_star - length)*(x_star - length) + (y_star) * (y_star);
        double term1 = 0.5*std::log(r1_sqr/r2_sqr);
        double term2 = std::atan2(y_star, x_star - length) - std::atan2(y_star, x_star);

        double u_star = 0.0;
        double w_star = 0.0;

        if(!is_second_node) {
            u_star = (1.0 - x_star / length) * term2 + (y_star / length) * term1;
            w_star = -(1.0 - x_star / length) * term1 + (y_star / length) * term2 - 1.0;
        } else {
            u_star = (x_star / length) * term2 - (y_star/length) * term1;
            w_star = -(x_star / length) * term1 - (y_star/length) * term2 + 1.0;
        }

        u_star /= (2.0 * std::numbers::pi);
        w_star /= (2.0 * std::numbers::pi);

        double u_global = u_star * cos_th - w_star * sin_th;
        double w_global = u_star * sin_th + w_star * cos_th;

        return u_global * normal_i.x() + w_global * normal_i.y();
    };

    if (vortex_node_idx > 0) influence += calcPanelInfluence(vortex_node_idx -1, true);
    if (vortex_node_idx < N) influence += calcPanelInfluence(vortex_node_idx, false);
    return influence;
}

double PanelMethod::tangentialInfluenceCoefficient(int control_panel_idx, int vortex_node_idx) const {
    int N = panel_points.size() - 1;
    double influence = 0.0;

    auto calcPanelInfluence = [&](int panel_idx, bool is_second_node) {
        if (control_panel_idx == panel_idx) return 0.0;

        Eigen::Vector2d p1 = panel_points[panel_idx];
        Eigen::Vector2d p2 = panel_points[panel_idx+1];
        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        double length = std::hypot(dx,dy);
        double cos_th = dx/length;
        double sin_th = dy/length;

        Eigen::Vector2d p_i1 = panel_points[control_panel_idx];
        Eigen::Vector2d p_i2 = panel_points[control_panel_idx+1];
        Eigen::Vector2d control = 0.5*(p_i1 + p_i2);

        // Uso de geometría unificada
        Eigen::Vector2d tangent_i = getTangent(control_panel_idx);

        double xt = control.x() - p1.x();
        double yt = control.y() - p1.y();
        double x_star = xt*cos_th+yt*sin_th;
        double y_star = -xt*sin_th+yt*cos_th;

        double r1_sqr = x_star*x_star + y_star*y_star;
        double r2_sqr = (x_star - length)*(x_star - length) + (y_star) * (y_star);
        double term1 = 0.5*std::log(r1_sqr/r2_sqr);
        double term2 = std::atan2(y_star, x_star - length) - std::atan2(y_star, x_star);

        double u_star = 0.0;
        double w_star = 0.0;

        if(!is_second_node) {
            u_star = (1.0 - x_star / length) * term2 + (y_star / length) * term1;
            w_star = -(1.0 - x_star / length) * term1 + (y_star / length) * term2 - 1.0;
        }else {
            u_star = (x_star / length) * term2 - (y_star/length) * term1;
            w_star = -(x_star / length) * term1 - (y_star/length) * term2 + 1.0;
        }

        u_star /= (2.0 * std::numbers::pi);
        w_star /= (2.0 * std::numbers::pi);

        double u_global = u_star * cos_th - w_star * sin_th;
        double w_global = u_star * sin_th + w_star * cos_th;

        return u_global * tangent_i.x() + w_global * tangent_i.y();
    };

    if (vortex_node_idx > 0) influence += calcPanelInfluence(vortex_node_idx -1, true);
    if (vortex_node_idx < N) influence += calcPanelInfluence(vortex_node_idx, false);
    return influence;
}

// ==============================================================================
// 4. SOLVER PRINCIPAL
// ==============================================================================
void PanelMethod::solve() {
    int N = panel_points.size() -1;

    auto[A, b] = buildLinearSystem();
    gamma = A.partialPivLu().solve(b);

    double alpha_rad = freestream.alpha_deg * std::numbers::pi / 180.0;
    Eigen::Vector2d V_inf(std::cos(alpha_rad), std::sin(alpha_rad));
    V_inf *= freestream.v_inf;

    double beta = 1.0;
    if (freestream.mach > 0.0 && freestream.mach < 0.9) {
        beta = std::sqrt(1.0 - freestream.mach * freestream.mach);
    }

    for (int i = 0; i < N; ++i) {
        double v_inf_t = V_inf.dot(getTangent(i)); // Uso de geometría unificada

        double v_induced = 0.0;
        for (int j = 0; j <= N; ++j) {
            v_induced += tangentialInfluenceCoefficient(i, j) * gamma(j);
        }

        double gamma_mid = -0.5 * (gamma(i) + gamma(i+1));

        // La autoinducción proyectada matemáticamente sobre la tangente adaptativa
        double v_self = 0.5 * gamma_mid;

        double v_tangent = v_inf_t + v_induced + v_self;
        double cp_inc = 1.0 - std::pow(v_tangent / freestream.v_inf, 2);
        cp_distribution(i) = cp_inc / beta;
    }
}

// ==============================================================================
// 5. GETTERS AERODINÁMICOS
// ==============================================================================
const Eigen::VectorXd& PanelMethod::getCpDistribution() const {
    return cp_distribution;
}

Eigen::VectorXd PanelMethod::getEdgeVelocity() const {
    int N = panel_points.size() -1;
    Eigen::VectorXd u_e = Eigen::VectorXd::Zero(N);

    double alpha_rad = freestream.alpha_deg * std::numbers::pi / 180.0;
    Eigen::Vector2d V_inf(std::cos(alpha_rad), std::sin(alpha_rad));
    V_inf *= freestream.v_inf;

    for (int i = 0; i < N; ++i) {
        double v_inf_t = V_inf.dot(getTangent(i));
        double v_induced = 0.0;
        for (int j = 0; j <= N; ++j) {
            v_induced += tangentialInfluenceCoefficient(i, j) * gamma(j);
        }

        double gamma_mid = -0.5 * (gamma(i) + gamma(i+1));
        double v_self = 0.5 * gamma_mid;

        u_e(i) = v_inf_t + v_induced + v_self;
    }
    return u_e;
}

double PanelMethod::getCl() const {
    int N = panel_points.size() -1;

    double Cl_presiones = 0.0;
    double Gamma_total = 0.0;

    double alpha_rad = freestream.alpha_deg * std::numbers::pi / 180.0;
    double cos_a = std::cos(alpha_rad);
    double sin_a = std::sin(alpha_rad);

    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d p1 = panel_points[i];
        Eigen::Vector2d p2 = panel_points[i+1];
        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        double length = std::hypot(dx,dy);

        Eigen::Vector2d normal = getNormal(i); // Uso de geometría unificada

        double cfx = -cp_distribution(i) * length * normal.x();
        double cfy = -cp_distribution(i) * length * normal.y();

        Cl_presiones += cfy * cos_a - cfx * sin_a;
        Gamma_total += 0.5 * (gamma(i) + gamma(i+1)) * length;
    }

    double Cl_KJ = 2.0 * Gamma_total / freestream.v_inf;

    std::cout << "\n[RESULTADOS FINALES Cl]\n";
    std::cout << " -> Cl (Integrando Cp)   : " << Cl_presiones << "\n";
    std::cout << " -> Cl (Kutta-Joukowski) : " << Cl_KJ << "\n";

    return Cl_presiones;
}

double PanelMethod::getCm() const {
    int N = panel_points.size() -1;
    double Cm = 0.0;
    Eigen::Vector2d ref_point(0.25, 0.0);

    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d p1 = panel_points[i];
        Eigen::Vector2d p2 = panel_points[i+1];
        Eigen::Vector2d control_pt = 0.5 * (p1 + p2);

        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        double length = std::hypot(dx,dy);

        Eigen::Vector2d normal = getNormal(i); // Uso de geometría unificada

        double cfx = -cp_distribution(i) * length * normal.x();
        double cfy = -cp_distribution(i) * length * normal.y();

        double rx = control_pt.x() - ref_point.x();
        double ry = control_pt.y() - ref_point.y();

        Cm += rx * cfy - ry * cfx;
    }
    return Cm;
}