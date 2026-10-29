#ifndef PANEL_METHOD_H
#define PANEL_METHOD_H

#include <vector>
#include <Eigen/Dense>

/**
 * @file panel_method.h
 * @brief Panel method de vorticidad lineal (formulacion Katz & Plotkin) con
 *        condicion de Kutta, para flujo potencial incompresible 2D alrededor
 *        de un perfil.
 * @author Carlos Ortega Mantero
 * @date 22/07/2026
 */

/**
 * @struct FreestreamConditions
 * @brief Condiciones de flujo libre para una evaluacion del solver.
 */
struct FreestreamConditions {
    double v_inf;      /**< Velocidad de flujo libre (m/s), normalmente 1.0 en forma adimensional. */
    double alpha_deg;  /**< Angulo de ataque, en grados. */
    double mach = 0.0; /**< Numero de Mach de flujo libre, para correccion de compresibilidad. 0 = incompresible puro. */
};

/**
 * @class PanelMethod
 * @brief Resuelve la distribucion de vorticidad lineal sobre un perfil
 *        discretizado en N paneles, e imponiendo la condicion de Kutta en
 *        el borde de salida.
 */
class PanelMethod {
private:
    // Puntos del contorno del perfil, en orden "herradura" (TE extrados -> LE -> TE intrados),
    // los mismos que ya generas con Parsec::exportToDat pero en memoria (Eigen, no fichero).
    std::vector<Eigen::Vector2d> panel_points;

    FreestreamConditions freestream;

    // Resultado del sistema lineal: gamma en cada nodo (tamano N+1, uno por punto).
    Eigen::VectorXd gamma;

    // Cp en el punto de control de cada panel (tamano N).
    Eigen::VectorXd cp_distribution;

    /**
     * @brief Obtiene el vector normal unitario apuntando hacia el exterior del panel i.
     *        Unifica la geometria para ser robusta ante la orientacion del perfil.
     * @param i Indice del panel.
     * @return Vector normal exterior 2D.
     */
    Eigen::Vector2d getNormal(int i) const;

    /**
     * @brief Obtiene el vector tangente unitario del panel i.
     *        Ortogonal a la normal exterior, formando un sistema dextrogiro consistente.
     * @param i Indice del panel.
     * @return Vector tangente 2D.
     */
    Eigen::Vector2d getTangent(int i) const;

    /**
     * @brief Monta la matriz de coeficientes de influencia (N+1 x N+1) y el
     *        vector independiente, incluyendo la fila de la condicion de Kutta.
     * @return Par {A, b} del sistema lineal a resolver para gamma.
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> buildLinearSystem() const;

    /**
     * @brief Coeficiente de influencia de un panel fuente (nodo j a nodo j+1)
     *        sobre el punto de control de otro panel, para la componente
     *        normal de velocidad inducida por vorticidad lineal.
     */
    double influenceCoefficient(int control_panel_idx, int vortex_node_idx) const;

    /**
     * @brief Coeficiente de influencia de un panel fuente (nodo j a nodo j+1)
     *        sobre el punto de control de otro panel, para la componente
     *        tangencial de velocidad inducida por vorticidad lineal.
     *        Esencial para el calculo global de la velocidad superficial y el Cp.
     *
     * @param control_panel_idx Indice del panel donde se evalua la velocidad (receptor).
     * @param vortex_node_idx Indice del nodo que actua como origen del vortice (emisor).
     * @return Valor de la velocidad tangencial inducida por ese nodo.
     */
    double tangentialInfluenceCoefficient(int control_panel_idx, int vortex_node_idx) const;

public:
    /**
     * @brief Construye el solver a partir de los puntos del contorno (formato
     *        herradura) y las condiciones de flujo libre.
     * @param points Puntos del perfil en orden herradura (mismo que exportToDat).
     * @param conditions Condiciones de flujo libre (v_inf, alpha, mach opcional).
     */
    PanelMethod(std::vector<Eigen::Vector2d> points, FreestreamConditions conditions);

    /**
     * @brief Resuelve el sistema lineal para gamma y calcula la distribucion de Cp.
     */
    void solve();

    /**
     * @brief Distribucion de Cp a lo largo del perfil, ya con correccion de
     *        compresibilidad aplicada si mach > 0 (Prandtl-Glauert).
     */
    [[nodiscard]] const Eigen::VectorXd& getCpDistribution() const;

    /**
     * @brief Coeficiente de sustentacion, integrado a partir de la distribucion de Cp.
     */
    [[nodiscard]] double getCl() const;

    /**
     * @brief Coeficiente de momento respecto al borde de ataque (o cuarto de cuerda, a decidir).
     */
    [[nodiscard]] double getCm() const;

    /**
     * @brief Distribucion de velocidad tangencial en cada punto de control,
     *        necesaria como Ue(s) de entrada para el acoplamiento con
     *        Thwaites/Michel/Head en la siguiente etapa.
     */
    [[nodiscard]] Eigen::VectorXd getEdgeVelocity() const;
};

#endif // PANEL_METHOD_H