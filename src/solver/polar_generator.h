//
// Created by carlo on 29/10/2026.
//

/**
 * @file polar_generator.h
 * @brief Generador de polares aerodinámicas para V-Genfoil.
 * @details Modulo encargado de automatizar barridos de ángulo de ataque (alpha-sweep)
 *          mediante el método de paneles y exportar los coeficientes resultantes.
 */

#ifndef V_GENFOIL_POLAR_GENERATOR_H
#define V_GENFOIL_POLAR_GENERATOR_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "solver/panel_method.h"

/**
 * @struct PolarConfig
 * @brief Configuración de parámetros para el barrido aerodinámico.
 */
struct PolarConfig {
    double alpha_start = -5.0; /**< Ángulo de ataque inicial [grados]. */
    double alpha_end   = 15.0;  /**< Ángulo de ataque final [grados]. */
    double alpha_step  = 0.5;   /**< Paso de incremento de ángulo [grados]. */
    double v_inf       = 10.0;  /**< Velocidad de la corriente libre [m/s]. */
    double mach        = 0.0;   /**< Número de Mach (0.0 para flujo incompresible). */
};

/**
 * @struct PolarPoint
 * @brief Contenedor de datos aerodinámicos calculados para un ángulo de ataque específico.
 */
struct PolarPoint {
    double alpha_deg;  /**< Ángulo de ataque evaluado [grados]. */
    double cl_cp;      /**< Coeficiente de sustentación ($C_l$) obtenido por integración de $C_p$. */
    double cl_kj;      /**< Coeficiente de sustentación ($C_l$) obtenido por Kutta-Joukowski. */
    double cm;         /**< Coeficiente de momento de cabeceo ($C_m$) respecto a $0.25c$. */
};

/**
 *@class PolarGenerator
 *@brief Manejador de simulación para la generación de polares y exportación de datos
 **/
class PolarGenerator {
    public:
    /**@brief Constructor de PolarGenerator
     *@param airfoil_points Vector de puntos 2D (Eigen::Vector2d) que define la geometría de perfil
     *@param config Configuración con los límites y velocidad de la corriente libre para el barrido
     **/
    PolarGenerator(std::vector<Eigen::Vector2d> airfoil_points, PolarConfig config);

    /**
     *@brief Ejecuta el barrido completo de ángulo de ataque
     *@details Resuelve iterativamente el sistema matricial del PanelMethod para cada paso
     *         de alpha y almacena los coeficientes en la lista interna de resultados
     **/
    void runSweep();

    /**
     *@brief Exporta los resultados calculados a un archivo plano separado por comas (CSV)
     *@param filename Ruta o nombre del archivo de salida (ej. "polar_naca2412.csv")
     *@return true Si el archivo se creó y escribió correctamente, false en caso contrario
     **/
    bool exportToCSV(const std::string& filename) const;

    /**
     *@brief Exporta los resultados calculados a un archivo formateado en JSON
     *@param filename Ruta o nombre del archivo de salida (ej. "polar_naca2412.json")
     *@return true Si el archivo se creó y escribió correctamente, false en caso contrario
     **/
    bool exportToJSON(const std::string& filename) const;

    /**
     *@brief Obtiene la referencia al vector de resultado de la polar
     *@return Referencia constante al vector que contiene todos los puntos PolarPoint generados.
     **/
    const std::vector<PolarPoint>& getResults()const {return results;}

    private:
    std::vector<Eigen::Vector2d> points; /**< Nube de puntos 2D del perfil aerodinámico **/
    PolarConfig config;                  /**< Configuración del barrido de simulación */
    std::vector<PolarPoint> results;     /**< Lista de coeficientes aerodinámicos resultantes */

};

#endif //V_GENFOIL_POLAR_GENERATOR_H
