//
// Created by Lukasz on 26.04.2021.
//
#ifndef ISING2021_UTILS_H
#define ISING2021_UTILS_H

#include <iostream>
#include "pcg_random.hpp"
#include <random>
#include <string>
#include <sstream>
#include <iomanip>      // std::setprecision




int getRandomChoice(pcg64 &rng, std::uniform_int_distribution<int> &dist);
std::string generateFileName(const std::string& Quantity, int L, int MCS, int warmingTime, int saveMode, double T, const std::string& format);

#endif //ISING2021_UTILS_H
