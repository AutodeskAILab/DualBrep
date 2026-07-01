#ifndef GEOM_VORONOI_H
#define GEOM_VORONOI_H

#include "tools.h"

void compute_voronoi(const Point_set_3& sample_points, const std::vector<std::vector<bool>>& v_conn, const fs::path& v_root);

#endif
