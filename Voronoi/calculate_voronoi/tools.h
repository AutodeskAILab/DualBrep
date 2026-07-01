#ifndef TOOL_H
#define TOOL_H

#include <glog/logging.h>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/IO/io.h>
#include <CGAL/Linear_cell_complex_for_combinatorial_map.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Triangulation_3_to_lcc.h>
#include <CGAL/point_generators_3.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Point_set_3<K::Point_3> Point_set_3;
typedef CGAL::Triangle_3<K> Triangle_3;

typedef boost::tuple<K::Point_3, int>                           Point_and_int;
typedef CGAL::Search_traits_3<K>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
    CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
    Traits_base>                                              Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
typedef K_neighbor_search::Tree                             KDTree;
typedef K_neighbor_search::Distance                         Distance;


Point_set_3 sample_points(const std::vector<Triangle_3>& v_mesh, const int v_num_points, bool at_least_one_point);

Point_set_3 sample_poisson_points(const std::vector<Triangle_3>& v_mesh, const double radius, const int init_points);

std::vector<int> deduplicate_points(const std::vector<Point_and_int>& v_points, const double v_sq_threshold=1e-12);

inline std::vector<int> merge_duplicate_points_in_polygon_soup(std::vector<K::Point_3>& v_points, const double v_threshold = 1e-12)
{
    std::vector<K::Point_3> pp;
    std::vector<int> indexes;

    std::vector<std::vector<bool>> is_equal(v_points.size(), std::vector<bool>(v_points.size(), false));

    #pragma omp parallel for
    for (int i = 0; i < v_points.size(); ++i) {
        for (int j = i+1; j < v_points.size(); ++j) {
            if (abs(v_points[i].x() - v_points[j].x()) < v_threshold && abs(v_points[i].y() - v_points[j].y()) < v_threshold && abs(v_points[i].z() - v_points[j].z()) < v_threshold) {
                is_equal[i][j] = true;
            }
        }
    }
    for (int i = 0; i < v_points.size(); ++i) {
        int id_exist = -1;
        for (int j = 0; j < pp.size(); ++j) {
            if (is_equal[i][j] || is_equal[j][i]) {
                id_exist = j;
                break;
            }
        }
        if (id_exist == -1) {
            pp.push_back(v_points[i]);
            indexes.push_back(pp.size() - 1);
        } else {
            indexes.push_back(id_exist);
        }
    }
    std::swap(v_points, pp);
    return indexes;
}

inline void write_ply(const std::vector<Triangle_3>& tri_out, const std::string& v_file)
{
    std::vector<K::Point_3> pp;
    std::vector<std::vector<int>> ff;
    for (const auto& tri : tri_out) {
        std::vector<int> f;
        for (int i = 0; i < 3; ++i) {
            pp.push_back(tri.vertex(i));
            f.push_back(pp.size() - 1);
        }
        ff.push_back(f);
    }

    const int original_size = pp.size();
    LOG(INFO) << "Reduced vertices by CGAL: " << CGAL::Polygon_mesh_processing::merge_duplicate_points_in_polygon_soup(pp, ff);
    {
        std::vector<Point_and_int> points(pp.size());
		for (int i = 0; i < pp.size(); ++i)
			points[i] = Point_and_int(pp[i], i);
        KDTree tree(points.begin(), points.end());
        tree.build();

        std::vector<int> is_duplicate(points.size(), -1);
        #pragma omp parallel for
        for (int i = 0; i < points.size(); ++i)
        {
            K_neighbor_search search(tree, boost::get<0>(points[i]), 2);
            Distance tr_dist;
            for(int j=0;j<2;++j)
            {
                const auto& target = search.begin() + j;
                const int target_id = boost::get<1>(target->first);
                if (i <= target_id)
                    continue;
                if (tr_dist.inverse_of_transformed_distance(target->second) < 1e-6)
                    is_duplicate[i] = target_id;
            }
        }

        pp.erase(std::remove_if(pp.begin(), pp.end(), [&is_duplicate, &pp](const auto& item)
            {
                return is_duplicate[&item - &pp[0]] != -1;
            }), pp.end());

		std::unordered_map<int, int> indexes;
        int cur_size = 0;
        for (int i = 0; i < points.size(); ++i)
        {
            if (is_duplicate[i] != -1)
				continue;
            indexes[i] = cur_size++;
        }

		for (int i = 0; i < points.size(); ++i)
        {
			if (is_duplicate[i] == -1)
				continue;

            int a = is_duplicate[i];
            while(is_duplicate[a]!=-1)
				a = is_duplicate[a];
	        indexes[i] = indexes[a];
		}

        for (auto& f : ff)
		    for (auto& i : f)
		        i = indexes[i];
    }

    // const auto& indexes = merge_duplicate_points_in_polygon_soup(pp);
    LOG(INFO) << "Reduced points: " << original_size - pp.size();
    LOG(INFO) << "Reduced polygons: " << CGAL::Polygon_mesh_processing::merge_duplicate_polygons_in_polygon_soup(pp, ff);
    LOG(INFO) << "Now #v: " << pp.size() << "; #f: " << ff.size();
    CGAL::IO::write_polygon_soup(v_file, pp, ff);
}

#endif
