#include "geom_voronoi.h"

#include <geogram/delaunay/delaunay_3d.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/convex_cell.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <glog/logging.h>


void compute_voronoi(const Point_set_3& sample_points, const std::vector<std::vector<bool>>& v_conn, const fs::path& v_root)
{
	std::vector<double> vertices;
	for (int i = 0; i < sample_points.number_of_points(); ++i)
	{
		vertices.push_back(sample_points.point(i).x());
		vertices.push_back(sample_points.point(i).y());
		vertices.push_back(sample_points.point(i).z());
	}
	const auto& index_map = sample_points.property_map<int>("primitive_index").first;

	GEO::initialize();

	GEO::PeriodicDelaunay3d* delaunay_ = new GEO::PeriodicDelaunay3d(false);
	delaunay_->set_stores_neighbors(true);
	delaunay_->set_keeps_infinite(true);
	delaunay_->set_vertices(sample_points.number_of_points(), vertices.data());
	delaunay_->compute();

	GEO::Mesh m;

	VBW::ConvexCell C;

	// int debug_id = -1;
	// for (int i = 0; i < sample_points.size(); i++)
	// {
	// 	if(CGAL::squared_distance(sample_points.point(i), K::Point_3(0.222539, 0.808, 0.845367)) < 1e-4)
	// 	{
	// 		debug_id = i;
	// 		LOG(INFO) << "Debug point: " << debug_id;
	// 		break;
	// 	}
	// }

	std::vector<std::vector<K::Point_3>> tris(delaunay_->nb_vertices());
	for (GEO::index_t i = 0; i < delaunay_->nb_vertices(); ++i) {
		GEO::PeriodicDelaunay3d::IncidentTetrahedra W;
		delaunay_->copy_Laguerre_cell_from_Delaunay(i, C, W);
		// do something with C
		// C.clip_by_plane(GEO::vec4(1.0, 0.0, 0.0, 1.0));
		// C.clip_by_plane(GEO::vec4(-1.0, 0.0, 0.0, 1.0));
		// C.clip_by_plane(GEO::vec4(0.0, 1.0, 0.0, 1.0));
		// C.clip_by_plane(GEO::vec4(0.0, -1.0, 0.0, 1.0));
		// C.clip_by_plane(GEO::vec4(0.0, 0.0, 1.0, 1.0));
		// C.clip_by_plane(GEO::vec4(0.0, 0.0, -1.0, 1.0));
		C.compute_geometry();

		for (GEO::index_t v = 1; v < C.nb_v(); ++v) {
			GEO::index_t t = C.vertex_triangle(v);
             
			//  Happens if a clipping plane did not
			// clip anything.
			if (t == VBW::END_OF_LIST) {
				continue;
			}

			GEO::vec3 P[3];
			GEO::index_t n = 0;
			do {
				// Triangulate the Voronoi facet and send the
				// triangles to OpenGL/GLUP.
				if (n == 0) {
					P[0] = C.triangle_point(VBW::ushort(t));
				}
				else if (n == 1) {
					P[1] = C.triangle_point(VBW::ushort(t));
				}
				else {
					P[2] = C.triangle_point(VBW::ushort(t));
					tris[i].emplace_back(P[0].x, P[0].y, P[0].z);
					tris[i].emplace_back(P[1].x, P[1].y, P[1].z);
					tris[i].emplace_back(P[2].x, P[2].y, P[2].z);
					P[1] = P[2];
				}
				//   Move to the next triangle incident to
				// vertex v.
				GEO::index_t  lv = C.triangle_find_vertex(t, v);
				t = C.triangle_adjacent(t, (lv + 1) % 3);
				++n;
			} while (t != C.vertex_triangle(v));
		}
	}

	std::vector<Triangle_3> tri_out;
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < delaunay_->nb_vertices(); ++i)
	{
		if (index_map[i] == -1)
			continue;

		const std::vector<K::Point_3>& tri = tris[i];
		const int num_tri1 = tri.size() / 3;
		std::vector<bool> is_inserted(num_tri1, false);

		std::vector<Triangle_3> local_tris_out;
		GEO::vector<GEO::index_t> neighbors;
		delaunay_->get_neighbors(i, neighbors);

		// Debug
		if(false)
		{
			std::vector<Triangle_3> ts;
			for (int ii = 0; ii < num_tri1; ++ii)
				ts.emplace_back(tri[ii * 3], tri[ii * 3 + 1], tri[ii * 3 + 2]);
			write_ply(ts, "debug1.ply");
			for (int i_face = 0; i_face < num_tri1; ++i_face)
			{
				ts.clear();
				ts.emplace_back(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]);
				write_ply(ts, "debug2.ply");

				for (int i_neighbour = 34; i_neighbour < neighbors.size(); ++i_neighbour)
				{
					const std::vector<K::Point_3>& tris2 = tris[neighbors[i_neighbour]];
					const int num_tri2 = tris2.size() / 3;
					ts.clear();
					for (int ii = 0; ii < num_tri2; ++ii)
						ts.emplace_back(tris2[ii * 3], tris2[ii * 3 + 1], tris2[ii * 3 + 2]);
					write_ply(ts, "debug3.ply");

					if (neighbors[i_neighbour] == GEO::NO_FACET || index_map[i] == index_map[neighbors[i_neighbour]])
						continue;
					for (int i_face2 = 0; i_face2 < num_tri2; ++i_face2)
					{
						ts.clear();
						ts.emplace_back(tris2[i_face2 * 3], tris2[i_face2 * 3 + 1], tris2[i_face2 * 3 + 2]);
						write_ply(ts, "debug4.ply");
						std::cout << squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3]) << ", " <<
							squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3 + 1]) << ", " <<
							squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3 + 2]) << std::endl;
					}
				}

			}
		}
		
		for (int i_neighbour = 0; i_neighbour < neighbors.size(); ++i_neighbour)
		{
			if (neighbors[i_neighbour] == GEO::NO_FACET || index_map[i] == index_map[neighbors[i_neighbour]] || index_map[neighbors[i_neighbour]] == -1)
				continue;
			// Not very stable with equal distance to 3 primitives
			// if (!v_conn[index_map[i]][index_map[neighbors[i_neighbour]]])
				// continue;
			const std::vector<K::Point_3>& tris2 = tris[neighbors[i_neighbour]];
			const int num_tri2 = tris2.size() / 3;
			for (int i_face = 0; i_face < num_tri1; ++i_face)
			{
				if (is_inserted[i_face])
					continue;
				for (int i_face2 = 0; i_face2 < num_tri2; ++i_face2)
				{
					if (
						squared_distance(Triangle_3(tris2[i_face2 * 3], tris2[i_face2 * 3 + 1], tris2[i_face2 * 3 + 2]), tri[i_face * 3]) < 1e-12 &&
						squared_distance(Triangle_3(tris2[i_face2 * 3], tris2[i_face2 * 3 + 1], tris2[i_face2 * 3 + 2]), tri[i_face * 3 + 1]) < 1e-12 &&
						squared_distance(Triangle_3(tris2[i_face2 * 3], tris2[i_face2 * 3 + 1], tris2[i_face2 * 3 + 2]), tri[i_face * 3 + 2]) < 1e-12 &&
						squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3]) < 1e-12 &&
						squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3 + 1]) < 1e-12 &&
						squared_distance(Triangle_3(tri[i_face * 3], tri[i_face * 3 + 1], tri[i_face * 3 + 2]), tris2[i_face2 * 3 + 2]) < 1e-12
						)
					{
						local_tris_out.emplace_back(tri[i_face * 3], tri[i_face * 3+1], tri[i_face * 3 + 2]);
						// local_tris_out.emplace_back(i, i_face * 3);
						// local_tris_out.emplace_back(i, i_face * 3 + 1);
						// local_tris_out.emplace_back(i, i_face * 3 + 2);
						is_inserted[i_face] = true;
						break;
					}
				}
			}
		}
#pragma omp critical
		{
			tri_out.insert(tri_out.end(), local_tris_out.begin(), local_tris_out.end());
		}
	}

	LOG(INFO) << "Removing degenerated faces";
	std::vector<bool> is_degenerated(tri_out.size(), false);
	// #pragma omp parallel for
	// for (int i = tri_out.size() - 1; i >= 0; i--)
	// {
		// if (tri_out[i].squared_area() < 1e-16)
			// is_degenerated[i] = true;
	// }
	// #pragma omp parallel for
	// for (int i = tri_out.size() - 1; i >= 0; i--)
	// {
	// 	if (tri_out[i].bbox().max(0) > 1. || 
	// 		tri_out[i].bbox().max(1) > 1. ||
	// 		tri_out[i].bbox().max(2) > 1. ||
	// 		tri_out[i].bbox().max(0) < -1. ||
	// 		tri_out[i].bbox().max(1) < -1. ||
	// 		tri_out[i].bbox().max(2) < -1.
	// 		)
	// 		is_degenerated[i] = true;
	// }
	tri_out.erase(std::remove_if(tri_out.begin(), tri_out.end(), 
		[&is_degenerated, &tri_out](const Triangle_3& t)
		{
			return is_degenerated[&t-&tri_out[0]];
		}), tri_out.end());
	
	if (tri_out.size() < 10)
	{
		LOG(ERROR) << "No voronoi face is generated";
		throw std::runtime_error("No voronoi face is generated");
	}
	LOG(INFO) << "Start to write out";
	write_ply(tri_out, (v_root/"voronoi.ply").string());
	return;
}
