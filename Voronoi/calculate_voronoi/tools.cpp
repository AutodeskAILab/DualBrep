#include "tools.h"

#include <vcg/space/point3.h>
#include <vcg/complex/algorithms/point_sampling.h>

Point_set_3 sample_points(const std::vector<Triangle_3>& v_mesh, const int v_num_points, bool at_least_one_point)
{
	std::mt19937 gen; std::uniform_real_distribution<double> dist(0.0f, 1.0f);
	Point_set_3 o_point_set(true);
	auto index_map = o_point_set.add_property_map<int>("face_index", 0).first;

	double total_area = 0.;
	// #pragma omp parallel for reduction(+:total_area)
	for (int i_face = 0; i_face < v_mesh.size(); ++i_face)
		total_area += std::sqrt(v_mesh[i_face].squared_area());

	double point_per_area = (double)v_num_points / total_area;

	// #pragma omp parallel for
	for (int i_face = 0; i_face < v_mesh.size(); ++i_face)
	{
		K::Point_3 vertexes[3];
		vertexes[0] = v_mesh[i_face].vertex(0);
		vertexes[1] = v_mesh[i_face].vertex(1);
		vertexes[2] = v_mesh[i_face].vertex(2);

		K::Vector_3 normal = CGAL::cross_product(vertexes[1] - vertexes[0], vertexes[2] - vertexes[0]);
		normal /= std::sqrt(normal.squared_length());

		double area = std::sqrt(v_mesh[i_face].squared_area());

		double face_samples = area * point_per_area;
		unsigned int num_face_samples = face_samples;

		if (at_least_one_point)
			num_face_samples = std::max((unsigned int)1, num_face_samples);
		else
		{
			if (dist(gen) < face_samples - static_cast<double>(num_face_samples))
			{
				num_face_samples += 1;
			}
		}


		for (unsigned int j = 0; j < num_face_samples; ++j) {
			double r1 = dist(gen);
			double r2 = dist(gen);

			double tmp = std::sqrt(r1);
			double u = 1.0f - tmp;
			double v = r2 * tmp;

			double w = 1.0f - v - u;
			auto point = K::Point_3(
				u * vertexes[0].x() + v * vertexes[1].x() + w * vertexes[2].x(),
				u * vertexes[0].y() + v * vertexes[1].y() + w * vertexes[2].y(),
				u * vertexes[0].z() + v * vertexes[1].z() + w * vertexes[2].z()
			);
			// #pragma omp critical
			{
				index_map[*o_point_set.insert(point, normal)] = i_face;
			}
		}
	}
	return o_point_set;
}

Point_set_3 sample_poisson_points(const std::vector<Triangle_3>& v_mesh, const double radius, const int init_points)
{
	using namespace vcg;
	class MyEdge;
	class MyFace;
	class MyVertex;
	struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
		                                      Use<MyEdge>     ::AsEdgeType,
		                                      Use<MyFace>     ::AsFaceType> {};
	class MyVertex : public Vertex<MyUsedTypes, vertex::Coord3d, vertex::Normal3d, vertex::BitFlags  > {};
	class MyFace : public Face< MyUsedTypes, face::FFAdj, face::Normal3d, face::VertexRef, face::BitFlags > {};
	class MyEdge : public Edge<MyUsedTypes> {};
	class MyMesh : public tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace>, std::vector<MyEdge>  > {};

	Point_set_3 montecarlo_points = sample_points(v_mesh, std::max(10000, init_points), true);
	auto montecarlo_index_map = montecarlo_points.property_map<int>("face_index").first;
	MyMesh MontecarloMesh;
	tri::Allocator<MyMesh>::AddVertices(MontecarloMesh, montecarlo_points.size());
	// #pragma omp parallel for
	for (int i_point = 0; i_point < montecarlo_points.size(); ++i_point)
	{
		const auto& p = montecarlo_points.point(i_point);
		const auto& n = montecarlo_points.normal(i_point);
		MontecarloMesh.vert[i_point].P() = Point3d(p.x(), p.y(), p.z());
		MontecarloMesh.vert[i_point].N() = Point3d(n.x(), n.y(), n.z()).normalized();
	}

	tri::UpdateBounding<MyMesh>::Box(MontecarloMesh);
	std::vector<MyMesh::VertexPointer> poissonSamplesVP;
	tri::PoissonPruning<MyMesh>(
		MontecarloMesh, poissonSamplesVP, radius);


	Point_set_3 p(true);
	auto index_map = p.add_property_map<int>("face_index", 0).first;
	p.resize(poissonSamplesVP.size());

	// #pragma omp parallel for
	for (int i_point = 0; i_point < (int)poissonSamplesVP.size(); ++i_point)
	{
		int index = poissonSamplesVP[i_point] - &MontecarloMesh.vert[0];
		index_map[i_point] = montecarlo_index_map[index];

		p.point(i_point) = K::Point_3(
			poissonSamplesVP[i_point]->P()[0],
			poissonSamplesVP[i_point]->P()[1],
			poissonSamplesVP[i_point]->P()[2]
		);
		p.normal(i_point) = K::Vector_3(
			poissonSamplesVP[i_point]->N()[0],
			poissonSamplesVP[i_point]->N()[1],
			poissonSamplesVP[i_point]->N()[2]);
	}
	return p;
}

std::vector<int> deduplicate_points(const std::vector<Point_and_int>& v_points, const double v_sq_threshold)
{
	KDTree tree(v_points.begin(), v_points.end());
	tree.build();

	std::vector<int> is_duplicate(v_points.size(), -1);
#pragma omp parallel for
	for (int i = 0; i < v_points.size(); ++i)
	{
		K_neighbor_search search(tree, boost::get<0>(v_points[i]), 2);
		Distance tr_dist;
		for (int j = 0; j < 2; ++j)
		{
			const auto& target = search.begin() + j;
			const int target_id = boost::get<1>(target->first);
			if (i <= target_id)
				continue;
			if (target->second < v_sq_threshold)
				is_duplicate[i] = target_id;
		}
		
	}
	return is_duplicate;
}
