#include <iostream>
#include <unordered_set>

#include <glog/logging.h>

#include "geom_voronoi.h"

#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <Geom2d_Circle.hxx>
#include <Geom2d_Curve.hxx>
#include <Geom2d_Ellipse.hxx>
#include <Geom2d_Line.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Curve.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_Line.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Point.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <Geom_Surface.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <gp_Circ.hxx>
#include <gp_Circ2d.hxx>
#include <gp_Cone.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Elips2d.hxx>
#include <gp_Lin.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

#include <TopAbs_ShapeEnum.hxx>
#include <Poly_Triangulation.hxx>
#include <Mathematics/QuadricSurface.h>
#include <Mathematics/ApprQuadratic2.h>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <StlAPI_Writer.hxx>
#include <ShapeConstruct_ProjectCurveOnSurface.hxx>
#include <GeomLProp_SLProps.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GProp_GProps.hxx>
#include <BRepGProp.hxx>
#include <STEPControl_Reader.hxx>
#include <BRepClass_FaceClassifier.hxx>

void write_topp_shape(const TopoDS_Shape& v_shape, const fs::path& v_filename)
{
    std::vector<K::Point_3> vertices;
    std::vector<std::vector<int>> triangles;

    TopExp_Explorer exp;
    for (exp.Init(v_shape, TopAbs_FACE); exp.More(); exp.Next()) {
        TopoDS_Face face = TopoDS::Face(exp.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, loc);

        const int cur_vertex_size = vertices.size();
        for (int i = 0; i < triangulation->NbNodes(); ++i)
        {
            gp_Pnt p = triangulation->Node(i + 1);
            vertices.emplace_back(p.X(), p.Y(), p.Z());
        }

        for (int i = 0; i < triangulation->NbTriangles(); ++i)
        {
            Standard_Integer n1, n2, n3;
            triangulation->Triangle(i + 1).Get(n1, n2, n3);
			if (face.Orientation() == TopAbs_REVERSED)
				triangles.push_back({ n3 - 1 + cur_vertex_size, n2 - 1 + cur_vertex_size, n1 - 1 + cur_vertex_size });
			else
				triangles.push_back({ n1 - 1 + cur_vertex_size, n2 - 1 + cur_vertex_size, n3 - 1 + cur_vertex_size });
        }
    }
    CGAL::Polygon_mesh_processing::repair_polygon_soup(vertices, triangles);
    // CGAL::Polygon_mesh_processing::merge_duplicate_points_in_polygon_soup(vertices, triangles);
    // CGAL::Polygon_mesh_processing::merge_duplicate_polygons_in_polygon_soup(vertices, triangles);

    CGAL::IO::write_polygon_soup(v_filename.string(), vertices, triangles);
}

Point_set_3 sample_points_surface(const TopoDS_Shape& shape, std::vector<std::vector<bool>>& v_conn, const fs::path& output_root)
{
    LOG(INFO) << "== Begin sample only on surface";

    const double sample_radius = 0.005; // sampling density on meshes
    const int num_points_per_m = 1000; // sampling density on edges
    const double voronoi_distance = 0.0001;  // Set this to the desired distance by which to move the point to let the voronoi face across edges
    const double threshold_consider_same_point = 1e-4;
    const double jerk_epsilon = 1e-6; // Slightly move the point to avoid degenerated cases

    Point_set_3 sample_points;
    auto index_map = sample_points.add_property_map<int>("primitive_index", 0).first;

    int num_faces = 0;

    std::vector<TopoDS_Face> surfaces;
    std::unordered_map<TopoDS_Edge, std::vector<TopoDS_Face>> edge_face_map;
    std::unordered_map<TopoDS_Vertex, std::vector<TopoDS_Edge>> vertex_edge_map;
    // Save edge_face connectivity; throw if find degenerated edge
    for (TopExp_Explorer it_face(shape, TopAbs_FACE); it_face.More(); it_face.Next()) {
        TopoDS_Face face = TopoDS::Face(it_face.Current());
        surfaces.emplace_back(face);

        std::vector<TopoDS_Edge> local_edges;
        for (TopExp_Explorer it(face, TopAbs_EDGE); it.More(); it.Next()) {
            TopoDS_Edge edge = TopoDS::Edge(it.Current());

            if (BRep_Tool::Degenerated(edge))
            {
                LOG(ERROR) << "Found degenerated edges";
                throw;
            }
            local_edges.emplace_back(edge);
        }

        std::vector<bool> is_seam(local_edges.size(), false);
        for (int i = local_edges.size() - 1; i >= 0; --i)
        {
            const auto reveresed = std::find(local_edges.begin(), local_edges.end(), local_edges[i].Reversed());
            if (reveresed != local_edges.end())
            {
                is_seam[i] = true;
                is_seam[std::distance(local_edges.begin(), reveresed)] = true;
            }
        }
        for (int i = local_edges.size() - 1; i >= 0; --i)
        {
            if (is_seam[i])
				continue;
            TopoDS_Edge edge = local_edges[i];
            // Remove the curve if it is a generatrix
            if (edge_face_map.find(edge) != edge_face_map.end())
                edge_face_map[edge].push_back(face);
            else if (edge_face_map.find(TopoDS::Edge(edge.Reversed())) != edge_face_map.end())
                edge_face_map[TopoDS::Edge(edge.Reversed())].push_back(face);
            else
                edge_face_map.insert(std::make_pair(edge, std::vector<TopoDS_Face>{face}));
        }
        num_faces+=1;
    }
    v_conn.resize(num_faces, std::vector<bool>(num_faces, false));

    // Throw exception if one edge is shared by two more faces
    for (const auto& edge_face : edge_face_map)
        if (edge_face.second.size() != 2)
        {
            LOG(ERROR) << "Found edge shared by " << edge_face.second.size() << " faces";
            throw;
        }

    // Remove generatrix if needed; build vertex_edge_map
    LOG(INFO) << edge_face_map.size() << " curves before removing generatrix";
     {
        // Remove curve if the two adjacent faces are the same
        std::vector<TopoDS_Edge> curves_to_remove;
        for (const auto& edge_face : edge_face_map) {
            if (!edge_face.second[0].IsEqual(edge_face.second[1]))
                continue;
            curves_to_remove.emplace_back(edge_face.first);
        }
        for (const auto& item : curves_to_remove)
            edge_face_map.erase(item);

        // Build vertex-edge map
        for (const auto& curve_pair : edge_face_map)
        {
            const auto& curve = curve_pair.first;
            for (TopExp_Explorer it(curve, TopAbs_VERTEX); it.More(); it.Next()) {
                TopoDS_Vertex vertex = TopoDS::Vertex(it.Current());
                if (vertex_edge_map.find(vertex) != vertex_edge_map.end())
                    vertex_edge_map[vertex].push_back(curve);
                else if (vertex_edge_map.find(TopoDS::Vertex(vertex.Reversed())) != vertex_edge_map.end())
                    vertex_edge_map[TopoDS::Vertex(vertex.Reversed())].push_back(curve);
                else
                    vertex_edge_map.insert(std::make_pair(vertex, std::vector<TopoDS_Edge>{curve}));
            }
        }

        // Remove vertex if it is shared by only one curve
        Point_set_3 points;
        auto t_vertex_edge_map(vertex_edge_map);
        for (const auto& vertex_edge : vertex_edge_map) {
            if (vertex_edge.second.size() == 1 || vertex_edge.second.size() == 2 && vertex_edge.second[0].IsSame(vertex_edge.second[1]))
                t_vertex_edge_map.erase(vertex_edge.first);
        }
        vertex_edge_map = t_vertex_edge_map;
    }
    LOG(INFO) << "Number of vertices/edges/faces: " << surfaces.size() << "/" << edge_face_map.size() << "/" << vertex_edge_map.size();

    // Build the vertices and distance query function
    Point_set_3 vertices;
    std::vector<gp_Pnt> vertices_points;
    for (const auto& vertex_edge : vertex_edge_map)
    {
        vertices_points.push_back(BRep_Tool::Pnt(vertex_edge.first));
        vertices.insert(K::Point_3(vertices_points.back().X(), vertices_points.back().Y(), vertices_points.back().Z()));
    }
    auto is_point_exist = [&vertices_points, &threshold_consider_same_point](const gp_Pnt& p) {
        for (const auto& v : vertices_points)
            if (v.Distance(p) < threshold_consider_same_point)
                return true;
        return false;
    };
    CGAL::IO::write_point_set((output_root / "vertices.ply").string(), vertices);

    {
        int id_curve = 0;
        std::mt19937_64 mt(0);
        std::uniform_real_distribution<double> gen(jerk_epsilon/2, jerk_epsilon);
        for (const auto& curve_pair : edge_face_map) {
            TopoDS_Edge edge = curve_pair.first;
            TopoDS_Face face1 = curve_pair.second[0];
            TopoDS_Face face2 = curve_pair.second[1];

            Handle(Geom_Surface) surface1 = BRep_Tool::Surface(face1);
            Handle(Geom_Surface) surface2 = BRep_Tool::Surface(face2);

            const int id_surface1 = std::distance(surfaces.begin(), std::find(surfaces.begin(), surfaces.end(), face1));
            const int id_surface2 = std::distance(surfaces.begin(), std::find(surfaces.begin(), surfaces.end(), face2));
            v_conn[id_surface1][id_surface2] = true;
            v_conn[id_surface2][id_surface1] = true;
            GeomLProp_SLProps props1(surface1, 1, threshold_consider_same_point);
            GeomLProp_SLProps props2(surface2, 1, threshold_consider_same_point);

            double first, last;
            Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, first, last);
            GeomAdaptor_Curve adaptorCurve(curve, first, last);
            const double length = GCPnts_AbscissaPoint::Length(adaptorCurve);
            const int num_points = std::max(10., length * num_points_per_m);

            const double target_distance = 1e-3; // Start sampling from this distance and end before this distance
            double start_step = 1e-3, end_step = 1e-3;
            gp_Pnt sp, ep;
            gp_Pnt p;
            curve->D0(first, sp);
            curve->D0(last, ep);
            curve->D0(first+start_step, p);
			double distance = sp.Distance(p);
            int try_times = 1000000;
            while (std::abs(distance - target_distance) > 1e-4 && try_times > 0)
            {
                if (distance < target_distance)
                    start_step += 1e-6;
                else
                    start_step -= 1e-6;
                try_times--;
				curve->D0(first + start_step, p);
				distance = sp.Distance(p);
            }
            if (try_times == 0)
			{
                LOG(ERROR) << start_step;
                LOG(ERROR) << distance;
                LOG(ERROR) << length;
                LOG(ERROR) << sp.X() << ", " << sp.Y() << ", " << sp.Z();
                LOG(ERROR) << ep.X() << ", " << ep.Y() << ", " << ep.Z();
				LOG(ERROR) << "Cannot find the start step";
				throw;
			}

            try_times = 1000000;
            curve->D0(last - end_step, p);
            distance = ep.Distance(p);
            while (std::abs(distance - target_distance) > 1e-4 && try_times > 0)
            {
                if (distance < target_distance)
                    end_step += 1e-6;
                else
                    end_step -= 1e-6;
                try_times--;
                curve->D0(last - end_step, p);
                distance = ep.Distance(p);
            }
            if (try_times == 0)
            {
                LOG(ERROR) << "Cannot find the end step";
                throw;
            }

            std::vector<double> local_us;
            for (int i_p = 0; i_p < num_points; ++i_p) 
            {
                double u = (last - first) * i_p / (num_points - 1) + first + gen(mt);
                if (i_p == 0)
                    u = first + start_step;
                else if (i_p == num_points - 1)
					u = last - end_step;
            	local_us.push_back(u);
            }

            std::vector<gp_Pnt> local_curve_points;
            for (int i_p = 0; i_p < local_us.size(); ++i_p) {
                const double u = local_us[i_p];
                gp_Pnt p;
                curve->D0(u, p);
                const double voronoi_distance_local = voronoi_distance + gen(mt);
                // if (is_point_exist(p))
                    // continue;
                // index_map[*sample_points.insert(K::Point_3(p.X(), p.Y(), p.Z()))] = id_curve + surfaces.size();
                // local_curve_points.emplace_back(p);

                GeomAPI_ProjectPointOnSurf projector1(p, surface1, threshold_consider_same_point);
                GeomAPI_ProjectPointOnSurf projector2(p, surface2, threshold_consider_same_point);

                double u0, v0;
                projector1.LowerDistanceParameters(u0, v0);

                gp_Vec tangent;
                curve->D1(u, p, tangent);

                props1.SetParameters(u0, v0);

                if (props1.IsNormalDefined()) {
                    gp_Vec normal = props1.Normal();
                    gp_Vec orthogonalDirection = normal.Crossed(tangent);
                    if (face1.Orientation() == TopAbs_REVERSED)
                        orthogonalDirection.Reverse();
                    if (edge.Orientation() == TopAbs_REVERSED)
                        orthogonalDirection.Reverse();
                    orthogonalDirection.Normalize();
                    gp_Pnt newPoint = p.Translated(orthogonalDirection * (voronoi_distance + gen(mt)));

                    // Check if the point is inside the boundary
                    BRepClass_FaceClassifier checkPoint(face1, newPoint, 1e-4);
                    if (checkPoint.State() == TopAbs_IN)
						index_map[*sample_points.insert(K::Point_3(newPoint.X(), newPoint.Y(), newPoint.Z()))] = id_surface1;
                }
                else
                {
                    LOG(ERROR) << "Normal not defined";
                    throw;
                }

                projector2.LowerDistanceParameters(u0, v0);
                props2.SetParameters(u0, v0);
                if (props2.IsNormalDefined())
                {
                    gp_Vec normal = -props2.Normal();
                    gp_Vec orthogonalDirection = normal.Crossed(tangent);
                    if (face2.Orientation() == TopAbs_REVERSED)
                        orthogonalDirection.Reverse();
                    if (edge.Orientation() == TopAbs_REVERSED)
                        orthogonalDirection.Reverse();
                    orthogonalDirection.Normalize();
                    gp_Pnt newPoint = p.Translated(orthogonalDirection * (voronoi_distance + gen(mt)));
                    BRepClass_FaceClassifier checkPoint(face2, newPoint, 1e-4);
                    if (checkPoint.State() == TopAbs_IN)
						index_map[*sample_points.insert(K::Point_3(newPoint.X(), newPoint.Y(), newPoint.Z()))] = id_surface2;
                }
                else
                {
                    LOG(ERROR) << "Normal not defined";
                    throw;
                }
            }
            id_curve++;
            // CGAL::IO::write_point_set((output_root / "sampled_points.ply").string(), sample_points);
            // save_face_as_stl(face, "face.stl");
        }
    }
    CGAL::IO::write_point_set((output_root / "sampled_points.ply").string(), sample_points);
    LOG(INFO) << sample_points.size() << " points sampled on the curves and surfaces";

    // Remove duplicate points; Remove points near the vertices
    {
        std::vector<Point_and_int> points(sample_points.size());
        for (int i = 0; i < sample_points.size(); ++i)
            points[i] = Point_and_int(sample_points.point(i), i);
        const auto point_flags = deduplicate_points(points, threshold_consider_same_point * threshold_consider_same_point);

        for (int i = sample_points.size() - 1; i >= 0; --i)
        {
            if (point_flags[i] != -1)
                sample_points.remove(i);
            else if (is_point_exist(gp_Pnt(sample_points.point(i).x(), sample_points.point(i).y(), sample_points.point(i).z())))
                sample_points.remove(i);
        }
        sample_points.collect_garbage();
    }
    LOG(INFO) << sample_points.size() << " points after removing duplication";
    CGAL::IO::write_point_set((output_root / "sampled_points.ply").string(), sample_points);

    // Surface
    for (int ii = 0; ii < surfaces.size(); ++ii)
    {
        std::unordered_set<TopoDS_Edge> local_curves;
        for (TopExp_Explorer it(surfaces[ii], TopAbs_EDGE); it.More(); it.Next()) {
            TopoDS_Edge edge = TopoDS::Edge(it.Current());
            if (edge_face_map.find(edge) != edge_face_map.end() ||
                edge_face_map.find(TopoDS::Edge(edge.Reversed())) != edge_face_map.end())
                local_curves.insert(edge);
        }

        std::vector<Triangle_3> local_tris;
        TopLoc_Location loc;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(surfaces[ii], loc);
        if (!triangulation.IsNull()) {
            for (int i = 1; i <= triangulation->NbTriangles(); i++) {
                const Poly_Triangle& t = triangulation->Triangle(i);
                Standard_Integer n1, n2, n3;
                t.Get(n1, n2, n3);

                local_tris.emplace_back(
                    K::Point_3(triangulation->Node(n1).X(), triangulation->Node(n1).Y(), triangulation->Node(n1).Z()),
                    K::Point_3(triangulation->Node(n2).X(), triangulation->Node(n2).Y(), triangulation->Node(n2).Z()),
                    K::Point_3(triangulation->Node(n3).X(), triangulation->Node(n3).Y(), triangulation->Node(n3).Z()));
            }
        }
        const double area = std::accumulate(local_tris.begin(), local_tris.end(),
            0.0, [](double acc, const Triangle_3& t) {
                return acc + std::sqrt(t.squared_area());
            });

        Point_set_3 local_points = sample_poisson_points(local_tris, sample_radius, area / sample_radius / sample_radius * 10);
        Point_set_3 local_sample_points;
        auto local_index_map = local_sample_points.add_property_map<int>("primitive_index", 0).first;
        for (int i_p = 0; i_p < local_points.number_of_points(); ++i_p) {
            bool safe = true;
            for (const auto& edge : local_curves) {
                gp_Pnt p(local_points.point(i_p).x(), local_points.point(i_p).y(), local_points.point(i_p).z());
                double first, last;
                Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, first, last);
                GeomAPI_ProjectPointOnCurve projector(p, curve, first, last);
                if (projector.NbPoints() == 0)
                    continue;
                if (projector.LowerDistance() < 5e-3) {
                    safe = false;
                    break;
                }
            }
            if (safe)
                index_map[*sample_points.insert(local_points.point(i_p))] = ii;
        }
    }

    LOG(INFO) << "Add bounding points";
    {
        std::mt19937_64 mt(0);
        std::uniform_real_distribution<double> gen(jerk_epsilon / 2, jerk_epsilon);
        const double s = 2.;
        // Top
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(0.1 * i - s + gen(mt), 0.1 * j - s + gen(mt), s + gen(mt)))] = -1;
        // Bottom
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(0.1 * i - s + gen(mt), 0.1 * j - s + gen(mt), -s + gen(mt)))] = -1;
        // Left
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(-s + gen(mt), 0.1 * i - s + gen(mt), 0.1 * j - s + gen(mt)))] = -1;
        // Right
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(s + gen(mt), 0.1 * i - s + gen(mt), 0.1 * j - s + gen(mt)))] = -1;
        // Front
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(0.1 * i - s + gen(mt), -s + gen(mt), 0.1 * j - s + gen(mt)))] = -1;
        // Back
        for (int i = 1; i < 40; ++i)
            for (int j = 1; j < 40; ++j)
                index_map[*sample_points.insert(K::Point_3(0.1 * i - s + gen(mt), s + gen(mt), 0.1 * j - s + gen(mt)))] = -1;

    }

    CGAL::IO::write_point_set((output_root / "sampled_points.ply").string(), sample_points);
    LOG(INFO) << "== Done sampling with number of points: " << sample_points.size();
    return sample_points;
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging("calculate_voronoi");
    google::SetStderrLogging(google::GLOG_INFO);

	if (argc != 3)
	{
		LOG(ERROR) << "Usage: " << argv[0] << " <input_step_file> <output_dir>";
		return -1;
	}

    const double boundings = 0.9;
    const std::string file_name(argv[1]);
    const fs::path output_root(argv[2]);
    // const int is_generatrix(std::atoi(argv[3]));
    const int is_generatrix = 0;
    if (!fs::exists(output_root))
        fs::create_directories(output_root);
    LOG(INFO) << "Preserve generatrix: " << is_generatrix;
    TopoDS_Shape shape;
    {
        TopoDS_Shape original_shape;
			// Read step file to TopoDS_Shape
			STEPControl_Reader reader;
			reader.ReadFile(file_name.c_str());
			reader.TransferRoots();
            original_shape = reader.OneShape();
        Bnd_Box boundingBox;
        BRepBndLib::Add(original_shape, boundingBox);

        Standard_Real xmin, ymin, zmin, xmax, ymax, zmax;
        boundingBox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
        Standard_Real scaleX = boundings * 2 / (xmax - xmin);
        Standard_Real scaleY = boundings * 2 / (ymax - ymin);
        Standard_Real scaleZ = boundings * 2 / (zmax - zmin);

        // Use the smallest scale factor to keep the aspect ratio intact
        Standard_Real scaleFactor = std::min({ scaleX, scaleY, scaleZ });

        gp_Vec translation1(-(xmin + xmax) / 2, -(ymin + ymax) / 2, -(zmin + zmax) / 2);

        gp_Trsf trsf1;
        trsf1.SetTranslationPart(translation1); // Translate to center
        gp_Trsf trsf2;
        trsf2.SetScaleFactor(scaleFactor); // Scale around the origin
        trsf2.Multiply(trsf1);
        BRepBuilderAPI_Transform transformer(trsf2);
        transformer.Perform(original_shape);
        shape = transformer.Shape();

        std::ofstream ofs((output_root / "normalized_params.txt").string());
        ofs << -(xmin + xmax) / 2 << " " << -(ymin + ymax) / 2 << " " << - (zmin + zmax) / 2 << " " << scaleFactor;
        ofs.close();
    }
    const double mesh_threshold = 0.001;
    BRepMesh_IncrementalMesh mesh(shape, mesh_threshold);
    // StlAPI_Writer writer;
	// writer.Write(shape, (output_root / "original_mesh.stl").string().c_str());
    write_topp_shape(shape, output_root / "normalized_mesh.ply");

    // Sample points
    std::vector<std::vector<bool>> v_conn;
    const auto& sample_points = sample_points_surface(shape, v_conn, output_root);

    // Calculate Voronoi
    LOG(INFO) << "Start to calculate Voronoi";
    compute_voronoi(sample_points, v_conn, output_root);
    LOG(INFO) << "Done";
    return 0;
}
