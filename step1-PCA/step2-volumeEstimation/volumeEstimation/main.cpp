#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
// !OpenCV

// Eigen3
#include "Eigen/Eigen"
// !Eigen3

using namespace std;
using namespace cv;
using namespace Eigen;

//
// Class Histogram
//
class Histogram3D
{
	public:
		Histogram3D(int binx, double minx, double maxx,
					int biny, double miny, double maxy,
					int binz, double minz, double maxz);
		~Histogram3D();

	public:
		void	Fill(double x, double y, double z, double weight);
		void	GetBinCentor(int IDx, int IDy, int IDz, double &x, double &y, double &z);
		double	GetBinCounts(int IDx, int IDy, int IDz);
		void	SetBinCounts(int IDx, int IDy, int IDz, double counts);
		int		GetID(int IDx, int IDy, int IDz);
		void	Normalize();
		void	Show();

	
	private:
		int binx_;
		int biny_;
		int binz_;

		double minx_, maxx_;
		double miny_, maxy_;
		double minz_, maxz_;

		double binWidthx_;
		double binWidthy_;
		double binWidthz_;

		vector<double> Hist_; // the 3D histogram

		int CountsLower_;
		int CountsUpper_;

};

Histogram3D::Histogram3D(int binx, double minx, double maxx,
					int biny, double miny, double maxy,
					int binz, double minz, double maxz)
{
	binx_ =  binx;
    biny_ =  biny;
    binz_ =  binz;

	minx_ =  minx;
    miny_ =  miny;
    minz_ =  minz;

	maxx_ =  maxx;
    maxy_ =  maxy;
    maxz_ =  maxz;

	binWidthx_ = (maxx_-minx_)/double(binx_);
    binWidthy_ = (maxy_-miny_)/double(biny_);
    binWidthz_ = (maxz_-minz_)/double(binz_);

	CountsLower_ = 0;
	CountsUpper_ = 0;

	// debug
	cout<<"Histogram: "<<endl;
	cout<<"X - bin, min, max, binWidth: "<<binx_<<", "<<minx_<<", "<<maxx_<<", "<<binWidthx_<<endl;
	cout<<"Y - bin, min, max, binWidth: "<<biny_<<", "<<miny_<<", "<<maxy_<<", "<<binWidthy_<<endl;
	cout<<"Z - bin, min, max, binWidth: "<<binz_<<", "<<minz_<<", "<<maxz_<<", "<<binWidthz_<<endl;

	// initialize the 3D histogram
	int size = binx_*biny_*binz_;
	for(int i=0;i<size;i++)
	{
		Hist_.push_back(0);
	}

}

Histogram3D::~Histogram3D()
{}

void Histogram3D::Fill(double x, double y, double z, double weight)
{
	//
	// ID = IDx*biny_*binz_ + IDy*binz_ + IDz

	if(x<=minx_||y<=miny_||z<=minz_) 
	{
		CountsLower_ ++;
		return;
	}

	if(x>=maxx_||y>=maxy_||z>=maxz_) 
	{
		CountsUpper_ ++;
		return;
	}

	int IDx = ceil((x-minx_)/binWidthx_);
	int IDy = ceil((y-miny_)/binWidthy_);
	int IDz = ceil((z-minz_)/binWidthz_);

	int ID = GetID(IDx, IDy, IDz);

	Hist_[ID] += weight;
}

void Histogram3D::GetBinCentor(int IDx, int IDy, int IDz, double &x, double &y, double &z)
{
	x = minx_ + double(IDx+0.5)*binWidthx_;
	y = miny_ + double(IDy+0.5)*binWidthy_;
	z = minz_ + double(IDz+0.5)*binWidthz_;
}

double	Histogram3D::GetBinCounts(int IDx, int IDy, int IDz)
{
	int ID = GetID(IDx, IDy, IDz);
	double counts = Hist_[ID];

	return counts;
}

void Histogram3D::SetBinCounts(int IDx, int IDy, int IDz, double counts)
{
	int ID = GetID(IDx, IDy, IDz);
	Hist_[ID] = counts;

	return;
}

int Histogram3D::GetID(int IDx, int IDy, int IDz)
{
	int ID = IDx*biny_*binz_ + IDy*binz_ + IDz;
	return ID;
}

void Histogram3D::Normalize()
{
	double TotalCounts = 0;

	for(int i=0;i<binx_;i++)
	for(int j=0;j<biny_;j++)
	for(int k=0;k<binz_;k++)
	{
		double counts = GetBinCounts(i, j, k);
		TotalCounts += counts;
	}

	if(TotalCounts==0)
		return;


	for(int i=0;i<binx_;i++)
	for(int j=0;j<biny_;j++)
	for(int k=0;k<binz_;k++)
	{
		double counts = GetBinCounts(i, j, k);
		double normalizedCounts = counts/TotalCounts;
		SetBinCounts(i, j, k, normalizedCounts);
	}

	return;
}

void Histogram3D::Show()
{
	for(int i=0;i<binx_;i++)
	for(int j=0;j<biny_;j++)
	for(int k=0;k<binz_;k++)
	{
		double x = 0;
		double y = 0;
		double z = 0;
		GetBinCentor(i, j, k, x, y, z);

		double counts = GetBinCounts(i, j, k);

		cout<<"ID(x,y,z): "<<i<<", "<<j<<", "<<k
			<<"; BinCentor(x,y,z): "<<x<<", "<<y<<", "<<z
			<<"; Counts: "<<counts<<endl;
	}
}

//
// !Class Histogram
//

int ReadPointCloud(string filename, vector<Point3f> &points)
{
	points.clear();

	//
	// import 3D point cloud
	//
	ifstream file(filename);

	if(file.fail())
	{
		cout<<"Can not find the file \" "<<filename<<" \""<<endl;
		return 0;
	}

	double x,y,z;
	double nx,ny,nz;
	double flag;
	while(!file.eof())
	{
		file>>x>>y>>z>>nx>>ny>>nz>>flag;

		if(file.eof()) break;

		if(flag==0) continue;

		if(x==0&&y==0&&z==0) continue;

		//if(z>1500) continue;

		Point3f	p(x,y,z);
		points.push_back(p);
	}

	return 1;
}

// Normal Estiamtion
void GetNormal(int size, Point3f points[], Point3f &normal)
{
	//
	// method 24
	//

	// step 0 : if no point inputed, set the normal (0,0,0)
	if(size==0)
	{
		normal.x = 0;
		normal.y = 0;
		normal.z = 0;
		return;
	}

	// step 1 : get expectation of coordinates of the points
	float x_mean = 0;
	float y_mean = 0;
	float z_mean = 0;

	for(int i=0;i<size;i++)
	{
		x_mean += points[i].x;
		y_mean += points[i].y;
		z_mean += points[i].z;
	}
	x_mean /= float(size);
    y_mean /= float(size);
    z_mean /= float(size);

	//cout<<"size: "<<size<<endl;
	//cout<<"mean: "<<x_mean<<", "<<y_mean<<", "<<z_mean<<", "<<endl;

	//
	// step 2 : Covariance matrix
	//
	Mat Cov = Mat::zeros(3,3,CV_32F); //Covariance matrix
	//cout<<"Cov\n"<<Cov<<endl;

	for(int i=0;i<size;i++)
	{
		points[i].x -= x_mean;
    	points[i].y -= y_mean;
		points[i].z -= z_mean;
		//cout<<"points[i]: "<<points[i]<<endl;

		Mat m = Mat::zeros(3,1,CV_32F);
		m.ptr<float>(0)[0] = points[i].x;
		m.ptr<float>(1)[0] = points[i].y;
		m.ptr<float>(2)[0] = points[i].z;
		//cout<<"m: "<<m<<endl;

		Cov = Cov + m*m.t();
		//cout<<"Cov: "<<Cov<<endl;
	}
	//cout<<"Cov: "<<Cov<<endl;

	Cov /= float(size);
	//cout<<"Cov: "<<Cov<<endl;


	//
	// PCA
	//
	// A (Cov)=u*w*vt
	// A (Cov): the input and decomposed matrix - Cov: \n"<<Cov<<endl<<endl;
	// u : calculated singular values: \n"<<w<<endl<<endl;
	// w : calculated left singular vectors: \n"<<u<<endl<<endl;
	// vt: transposed matrix of right singular values: \n"<<vt<<endl<<endl;
	Mat w,u,vt;
	SVD::compute(Cov, w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	//SVD::compute(Cov, w,u,vt, cv::SVD::FULL_UV);

	// get the minimum eigen value
	int Idx_minEigVal = 0;
	float minEigVal = w.ptr<float>(0)[0];
	for(int i=0;i<2;i++)
	{
		int idx = i+1;
		float curEigVal = w.ptr<float>(idx)[0];

		if(curEigVal<minEigVal) 
		{
			minEigVal=curEigVal;
			Idx_minEigVal = idx;
		}
	}

	//cout<<"Idx_minEigVal: "<<Idx_minEigVal<<endl;
	//cout<<"minEigVal: "<<minEigVal<<endl;

	// normal
	normal.x = vt.ptr<float>(Idx_minEigVal)[0];
	normal.y = vt.ptr<float>(Idx_minEigVal)[1];
	normal.z = vt.ptr<float>(Idx_minEigVal)[2];

	//
	// direction of the normal vector
	//
	Point3f normalCamera(0,0,-1);
	double cosAngle = normal.dot(normalCamera) / (norm(normal)*norm(normalCamera));
	if(cosAngle>1)			cosAngle = 1;
	else if(cosAngle<-1)	cosAngle = -1;
	double angle = acos(cosAngle);

	if(angle>0.5*M_PI)
	{
		normal *= -1;
	}

	return;
}
// !Normal Estiamtion

// 
// find a plane that corresponds to the most of the points 
// 
// Method 0 : the voting metheod that uas no weight is used 
void FindPlane_SdandardHough(vector<Point3f>	points, 
				  vector<double>	List_theta, 
				  vector<double> 	List_phi, 
				  vector<double> 	List_r, 
				  vector<double> 	List_Counts,
				  vector<int>		&planeID
				  )
{
	// Find the plane that correspond to the most of the points 
	// the voting metheod with no weight is used 

	//
	// identify points according to the distance to a plane 
	//

	// initialize
	planeID.clear();
	for(int i=0;i<points.size();i++)
	{
		planeID.push_back(-1);
	}


	double threshold_planeDist = 1000; // mm
	double threshold_squared_planeDist = pow(threshold_planeDist,2); // mm

	// determine all points
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;

		// all planes that have been found are considered
		vector<double>	List_distance2;
		vector<int>		List_bestPlaneID;
		for(int j=0;j<List_theta.size();j++)
		{
			double theta = List_theta[j]; // theta
			double phi   = List_phi  [j]; // phi
			double r     = List_r    [j]; // r

			double r_est =	x*sin(theta)*cos(phi) 
					      + y*sin(theta)*sin(phi)
					      + z*cos(theta);

			double weight = 1.;

			double distance2 = weight*pow(r_est-r,2);

			//cout<<"r_est: "<<r_est<<", r: "<<r<<", distance: "<<sqrt(distance2)<<endl;

			// if the squared distance is smaller than the threshold,
			// we would consider that this plane is a candidate
			if(distance2<threshold_squared_planeDist)
			{
				List_distance2.push_back(distance2);
				List_bestPlaneID.push_back(j);
			}
		}

		// search the plane that the current point belongs to
		if(List_distance2.size()==0) continue;

		int bestPlaneID = 0;
		double bestDistance2 = List_distance2[0];
		for(int j=0;j<List_distance2.size();j++)
		{
			if(bestDistance2>List_distance2[j])
			{
				bestPlaneID = j;
				bestDistance2>List_distance2[j];
			}
		}
		planeID[i] = bestPlaneID;
	} 
}
// Method 1 : the weighted voting metheod is used 
void FindPlane_WV(vector<Point3f>	points, 
				  vector<double>	List_theta, 
				  vector<double> 	List_phi, 
				  vector<double> 	List_r, 
				  vector<double> 	List_Counts,
				  vector<int>		&planeID
				  )
{
	// Find the plane that correspond to the most of the points 
	// the weighted voting metheod is used 

	//
	// identify points according to the distance to a plane 
	//

	// initialize
	planeID.clear();
	for(int i=0;i<points.size();i++)
	{
		planeID.push_back(-1);
	}


	double threshold_planeDist = 1000; // mm
	double threshold_squared_planeDist = pow(threshold_planeDist,2); // mm

	// determine all points
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;

		// all planes that have been found are considered
		vector<double>	List_distance2;
		vector<int>		List_bestPlaneID;
		for(int j=0;j<List_theta.size();j++)
		{
			double theta = List_theta[j]; // theta
			double phi   = List_phi  [j]; // phi
			double r     = List_r    [j]; // r

			double r_est =	x*sin(theta)*cos(phi) 
					      + y*sin(theta)*sin(phi)
					      + z*cos(theta);

			double weight = 1./pow(List_Counts[j],3);

			double distance2 = weight*pow(r_est-r,2);

			//cout<<"r_est: "<<r_est<<", r: "<<r<<", distance: "<<sqrt(distance2)<<endl;

			// if the squared distance is smaller than the threshold,
			// we would consider that this plane is a candidate
			if(distance2<threshold_squared_planeDist)
			{
				List_distance2.push_back(distance2);
				List_bestPlaneID.push_back(j);
			}
		}

		// search the plane that the current point belongs to
		if(List_distance2.size()==0) continue;

		int bestPlaneID = 0;
		double bestDistance2 = List_distance2[0];
		for(int j=0;j<List_distance2.size();j++)
		{
			if(bestDistance2>List_distance2[j])
			{
				bestPlaneID = j;
				bestDistance2>List_distance2[j];
			}
		}
		planeID[i] = bestPlaneID;
	} 
}

// Method 2 : find the plane corresponding to the largest votes, as is named Statistic Hough Transformation 
void FindPlane_StaHT(vector<Point3f>	points, 
					vector<double>	List_theta, 
				  	vector<double> 	List_phi, 
				  	vector<double> 	List_r, 
				  	vector<double> 	List_Counts,
				  	vector<int>		&planeID
				   )
{
	// Method 2 : find the plane corresponding to the largest votes, as is named Statistic Hough Transformation 
	
	// initialize
	planeID.clear();
	for(int i=0;i<points.size();i++)
	{
		planeID.push_back(-1);
	}

	// find the plane ID corresponding to the largest counts, which is the number of vote
	int bestPlaneID = 0;
	double LargestCounts = List_Counts[0];
	for(int i=0;i<List_Counts.size();i++)
	{
		double counts = List_Counts[i]; // counts

		if(counts>LargestCounts)
		{
			LargestCounts = counts;
			bestPlaneID = i;
		}
	}

	double threshold_planeDist = 100; // mm
	double threshold_squared_planeDist = pow(threshold_planeDist,2); // mm

	vector<Point3f> planePoints_vec;

	// set best plane ID for all points
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;

		double theta = List_theta[bestPlaneID]; // theta
		double phi   = List_phi  [bestPlaneID]; // phi
		double r     = List_r    [bestPlaneID]; // r

		double r_est =	x*sin(theta)*cos(phi) 
				      + y*sin(theta)*sin(phi)
				      + z*cos(theta);

		double distance2 = pow(r_est-r,2);

		if(distance2<threshold_squared_planeDist)
		{
			planePoints_vec.push_back(Point3f(x,y,z));
		}
	}

	// Plane estimation
	Point3f planePoints[planePoints_vec.size()];
	Point3f planeNormal(0,0,0);
	for(int i=0;i<planePoints_vec.size();i++)
	{
		planePoints[i] = planePoints_vec[i];
		//cout<<"i"<<i<<", planePoints[i]: "<<planePoints[i]<<endl;
	}
	GetNormal(planePoints_vec.size(), planePoints, planeNormal);
	cout<<"planeNormal: "<<planeNormal<<endl;

	// get r, which is the distance in the plane function
	double plane_r = 0;
	for(int i=0;i<planePoints_vec.size();i++)
	{
		double x = planePoints_vec[i].x;
		double y = planePoints_vec[i].y;
		double z = planePoints_vec[i].z;

		//cout<<"planePoints_vec: "<<planePoints_vec[i]<<endl;

		// plane function : Ax + By + Cz - plane_r = 0
		double A = planeNormal.x;  // plane
		double B = planeNormal.y;  
		double C = planeNormal.z;  
		plane_r = plane_r + x*A + y*B + z*C;
	}
	plane_r /= double(planePoints_vec.size()); 
	cout<<"plane_r: "<<plane_r<<endl;

	// find points that belong to the plane 
	double threshold_planeDist_part2 = 100; // mm
	double threshold_squared_planeDist_part2 = pow(threshold_planeDist_part2, 2); 
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;

		// plane function : Ax + By + Cz - plane_r = 0
		double A = planeNormal.x;  // plane
		double B = planeNormal.y;  
		double C = planeNormal.z;  

		double distance = A*x + B*y + C*z - plane_r;
		double distance2 = pow(distance,2);

		if(distance2<threshold_squared_planeDist_part2)
		{
			planeID[i] = bestPlaneID;
		}
	}




	return;
}

// 
// ! find a plane that corresponds to the most of the points 
// 

// Pose estimation, PCA is used.
void PoseEstimation(vector<Point3f> points, vector<Point3f> &CCS, Point3f &centor)
{
	// vector<Point3f> CCS is the Canonical Coordinate System, which contains three unit vectors.
	// initiate the CCS
	CCS.clear();
	CCS.push_back(Point3f(1,0,0));
	CCS.push_back(Point3f(0,1,0));
	CCS.push_back(Point3f(0,0,1));

	// number of the TargetPoints
	int size = points.size();

	// step 0 : if no point inputed, set the normal (0,0,0)
	if(size==0)
		return;

	// step 1 : get expectation of coordinates of the points
	float x_mean = 0;
	float y_mean = 0;
	float z_mean = 0;

	for(int i=0;i<size;i++)
	{
		x_mean += points[i].x;
		y_mean += points[i].y;
		z_mean += points[i].z;
	}
	x_mean /= float(size);
    y_mean /= float(size);
    z_mean /= float(size);

	centor.x = x_mean;
	centor.y = y_mean;
	centor.z = z_mean;

	//cout<<"size: "<<size<<endl;
	//cout<<"mean: "<<x_mean<<", "<<y_mean<<", "<<z_mean<<", "<<endl;

	//
	// step 2 : Covariance matrix
	//
	Mat Cov = Mat::zeros(3,3,CV_32F); //Covariance matrix
	//cout<<"Cov\n"<<Cov<<endl;

	for(int i=0;i<size;i++)
	{
		points[i].x -= x_mean;
    	points[i].y -= y_mean;
		points[i].z -= z_mean;
		//cout<<"points[i]: "<<points[i]<<endl;

		Mat m = Mat::zeros(3,1,CV_32F);
		m.ptr<float>(0)[0] = points[i].x;
		m.ptr<float>(1)[0] = points[i].y;
		m.ptr<float>(2)[0] = points[i].z;
		//cout<<"m: "<<m<<endl;

		Cov = Cov + m*m.t();
		//cout<<"Cov: "<<Cov<<endl;
	}
	//cout<<"Cov: "<<Cov<<endl;

	Cov /= float(size);
	//cout<<"Cov: "<<Cov<<endl;

	//
	// PCA
	//
	// A (Cov)=u*w*vt
	// A (Cov): the input and decomposed matrix - Cov: \n"<<Cov<<endl<<endl;
	// w : calculated singular values: \n"<<w<<endl<<endl;
	// u : calculated left singular vectors: \n"<<u<<endl<<endl;
	// vt: transposed matrix of right singular values: \n"<<vt<<endl<<endl;
	Mat w,u,vt;
	SVD::compute(Cov, w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	//SVD::compute(Cov, w,u,vt, cv::SVD::FULL_UV);

	//cout<<"Singular Values: "<<w<<endl;
	//cout<<"Singular Vectors: "<<u<<endl;

	for(int i=0;i<3;i++)
	{
		CCS[i].x = vt.ptr<float>(i)[0];
		CCS[i].y = vt.ptr<float>(i)[1];
		CCS[i].z = vt.ptr<float>(i)[2];
	}


}
// !Pose estimation, PCA is used.

// Delaunay Triangulation
void Triangulation_Delaunay(vector<Point2f> points, vector<Vec6f> &triangles, vector<double> &boundries_rectangle)
{
	// step 1 : build a rectagular
	double minx = 0;
	double maxx = 0;
	double miny = 0;
	double maxy = 0;
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;

		if(minx>x) minx=x;
		if(maxx<x) maxx=x;

		if(miny>y) miny=y;
		if(maxy<y) maxy=y;
	}

	minx -= 1;
	miny -= 1;

	maxx += 1;
	maxy += 1;

	// set min and max values to boundries_rectangle
	boundries_rectangle.clear();
	boundries_rectangle.push_back(minx);
	boundries_rectangle.push_back(maxx);
	boundries_rectangle.push_back(miny);
	boundries_rectangle.push_back(maxy);


	// step 2 : Rectangle to be used with Subdiv2D
	Rect rect(minx, miny, maxx-minx, maxy-miny);

	// step 3 : Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// step 4 : Insert points into subdiv
	for(int i=0;i<points.size();i++)
	{
		subdiv.insert(points[i]);
	}

	// debug
	cout<<"Size of Triangulation_Delaunay points: "<<points.size()<<endl;
	cout<<"minx, maxx: "<<minx<<", "<<maxx<<endl;
	cout<<"miny, maxy: "<<miny<<", "<<maxy<<endl;
	// !debug

	// step 5 : Get Triangles
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);

	// step 6 : return the triangles
	triangles = triangleList;

	return;
}
// !Delaunay Triangulation

// Determine that if a point were inside a triangle
bool IsInsideATriangle(Vec6f Triangle, Point2f Point)
{
	// Point : the point that should be determined 
	// Triangle: the triangle that the point should be located in or not
	// Trangle ABC
	Vector2d TriA(Triangle[0], Triangle[1]);
	Vector2d TriB(Triangle[2], Triangle[3]);
	Vector2d TriC(Triangle[4], Triangle[5]);

	// Point
	Vector2d P(Point.x, Point.y);

	// Edges of the triangle ABC
	// v0 = C-A
	// v1 = B-A
	// v2 = Point-A
	Vector2d v0 = TriC-TriA;
	Vector2d v1 = TriB-TriA;
	Vector2d v2 = P   -TriA;

	//
	double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

	//
    double inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	// 
    double u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;

	//
	if (u < 0 || u > 1) // if u out of range, return directly
    {
        return false ;
    }

    double v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
    if (v < 0 || v > 1) // if v out of range, return directly
    {
        return false ;
    }

    return u + v <= 1 ;
}
// !Determine that if a point were inside a triangle


// Area Estimation
void Area_method1_Pixelation(vector<Point2f> points, vector<Vec6f> triangles, vector<double> boundries_rectangle, double &area, vector<Point2f> &pixelsInsideSilce)
{
	// etimate the area of the slice by pixelating the surface in which the slice lies.
	// step 0 : determine the bin sizes of the surface
	double mins[2];
	double maxs[2];
	mins[0] = boundries_rectangle[0];
    maxs[0] = boundries_rectangle[1];
    mins[1] = boundries_rectangle[2];
    maxs[1] = boundries_rectangle[3];

	int binSize = 20;
	double binWidths[2]; 
	binWidths[0] = (maxs[0]-mins[0])/double(binSize); // X axis
	binWidths[1] = (maxs[1]-mins[1])/double(binSize); // Y axis

	cout<<"binWidths : "<<binWidths[0]<<", "<<binWidths[1]<<endl;


	// step 1 : determine that if the bins located in any triangles
	// prepare a counter 
	int counter = 0;

	// prepare a vector to store pixels inside the slice
	pixelsInsideSilce.clear();

	// loop
	for(int i=0;i<binSize;i++) // X axis
	for(int j=0;j<binSize;j++) // Y axis
	{
		// centor of the current bin
		double x = mins[0] + binWidths[0]*(double(i)+0.5); // X centor of the current bin
		double y = mins[1] + binWidths[1]*(double(j)+0.5); // Y centor of the current bin

		//
		Point2f P(x, y);

		// loop 
		bool isInside = false;
		for(int k=0;k<triangles.size();k++)
		{
			isInside = IsInsideATriangle(triangles[k], P);
			if(isInside==true)
			{
				pixelsInsideSilce.push_back(P);
				break;
			}
		}
		if(isInside==true)
		{
			counter ++;
		}
	}

	// step 2 : estiamte the area
	area = binWidths[0]*binWidths[1]*double(counter);

	return;
}
// !Area Estimation


int main(int argc, char *argv[])
{
	cout<<"Hellol"<<endl;

	//
	// step 0 : start timing 
	//
	time_t timeStart = time(0);

	//
	// step 1 : import point cloud
	//
	vector<Point3f> points;
	string filename = "../../step1-pointCloud/build-3DCameraNormalEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/data_NorMap_RGBCam_Control_350000.txt";
	int IsFileGood = ReadPointCloud(filename, points);

	// debug
	cout<<"points.size(): "<<points.size()<<endl;
	// !debug

	//
	// step 2 :  get the maximum value of distances
	//
	double maxDist = 0;
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;
		double dist = sqrt(x*x + y*y + z*z);

		if(maxDist<dist) maxDist=dist;
	}
	cout<<"maxDist : "<<maxDist<<endl;
	// !get the maximum value of distances

	//
	// step 3 : Hough Voting
	//
	//
	// step 3.1 : build the Hough Space, which is a 3D histogram 
	//
	int N_bin = 20;
	double min = 0.;
	double maxx = M_PI; // theta
	double maxy = 2.*M_PI; // phi
	double maxz = maxDist; // r, maximum distance from the point to the origin

	double binWidthx = (maxx-min)/double(N_bin);
	double binWidthy = (maxy-min)/double(N_bin);
	double binWidthz = (maxz-min)/double(N_bin);

	Histogram3D h(	N_bin, min, maxx, 
					N_bin, min, maxy,
					N_bin, min, maxz
				);

	// test
	//double mint = 10;
	//double binWidth = 3;
	//double t1 = 9;
	//double t2 = ((t1 - mint)/binWidth);
	//cout<<"t2: "<<t2<<", ceil "<<ceil(t2)<<endl;
	// !test

	//
	// step 3.2 :  fill histogram
	//
	for(int i=0;i<points.size(); i++)
	{
		for(int hi=0;hi<N_bin;hi++)
		for(int hj=0;hj<N_bin;hj++)
		{
			double x = points[i].x;
			double y = points[i].y;
			double z = points[i].z;

			double theta = min + (double(hi)+0.5)*binWidthx;
			double phi   = min + (double(hj)+0.5)*binWidthy;
			double r =	x*sin(theta)*cos(phi) 
					  + y*sin(theta)*sin(phi)
					  + z*cos(theta);
			h.Fill(theta, phi, r, 1);
		}
	}

	//
	// step 3.3 : output the histogram
	//
	ofstream fileh3("data_histogram3d.txt");
	for(int i=0;i<N_bin;i++)
	for(int j=0;j<N_bin;j++)
	for(int k=0;k<N_bin;k++)
	{
		double counts = h.GetBinCounts(i,j,k);
		double x = 0; // theta
		double y = 0; // phi
		double z = 0; // r
		h.GetBinCentor(i, j, k, x, y, z);

		fileh3<<x<<" "<<y<<" "<<z<<" "<<counts<<endl;
	}
	fileh3.close();


	//
	// step 3.4 : normalization
	//
	h.Normalize();
	//h.Show();


	//
	// step 3.5 : determine planes that we found by Hough transformation
	//
	vector<double> List_theta;
	vector<double> List_phi;
	vector<double> List_r;
	vector<double> List_Counts;

	double threshold = 0.001;
	for(int i=0;i<N_bin;i++)
	for(int j=0;j<N_bin;j++)
	for(int k=0;k<N_bin;k++)
	{
		double counts = h.GetBinCounts(i,j,k);
		if(counts>threshold)
		{
			double x = 0; // theta
			double y = 0; // phi
			double z = 0; // r
			h.GetBinCentor(i, j, k, x, y, z);

			cout<<"Plane - ID(x,y,z): "<<i<<", "<<j<<", "<<k
				<<"; BinCentor(x,y,z): "<<x<<", "<<y<<", "<<z
				<<"; Counts: "<<counts<<endl;

			List_theta.push_back(x);
			List_phi  .push_back(y);
			List_r    .push_back(z);
			List_Counts.push_back(counts);
		}
	}

	//// test
	//vector<int> vec{1,2,3,4,5,6};
	//for(auto it = vec.begin(); it != vec.end(); it++)
	//{
	//	if(*it == 3)
	//	{
	//		it = vec.erase(it);
	//		if(it==vec.end()) break;
	//	}
	//}

	//for(int i=0;i<vec.size();i++)
	//{
	//	cout<<"ID: "<<i<<"; value: "<<vec[i]<<endl;
	//}
	//// !test

	//
	// step 4 : find best plane ID for each point
	//
	vector<int> planeID;

	//// Method 0 : the weighted voting metheod is used 
	//FindPlane_SdandardHough(points,
	//			 List_theta, 
    //             List_phi, 
    //             List_r, 
    //             List_Counts,
    //             planeID
	//			);


	//// Method 1 : the weighted voting metheod is used 
	//FindPlane_WV(points,
	//			 List_theta, 
    //             List_phi, 
    //             List_r, 
    //             List_Counts,
    //             planeID
	//			);

	// Method 2 : find the plane corresponding to the largest votes
	FindPlane_StaHT(points,
				 List_theta, 
                 List_phi, 
                 List_r, 
                 List_Counts,
                 planeID
				);


	//
	// step 5 : output points with planeIDs
	//
	ofstream fileOut("data_pointsWithShapes.txt");
	for(int i=0;i<points.size();i++)
	{
		int planeIDCur = planeID[i];

		if(planeIDCur==-1) continue;

		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;

		string PlaneName = to_string(planeIDCur) + "-Pla";
		fileOut<<PlaneName<<" "<<x<<" "<<y<<" "<<z<<endl;
	}
	fileOut.close();

	//
	// step 6 : end timing 
	//
	time_t timeEnd = time(0);
	time_t timeInterval = timeEnd - timeStart;
	cout<<"timeInterval: "<<timeInterval<<" s"<<endl;

	//
	// step 7 : analysis
	//
	//
	// step 7.1 : analysis - number of the points that correspond to the plane
	//
	int pointCounter = 0;
	for(int i=0;i<points.size();i++)
	{
		if(planeID[i]!=-1)
			pointCounter ++;
	}
	double detectionRatio = double(pointCounter)/double(points.size());

	cout<<"analysis : number of the points that correspond to the plane"<<endl;
	cout<<"pointCounter: "<<pointCounter<<"; size of the point cloud: "<<points.size()<<endl;
	cout<<"detectionRatio: "<<detectionRatio<<endl;

	//
	// step 7.1 : analysis - number of the points that correspond to the plane
	//
	vector<int> planeIDUsed;
	for(int i=0;i<planeID.size();i++)
	{
		int ID = planeID[i];

		if(ID==-1)
			continue;

		int CurrentIDExist = 0;
		for(int j=0;j<planeIDUsed.size();j++)
		{
			if(ID==planeIDUsed[j])
			{
				CurrentIDExist ++;
				continue;
			}
		}
		if(CurrentIDExist==0)
		{
			planeIDUsed.push_back(ID);
		}
	}
	cout<<"Number of Planes that are detected: "<<planeIDUsed.size()<<endl;

	//
	// step 8 : pose estiamtion
	//

	//
	// step 8.1 : Eliminate the ground, 
	// which is the plane that has been detected in the previous steps.
	// The rest of points are corresponding to human.  
	//
	vector<Point3f> HumanPoints;
	for(int i=0;i<points.size();i++)
	{
		// if the point belongs to a plane, we do not need it.
		if(planeID[i]!=-1) continue;

		HumanPoints.push_back(points[i]);
	}
	
	// debug
	cout<<"Number of Human_points: "<<HumanPoints.size()<<endl;
	// !debug

	//
	// step 8.2 : Pose normalization. PCA is used.
	// The purpose of this step is to get the Canonical Coordinate System (CCS), 
	// which corresponds to three unit vectors.
	//
	vector<Point3f> CCS; // the Canonical Coordinate System, which contains three unit vectors.
	Point3f centor; // centor of the human points
	PoseEstimation(HumanPoints, CCS, centor);

	//
	// step 8.3 : output the pose, which includes three unit vectors and the centor of the human
	//
	ofstream fileS81("data_human.txt");
	for(int i=0;i<HumanPoints.size();i++)
	{
		double x = HumanPoints[i].x;
		double y = HumanPoints[i].y;
		double z = HumanPoints[i].z;

		string Name = "Human";
		fileS81<<Name<<" "<<x<<" "<<y<<" "<<z<<endl;
	}
	fileS81.close();

	ofstream fileS82("data_humanPose.txt");
	fileS82<<"Centor"<<" "<<centor.x<<" "<<centor.y<<" "<<centor.z<<endl;
	for(int i=0;i<3;i++)
	{
		double x = CCS[i].x;
		double y = CCS[i].y;
		double z = CCS[i].z;
		fileS82<<"Vector"<<i<<" "<<x<<" "<<y<<" "<<z<<endl;
	}
	fileS82.close();

	//
	// step 9 : volume estiamte
	//
	//
	// step 9.1 : rigid body transformation, collaps the 3-rd axis in CCS and the Z-axis of the world
	//
	// step (0) : get rotation axis and angle
	// the main axis of the human is the third vector of CCS
	Vector3d vHZ(CCS[2].x,
				 CCS[2].y,
				 CCS[2].z
				);

	Vector3d vWZ(0,0,1); // Z axis of the world coordinate system 

	Vector3d RotationAxis(1,0,0);
	RotationAxis = vHZ.cross(vWZ);
	// normalization of the rotation axis
	RotationAxis /= RotationAxis.norm(); 

	//double CosRotationAngle = vWZ.dot(vHZ)/(vWZ.norm()*vWZ.norm());
	double CosRotationAngle = vWZ.dot(vHZ);
	double RotationAngle = acos(CosRotationAngle);

	cout<<"RotationAxis: "<<RotationAxis<<endl;
	cout<<"RotationAngle: "<<RotationAngle<<endl;

	AngleAxisd angleAxis(RotationAngle, RotationAxis);
	Matrix3d RotationMatrix = angleAxis.matrix();
	cout<<"RotationMatrix: \n"<<RotationMatrix<<endl;

	// step (1) : get translation vector
	Vector3d translation(centor.x,centor.y,centor.z);

	// step (2) : pose normalization, rotate and translate the human 
	vector<Point3f> NorHumanPoints;
	for(int i=0;i<HumanPoints.size();i++)
	{
		double x = HumanPoints[i].x;
		double y = HumanPoints[i].y;
		double z = HumanPoints[i].z;
		Vector3d Point(x,y,z);

		Vector3d PointNor(0,0,0); // normalized point
		PointNor = RotationMatrix*(Point - translation);
		//cout<<"ID: "<<i<<"; point: "<<PointNor.transpose()<<endl;

		Point3f point;
		point.x = PointNor(0);
		point.y = PointNor(1);
		point.z = PointNor(2);
		//cout<<"ID: "<<i<<"; point: "<<PointNor.transpose()<<"; point: "<<point<<endl;

		NorHumanPoints.push_back(point);
	}

	// step (3) : output the human point cloud, whose pose has been normalized
	ofstream fileS91("data_humanPoseNormalized.txt");
	for(int i=0;i<NorHumanPoints.size();i++)
	{
		double x = NorHumanPoints[i].x;
		double y = NorHumanPoints[i].y;
		double z = NorHumanPoints[i].z;

		string Name = "HumanPoseNormalized";
		fileS91<<Name<<" "<<x<<" "<<y<<" "<<z<<endl;
	}
	fileS91.close();


	//
	// step 9.2 : cut the human model into slices
	//
	// step (0) : get the maximum and minimum value in Z axis
	double maxZ = 0;
	double minZ = 0;
	for(int i=0;i<NorHumanPoints.size();i++)
	{
		double z = NorHumanPoints[i].z;

		if(maxZ<z) maxZ=z;

		if(minZ>z) minZ=z;
	}
	cout<<"maxZ: "<<maxZ<<"; minZ: "<<minZ<<endl;

	// step (1) : get slices and evaluate areas of all slices
	// Delauney triangulation is used.
	vector<int> HumanPointSliceID;
	for(int i=0;i<NorHumanPoints.size();i++)
	{
		HumanPointSliceID.push_back(-1);
	}

	int numberSlices = 10;
	double widthPerSlice = (maxZ-minZ)/double(numberSlices);
	for(int i=0;i<NorHumanPoints.size();i++)
	{
		double z = NorHumanPoints[i].z;
		int ID = (z-minZ)/widthPerSlice;
		HumanPointSliceID[i] = ID;

		//// debug
		//double minZCur = minZ + widthPerSlice*double(ID);
		//double maxZCur = minZ + widthPerSlice*double(ID+1);
		//cout<<"ID: "<<ID<<"; z: "<<z<<"; minZCur, maxZCur: "<<minZCur<<", "<<maxZCur<<endl;
		//// !debug
	}

	// step (2) : output slices of the normalized human model
	ofstream fileS92("data_slicesHumanModel.txt");
	for(int i=0;i<NorHumanPoints.size();i++)
	{
		double x = NorHumanPoints[i].x;
		double y = NorHumanPoints[i].y;
		double z = NorHumanPoints[i].z;

		int ID = HumanPointSliceID[i];

		string Name =  to_string(ID) + "-Slice";
		fileS92<<Name<<" "<<x<<" "<<y<<" "<<z<<endl;
	}
	fileS92.close();

	//
	// step 9.3 : Estimate areas of the slices. Delauney triangulation is used.
	//            Estiamte the volume of the human body
	//
	// step (0) : divide the human point cloud into slices, and triangulate 

	// volume of the human body
	double HumanVolume = 0;

	// output the triangles
	ofstream fileS93("data_DelaunayTriangles.txt");

	// output the pixels inside the slice
	ofstream fileS94("data_pixelsInsideTheSlices.txt");

	// output the areas 
	ofstream fileS95("data_areaOfEachSlice.txt");


	// triangulation : find triangles
	for(int i=0;i<numberSlices;i++)
	{
		int SliceID = i;
		vector<Point2f> NorHumanPoints_ASlice;
		for(int j=0;j<NorHumanPoints.size();j++)
		{
			if(HumanPointSliceID[j]==SliceID)
			{
				Point2f p(NorHumanPoints[j].x,
						  NorHumanPoints[j].y);
				NorHumanPoints_ASlice.push_back(p);
			}
		}

		double area = 0;
		vector<Vec6f> triangles;
		vector<double> boundries_rectangle;

		// triangulation with Delaunay 
		Triangulation_Delaunay(NorHumanPoints_ASlice, triangles, boundries_rectangle);

		// estimate area
		// method 1 : pixelate a surface of the slice
		vector<Point2f> pixelsInsideSilce;
		Area_method1_Pixelation(NorHumanPoints_ASlice, 
								triangles, 
								boundries_rectangle, 
								area, 
								pixelsInsideSilce
								);

		// debug
		cout<<"Area: "<<area/1e6<<" m2"<<endl;
		// !debug

		// estiamte the volume
		HumanVolume += (widthPerSlice*area/1e9); // unit : m3

		// output triangles
		for(int j=0;j<triangles.size();j++)
		{
			//cout<<"Triangle ID: "<<j<<"; triangle: "<<triangles[j]<<endl;
			fileS93<<SliceID<<"-Slice"<<" ";
			fileS93<<triangles[j][0]<<" "<<triangles[j][1]<<" ";
			fileS93<<triangles[j][2]<<" "<<triangles[j][3]<<" ";
			fileS93<<triangles[j][4]<<" "<<triangles[j][5]<<"\n";
		}

		// output pixels
		for(int j=0;j<pixelsInsideSilce.size();j++)
		{
			//cout<<"Triangle ID: "<<j<<"; triangle: "<<triangles[j]<<endl;
			fileS94<<SliceID<<"-Slice"<<" ";
			fileS94<<pixelsInsideSilce[j].x<<" "<<pixelsInsideSilce[j].y<<endl;
		}

		// output areas
		fileS95<<SliceID<<"-Slice"<<" ";
		fileS95<<area/1e6<<" m2 area"<<endl;
	}

	// output triangles
	fileS93.close();

	// output the pixels inside the slice
	fileS94.close();

	// output the areas 
	fileS95.close();

	// 
	cout<<"HumanVolume: "<<HumanVolume<<" m3"<<endl;


    return 0;
}

