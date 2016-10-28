#include "naricommon.h"
#include "ip/narigaussian.h"

#include <omp.h>
#include "stdlib.h"
#include "naritimer.h"
#include "naricaseinfo.h"
#include <algorithm>
#include "other/narimhd.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <sstream>
#include "naricommon.h"
#include "narisystem.h"
#include "naripathline.h"
#include "naricaseinfo.h"
#include "ip/narimorphology.h"
#include "ip/narilabeling.h"

#include "ip/naricontour.h"
#include "ip/naridistance.h"
#include "other/narimhd.h"
#include "naritimer.h"
#include "ip/narirbf.h"
#include "../mist/vector.h"
#include "narivectorpp.h"
#include "ip/nariinterpolate.h"
#include<iostream>
#include"model.h"
#include<fstream>
#include<vector>
#include "info.h"
#include "raw_io.h"
 
int main(int argc, char *argv[]) {
	text_info input_info;
	input_info.input(argv[1]);
	//テキストデータ読み込み
	std::vector<std::string> fcase;
	std::vector<std::string> rcase;
	std::ifstream f_case(input_info.dir_list + input_info.case_flist);
	std::ifstream r_case(input_info.dir_list + input_info.case_rlist);
	std::string buf_ft;
	std::string buf_rt;
	while (f_case&& getline(f_case, buf_ft))
	{
		fcase.push_back(buf_ft);
	}
	while (r_case&& getline(r_case, buf_rt))
	{
		rcase.push_back(buf_rt);
	}

	for (int i = 0; i < fcase.size(); i++) {
		std::cout << fcase[i] << std::endl;
		std::cout << "a" << std::endl;
		saito::model<double> wpca;
		std::vector<double> result;
		std::stringstream dir_wpca;
		std::stringstream dir_Result;
		dir_wpca << input_info.dir_wPCA << rcase[i] ;
		wpca.load_with_N(dir_wpca.str(), input_info.Rd);
		//通常はこっち
		dir_Result << input_info.dir_GP << fcase[i] << "/mean.raw";
		//dir_Result << input_info.dir_GP << fcase[i] << "/linear.raw";  //線形予測の結果を出したいときはこっち
		read_vector(result, dir_Result.str());
		Eigen::MatrixXd result_read = Eigen::Map<Eigen::MatrixXd>(&result[0], input_info.Rd, 1);
		
		////正解主成分スコアを読み込んで逆投影する
		//dir_Result << input_info.dir_ans << "/" << rcase[i] << "/mat.raw";
		//read_vector(result, dir_Result.str());
		//int k = (fcase.size()-2) * i;
		//Eigen::MatrixXd result_read = Eigen::Map<Eigen::MatrixXd>(&result[k], input_info.Rd, 1);
		//std::cout << result_read << std::endl;
		
		Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > Y;
		Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > weight;
		
		wpca.pre_image(Y, result_read);
		wpca.cal_weight(weight);
		nari::vector<double> R ;
		std::stringstream O_file;
		std::ostringstream O_file_r;
		std::ostringstream O_file_rl;
		O_file << input_info.dir_out << rcase[i] << "/mat";
		O_file_r << input_info.dir_out << rcase[i] << "/result.raw";
		O_file_rl << input_info.dir_out << rcase[i] << "/result_label.raw";
		nari::system::make_directry(O_file.str());
		write_matrix_raw_and_txt(Y, O_file.str());
		//閾値処理
		for (int j = 0; j < Y.rows(); j++) {
			for (int k = 0; k < Y.cols(); k++) {
				double rs = Y(j, k) / weight(j, k);
				R.push_back(rs);
			}
		}
		nari::vector<unsigned char> R_label(R.size(),0);
		for (int s = 0; s < R.size(); s++) {
			if (R[s] < 0) R_label[s] = 1;
		}

		nari::mhd mhdr;
		mhdr.size123(192,192,120);
		mhdr.reso123(0.26,0.26,0.27);
		mhdr.save_mhd_and_image(R_label, O_file_rl.str());
		std::string path = "H:/spatial_normalization/output/Mbrain/premove/normalized_label/" + rcase[i] + "_nmlzd.raw";
		std::string path_dist = "H:/spatial_normalization/output/Mbrain/premove/normalized_label/" + rcase[i] + "_dist.raw";
		mhdr.save_mhd_and_image(R_label, path);
		mhdr.save_mhd_and_image(R, path_dist);
	}
}