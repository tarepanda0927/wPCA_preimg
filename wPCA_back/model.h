#pragma once
#include<list>
#include<string>
#include<Eigen/Core>
#include<Eigen/Eigenvalues>

namespace saito {

	/*=== Class model ========================= �������� =========================*/

	/**
	* PCA�ɂ�郂�f�����������߂̃N���X
	*
	* @author A. Saito
	* @version 1.0
	*/
	template< class T >
	class model {
	public:
		typedef typename Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > Mat_t;
		typedef typename Eigen::Matrix< T, Eigen::Dynamic, 1 > Vec_t;
	protected:
		std::size_t num_of_dimX; /**< ���͋�Ԃ̎����� ( = D ) */
		std::size_t num_of_dimY; /**< ������Ԃ̎����� ( = d ) */
		Vec_t X_mean;            /**< ���σx�N�g�� ( D x 1 ) */
		Mat_t M;                 /**< �ʑ��s�� ( D x d ) */
		Vec_t Eval;              /**< �ŗL�l ( d x 1 ) */
		Vec_t sqrtL;             /**< �ŗL�l�̕����� ( d x 1 ) */
		Vec_t invsqrtL;          /**< �ŗL�l�̕������̋t�� ( d x 1 ) */
		Vec_t CCR;               /**< �ݐϊ�^�� ( d x 1 ) */
		Vec_t w;                 /**< �d�݃x�N�g�� ( d x 1 ) */
		bool isModelExists;      /**< ���f�������[�h�ς݂ł��邩�ǂ�����\���t���O */
		bool isWeighted;      /**< ���f�������[�h�ς݂ł��邩�ǂ�����\���t���O */

		void get_list_txt(std::list< std::string > &list_txt, std::string path_list);
		long get_file_size(std::string filename);
		void read_bin(T *p, std::size_t num, std::string file_name);
		void write_bin(const T *p, std::size_t num, std::string file_name);
		void write_text(const T *p, size_t num, std::string file_name);
		virtual void load_common(std::string modelDir, const std::size_t num_of_basis);
		void load_base(std::string modelDir, const std::size_t num_of_basis);

	public:
		model() {
			isModelExists = false;
			isWeighted = false;
		}
		/**
		* @brief PCA�ɂ�郂�f���̍쐬
		*
		* ���̓f�[�^�̃��X�g���烂�f�����쐬����D
		* @param[in] path_list �w�K�f�[�^���X�g�̃p�X
		* @return PCA�̎��s�ɐ����������ǂ�����\���t���O
		* @retval TRUE PCA�̎��s�ɐ���
		* @retval FALSE PCA�̎��s���Ɏ��s
		* @attention ���̓f�[�^�͑S�ē����������łȂ���΂Ȃ�Ȃ�
		*/
		void PCA(const std::string &path_list);
		/**
		* @brief PCA�ɂ�郂�f���̍쐬
		*
		* ���̓f�[�^�̃��X�g���烂�f�����쐬����D
		* @param[in] �f�[�^�s��
		* @return PCA�̎��s�ɐ����������ǂ�����\���t���O
		* @retval TRUE PCA�̎��s�ɐ���
		* @retval FALSE PCA�̎��s���Ɏ��s
		* @attention ���̓f�[�^�͑S�ē����������łȂ���΂Ȃ�Ȃ�
		*/
		void PCA(Mat_t X);
		/**
		* @brief WPCA�ɂ�郂�f���̍쐬
		*
		* ���̓f�[�^�̃��X�g���烂�f�����쐬����D
		* @param[in] path_list �w�K�f�[�^���X�g�̃p�X
		* @param[in] alpha �p�����[�^�� (w = 1/(1+exp(��*d)))
		* @return WPCA�̎��s�ɐ����������ǂ�����\���t���O
		* @retval TRUE PCA�̎��s�ɐ���
		* @retval FALSE PCA�̎��s���Ɏ��s
		* @attention ���̓f�[�^�͑S�ē����������łȂ���΂Ȃ�Ȃ�
		*/
		void WPCA(const std::string &path_list, T alpha);
		/**
		* @brief ���f���̃t�@�C���ւ̏o��
		*
		* ���f�����w�肵���f�B���N�g���ɕۑ�����
		* @param[in] outDir �o�̓f�B���N�g��
		* @attention �t�H���_���Ȃ��ꍇ�͎����I�ɍ쐬���邪�C
		* �e�f�B���N�g���̑��݂���ꏊ�̂݁D
		*/
		void output(std::string outDir);
		/**
		* @brief ���f���̓ǂݍ���
		*
		* ���f�����w�肵���f�B���N�g�����烍�[�h����D
		* ��ꐔ�͑S�āD
		* @param[in] modelDir ���f���̃f�B���N�g��
		*/
		void load(std::string modelDir);
		/**
		* @brief ���f���̓ǂݍ���
		*
		* ���f�����w�肵���f�B���N�g�����烍�[�h����D
		* ��ꐔ�͎w�肵���������D
		* @param[in] modelDir ���f���̃f�B���N�g��
		* @param[in] num_of_basis �ǂݍ��ފ��̐�
		* @attention �I�[�o�[���[�h�p��virtual�w��
		*/
		void load_with_N(std::string modelDir, const std::size_t num_of_basis);
		/**
		* @brief ���f���̓ǂݍ���
		*
		* ���f�����w�肵���f�B���N�g�����烍�[�h����D
		* ��ꐔ�͗ݐϊ�^���Ŏw��D
		* @param[in] modelDir ���f���̃f�B���N�g��
		* @param[in] num_of_basis �ǂݍ��ފ��̐�
		* @attention �ǂݍ��܂���ꐔ�́C�ݐϊ�^�����w�肵���l�ȏ�
		* �ƂȂ�ŏ��̖{���D
		*/
		void load_with_CCR(std::string modelDir, const T ccr);
		/**
		* @brief ���m���͂̕�����Ԃւ̎ʑ� (Out-of-sample)
		*
		* ���m���͂𐳋K���Ȃ��ŕ�����Ԃ֎ʑ�
		* @param[out] Y �ʑ���̎听�����_ (���K���Ȃ�)
		* @param[in]  X �ʑ�������W
		*/
		void out_of_sample(Mat_t &Y, const Mat_t &X);
		/**
		* @brief ���m���͂̕�����Ԃւ̎ʑ� (Out-of-sample)
		*
		* ���m���͂𐳋K�����ĕ�����Ԃ֎ʑ�
		* @param[out] Alpha �ʑ���̎听�����_ (���K������)
		* @param[in]  X �ʑ�������W
		*/
		void out_of_sample_normal(Mat_t &Alpha, const Mat_t &X);
		/**
		* @brief ������Ԃ�����͋�Ԃւ̋t�ʑ� (pre_image)
		*
		* ������Ԃ̍��W����͋�Ԃ̍��W�֋t�ʑ�
		* @param[out] X �t�ʑ���̍��W
		* @param[in] Y �t�ʑ�����听�����_ (���K���Ȃ�)
		*/
		void pre_image(Mat_t &X, const Mat_t &Y);
		/**
		* @brief ���K�����ꂽ������Ԃ�����͋�Ԃւ̋t�ʑ� (pre_image)
		*
		* ���K�����ꂽ������Ԃ̍��W����͋�Ԃ̍��W�֋t�ʑ�
		* @param[out] X �t�ʑ���̍��W
		* @param[in] Alpha �t�ʑ�����听�����_ (���K������)
		*/
		void pre_image_normal(Mat_t &Alpha, const Mat_t &Y);
		/**
		* @brief �ʑ��E�t�ʑ��ɂ��f�[�^�̍č\��
		*
		* @param[out] Xout �ʑ��E�t�ʑ��ɂ���čč\�����ꂽ�f�[�^�s��
		* @param[in] Xin ���͋�Ԃɂ�����f�[�^�s��
		*/
		void reconstruction(Mat_t &Xout, const Mat_t &Xin);
		/**
		* @brief ���f�����̕\��
		*
		* @attention �f�o�b�O�p�D��K�͂ȃf�[�^�͕\�����Ȃ����ƁD
		*/
		virtual void disp();
		virtual ~model() {}
	};

	/*=== Class model ========================= �����܂� =========================*/


};
