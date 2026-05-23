#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define PI 3.1415926535
#define e 2.718281828
#define NZ 128	
#define NY 128   //此处是y轴的网格点数量
#define NX 128
#define NP 32
#define NL 200  //变网格处
#define m 5    // 连续变换率
#define MM 1    // 连续变换率
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1
//#define BLOCK_SIZE_ZZ 2 //共享内存

float** space2d(int nr, int nc);
float*** space3d(int nr, int ny, int nc);
void free_space2d(float** a, int nr);
void free_space3d(float*** b, int nr, int ny);
void wfile(char filename[], float** data, int nr, int nc);
void wfile3d(char filename[], float*** data, int nr, int ny, int nc);
void create_model(float*** vp, float*** vs, float*** rhos, float*** vf, float*** rho, float*** rhof, float*** M, float*** C, float*** C1, float*** C2, float*** HH, float*** H2u, float*** mu, int nr, int ny, int nc);
float*** extmodel(float*** init_model, int nr, int ny, int nc, int np);
//void create_model(float*** vp, float*** vs, float*** rho, float*** vf, float*** rhof, float*** lamda, float*** lamda2u, float*** mu, int nr, int ny, int nc);

int divUp(int a, int b) { return (a - 1) / b + 1; }
//-------------------------------------------------------------------------------------------------------------------------------
//计算震源
__global__ void Source(float* txx, float* tyy, float* tzz, float I_sou, int sn, int NX_ext, int NY_ext, int NZ_ext)
{
	//加震源
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	if (offset == sn)
	{
		txx[offset] = txx[offset] + I_sou;
		tyy[offset] = tyy[offset] + I_sou;
		tzz[offset] = tzz[offset] + I_sou;

	}
}
//------------------------------------------------------------------------------------------------------------------



__global__ void FD_V(float* vux, float* vuy, float* vuz, float* rho_tempx, float* rho_tempy, float* rho_tempz,
	float* txx, float* tyy, float* tzz, float* txz, float* txy, float* tyz,
	float* pmlxSxx, float* pmlySxy, float* pmlzSxz, float* pmlxSxy, float* pmlySyy, float* pmlzSyz, float* pmlxSxz, float* pmlySyz, float* pmlzSzz,
	float* SXxx, float* SXxy, float* SXxz, float* SYxy, float* SYyy, float* SYyz, float* SZxz, float* SZyz, float* SZzz,
	float* e_dxi, float* dxi, float* e_dxi2, float* dxi2, float* e_dyj, float* dyj, float* e_dyj2, float* dyj2, float* e_dzk, float* dzk, float* dzk2, float* e_dzk2,
	float* rhof_ext, float* ss, float* vwx, float* vwy, float* vwz, float* vwx2, float* vwy2, float* vwz2, float* SXss, float* SYss, float* SZss, float* pmlxss, float* pmlyss, float* pmlzss, float* C1x, float* C1y, float* C1z, float* C2x, float* C2y, float* C2z, float DT)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float CC1x, CC1y, CC1z, CC2x, CC2y, CC2z;
	float H = 0.01f;
	// 差分系数
	float h = H / m;
	float h1 = h / m;
	float h2 = h1 / m;
	float h3 = h2 / m;
	float c1 = 2 / (h + H);
	float c2 = -c1;
	float cc1 = 2 / (h + h1);
	float cc2 = -cc1;
	float ccc1 = 2 / (h2 + h1);
	float ccc2 = -ccc1;
	float cccc1 = 2 / (h2 + h3);
	float cccc2 = -cccc1;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下
	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
		//地层参数
		CC1x = 0.5 * C1x[offset] * DT + C2x[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempx[offset];
		CC2x = C2x[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempx[offset] - 0.5 * C1x[offset] * DT;

		CC1y = 0.5 * C1y[offset] * DT + C2y[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempy[offset];
		CC2y = C2y[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempy[offset] - 0.5 * C1y[offset] * DT;

		CC1z = 0.5 * C1z[offset] * DT + C2z[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempz[offset];
		CC2z = C2z[offset] - rhof_ext[offset] * rhof_ext[offset] * rho_tempz[offset] - 0.5 * C1z[offset] * DT;

		x1 = (txx[offset_r] - txx[offset]) / H;
		x2 = (txy[offset] - txy[offset_h]) / H;
		x3 = (txz[offset] - txz[offset_b]) / H;
		s1 = (ss[offset_r] - ss[offset]) / H;

		y1 = (tyy[offset_q] - tyy[offset]) / H;
		y2 = (txy[offset] - txy[offset_l]) / H;
		y3 = (tyz[offset] - tyz[offset_b]) / H;
		s2 = (ss[offset_q] - ss[offset]) / H;

		z1 = (tzz[offset_u] - tzz[offset]) / H;
		z2 = (txz[offset] - txz[offset_l]) / H;
		z3 = (tyz[offset] - tyz[offset_h]) / H;
		s3 = (ss[offset_u] - ss[offset]) / H;

		pmlxSxx[offset] = pmlxSxx[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * SXxx[offset] + x1);
		pmlySxy[offset] = pmlySxy[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * SXxy[offset] + x2);
		pmlzSxz[offset] = pmlzSxz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * SXxz[offset] + x3);
		pmlxss[offset] = pmlxss[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * SXss[offset] + s1);
		SXxx[offset] = x1; SXxy[offset] = x2; SXxz[offset] = x3; SXss[offset] = s1;
		x1 = x1 + pmlxSxx[offset];
		x2 = x2 + pmlySxy[offset];
		x3 = x3 + pmlzSxz[offset];
		s1 = s1 + pmlxss[offset];

		pmlxSxy[offset] = pmlxSxy[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * SYxy[offset] + y2);
		pmlySyy[offset] = pmlySyy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * SYyy[offset] + y1);
		pmlzSyz[offset] = pmlzSyz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * SYyz[offset] + y3);
		pmlyss[offset] = pmlyss[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * SYss[offset] + s2);
		SYxy[offset] = y2; SYyy[offset] = y1; SYyz[offset] = y3; SYss[offset] = s2;
		y2 = y2 + pmlxSxy[offset];
		y1 = y1 + pmlySyy[offset];
		y3 = y3 + pmlzSyz[offset];
		s2 = s2 + pmlyss[offset];

		pmlxSxz[offset] = pmlxSxz[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * SZxz[offset] + z2);
		pmlySyz[offset] = pmlySyz[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * SZyz[offset] + z3);
		pmlzSzz[offset] = pmlzSzz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * SZzz[offset] + z1);
		pmlzss[offset] = pmlzss[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * SZss[offset] + s3);
		SZxz[offset] = z2; SZyz[offset] = z3; SZzz[offset] = z1; SZss[offset] = s3;
		z2 = z2 + pmlxSxz[offset];
		z3 = z3 + pmlySyz[offset];
		z1 = z1 + pmlzSzz[offset];
		s3 = s3 + pmlzss[offset];

		if (C1x[offset] == 0.0f)
		{
			vwx[offset] = 0.0f;
		}
		else
		{
			vwx[offset] = (vwx[offset] * CC2x - DT * (rhof_ext[offset] * rho_tempx[offset] * (x1 + x2 + x3) + s1)) / CC1x;
		}

		if (C1y[offset] == 0.0f)
		{
			vwy[offset] = 0.0f;
		}
		else
		{
			vwy[offset] = (vwy[offset] * CC2y - DT * (rhof_ext[offset] * rho_tempy[offset] * (y1 + y2 + y3) + s2)) / CC1y;
		}

		if (C1z[offset] == 0.0f)
		{
			vwz[offset] = 0.0f;
		}
		else
		{
			vwz[offset] = (vwz[offset] * CC2z - DT * (rhof_ext[offset] * rho_tempz[offset] * (z1 + z2 + z3) + s3)) / CC1z;
		}

		vux[offset] = vux[offset] + DT * rho_tempx[offset] * (x1 + x2 + x3) - rhof_ext[offset] * rho_tempx[offset] * (vwx[offset] - vwx2[offset]);
		vuy[offset] = vuy[offset] + DT * rho_tempy[offset] * (y1 + y2 + y3) - rhof_ext[offset] * rho_tempy[offset] * (vwy[offset] - vwy2[offset]);
		vuz[offset] = vuz[offset] + DT * rho_tempz[offset] * (z1 + z2 + z3) - rhof_ext[offset] * rho_tempz[offset] * (vwz[offset] - vwz2[offset]);
		vwx2[offset] = vwx[offset]; vwy2[offset] = vwy[offset]; vwz2[offset] = vwz[offset];
	}
}

__global__ void FD_T(float* vux, float* vuy, float* vuz, float* txx, float* tzz, float* tyy, float* txz, float* txy, float* tyz,
	float* pmlxVux, float* muxy, float* muxz, float* muyz,
	float* pmlyVuy, float* pmlzVuz, float* pmlxVuy, float* pmlyVux, float* pmlyVuz, float* pmlzVuy, float* pmlzVux, float* pmlxVuz,
	float* Vuxx, float* Vuxy, float* Vuxz, float* Vuyx, float* Vuyy, float* Vuyz, float* Vuzx, float* Vuzy, float* Vuzz,
	float* e_dxi, float* dxi, float* e_dxi2, float* dxi2, float* e_dyj2, float* dyj2, float* e_dyj, float* dyj, float* dzk2, float* e_dzk2, float* e_dzk, float* dzk,
	float* Vwxx, float* Vwyy, float* Vwzz, float* vwx, float* vwy, float* vwz, float* C_ext, float* M_ext, float* HH_ext, float* H2u_ext, float* ss, float* pmlxVwx, float* pmlyVwy, float* pmlzVwz, float DT)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	float H = 0.01f;
	// 差分系数
	float h = H / m;
	float h1 = h / m;
	float h2 = h1 / m;
	float h3 = h2 / m;
	float c1 = 2 / (h + H);
	float c2 = -c1;
	float cc1 = 2 / (h + h1);
	float cc2 = -cc1;
	float ccc1 = 2 / (h2 + h1);
	float ccc2 = -ccc1;
	float cccc1 = 2 / (h2 + h3);
	float cccc2 = -cccc1;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下

	uxx = (vux[offset] - vux[offset_l]) / H;
	uyy = (vuy[offset] - vuy[offset_h]) / H;
	uzz = (vuz[offset] - vuz[offset_b]) / H;

	wx = (vwx[offset] - vwx[offset_l]) / H;
	wy = (vwy[offset] - vwy[offset_h]) / H;
	wz = (vwz[offset] - vwz[offset_b]) / H;

	uxy = (vux[offset_q] - vux[offset]) / H;
	uyx = (vuy[offset_r] - vuy[offset]) / H;

	uxz = (vux[offset_u] - vux[offset]) / H;
	uzx = (vuz[offset_r] - vuz[offset]) / H;

	uyz = (vuy[offset_u] - vuy[offset]) / H;
	uzy = (vuz[offset_q] - vuz[offset]) / H;


	pmlxVux[offset] = pmlxVux[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * Vuxx[offset] + uxx);
	pmlyVuy[offset] = pmlyVuy[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * Vuyy[offset] + uyy);
	pmlzVuz[offset] = pmlzVuz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * Vuzz[offset] + uzz);
	Vuxx[offset] = uxx; Vuyy[offset] = uyy; Vuzz[offset] = uzz;
	uxx = uxx + pmlxVux[offset];
	uyy = uyy + pmlyVuy[offset];
	uzz = uzz + pmlzVuz[offset];


	pmlxVwx[offset] = pmlxVwx[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * Vwxx[offset] + wx);
	pmlyVwy[offset] = pmlyVwy[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * Vwyy[offset] + wy);
	pmlzVwz[offset] = pmlzVwz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * Vwzz[offset] + wz);
	Vwxx[offset] = wx; Vwyy[offset] = wy; Vwzz[offset] = wz;
	wx = wx + pmlxVwx[offset];
	wy = wy + pmlyVwy[offset];
	wz = wz + pmlzVwz[offset];

	pmlxVuy[offset] = pmlxVuy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * Vuxy[offset] + uxy);
	pmlyVux[offset] = pmlyVux[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * Vuyx[offset] + uyx);
	Vuxy[offset] = uxy; Vuyx[offset] = uyx;
	uxy = uxy + pmlxVuy[offset];
	uyx = uyx + pmlyVux[offset];

	pmlxVuz[offset] = pmlxVuz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * Vuxz[offset] + uxz);
	pmlzVux[offset] = pmlzVux[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * Vuzx[offset] + uzx);
	Vuxz[offset] = uxz; Vuzx[offset] = uzx;
	uxz = uxz + pmlxVuz[offset];
	uzx = uzx + pmlzVux[offset];

	pmlyVuz[offset] = pmlyVuz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * Vuyz[offset] + uyz);
	pmlzVuy[offset] = pmlzVuy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * Vuzy[offset] + uzy);
	Vuzy[offset] = uzy; Vuyz[offset] = uyz;
	uyz = uyz + pmlyVuz[offset];
	uzy = uzy + pmlzVuy[offset];

	ss[offset] = ss[offset] - DT * (C_ext[offset] * (uxx + uyy + uzz) + M_ext[offset] * (wx + wy + wz));
	txx[offset] = txx[offset] + DT * (H2u_ext[offset] * (uyy + uzz) + HH_ext[offset] * uxx + C_ext[offset] * (wx + wy + wz));
	tyy[offset] = tyy[offset] + DT * (H2u_ext[offset] * (uxx + uzz) + HH_ext[offset] * uyy + C_ext[offset] * (wx + wy + wz));
	tzz[offset] = tzz[offset] + DT * (H2u_ext[offset] * (uxx + uyy) + HH_ext[offset] * uzz + C_ext[offset] * (wx + wy + wz));
	txy[offset] = txy[offset] + muxy[offset] * DT * (uxy + uyx);
	tyz[offset] = tyz[offset] + muyz[offset] * DT * (uyz + uzy);
	txz[offset] = txz[offset] + muxz[offset] * DT * (uxz + uzx);
}


//---------------------------------------------------------------------------------------------------------------
//主函数
int main()
{
	clock_t start, finish;
	start = clock();
	//给定参数
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	const int BLOCK_X = BLOCK_SIZE_X;
	const int BLOCK_Y = BLOCK_SIZE_Y;
	int sx = NX_ext / 2;           //震源坐标点号
	int sy = NY_ext / 2;
	int sz = 20 + NP;
	int sn = sx + sy * NX_ext + sz * NX_ext * NY_ext;
	int NT = 3500;	        //时间层数
	//int NT1 = NT * m * m; // 连续变换2次
	float H = 0.01;//空间步长
	float RC = 1.0 * pow(10.0, -6);
	float DT = 0.9 * pow(10.0, -6);	    //时间步长
	float DP = NP * H;
	float DT_H = DT / H;
	float F0 = 3 * pow(10.0, 3);		//震源主频,主频太高会造成频散
	float T0 = 1.2 / F0;
	float Vpmax = 4270.0;    //模型最大纵波速度,用于稳定性计算
	float Vpmin = 1500.0;	   //模型最小纵波速度,用于控制数值频散
	float Vsmax = 2650.0;
	float Vsmin = 1500.0;
	float** sis_x;  //地震记录z分量
	float** sis_y;
	float** sis_z;
	float** sis_vu;
	float** sis_vw;
	float** sis_p;
	sis_x = space2d(NZ_ext, NT);
	sis_y = space2d(NZ_ext, NT);
	sis_z = space2d(NZ_ext, NT);
	sis_vu = space2d(NZ_ext, NT);
	sis_vw = space2d(NZ_ext, NT);
	sis_p = space2d(NZ_ext, NT);
	//dim3 Blocksize1 = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);          //GPU中线程块分配
	//dim3 Gridsize1 = dim3(divUp(NX_ext, BLOCK_SIZE_X), divUp(NY_ext, BLOCK_SIZE_Y), 1); //GPU中线程分配

	dim3 Blocksize = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);          //GPU中线程块分配
	dim3 Gridsize = dim3(divUp(NX_ext, BLOCK_SIZE_X), divUp(NY_ext, BLOCK_SIZE_Y), divUp(NZ_ext, BLOCK_SIZE_Z)); //GPU中线程分配
	size_t mem_size = NZ_ext * NY_ext * NX_ext * sizeof(float);     //内存大小
	printf("gridsize = %d   %d    %d\n", divUp(NX_ext, BLOCK_SIZE_X), divUp(NY_ext, BLOCK_SIZE_Y), divUp(NZ_ext, BLOCK_SIZE_Z));
	//---------------------------------------------------------------------------
	//参数开辟内存
	float*** vs;	//初始模型横波速度
	float*** vp;	//初始模型纵波速度
	float*** vf;	//孔隙流体速度
	float*** rho;	//初始模型密度
	float*** rhof;  //孔隙流体密度
	float*** rhos;
	float*** mu;
	float*** vs_ext;
	float*** vp_ext;
	float*** vf_ext;
	float*** rho_ext;
	float*** rhof_ext;
	float*** mu_ext;
	float*** M, *** C, *** HH, *** H2u, *** C1, *** C2;
	float*** M_ext;
	float*** C_ext;
	float*** HH_ext;
	float*** H2u_ext;
	float*** C1_ext;
	float*** C2_ext;
	float*** rhos_ext;
	vs = space3d(NZ, NY, NX);
	vp = space3d(NZ, NY, NX);
	rhof = space3d(NZ, NY, NX);
	rhos = space3d(NZ, NY, NX);
	rho = space3d(NZ, NY, NX);
	vf = space3d(NZ, NY, NX);
	mu = space3d(NZ, NY, NX);
	M = space3d(NZ, NY, NX);
	C = space3d(NZ, NY, NX);
	HH = space3d(NZ, NY, NX);
	H2u = space3d(NZ, NY, NX);
	C1 = space3d(NZ, NY, NX);
	C2 = space3d(NZ, NY, NX);
	//----------------------------------------------------------------------------------
	//创建模型赋值
	create_model(vp, vs, rhos, vf, rho, rhof, M, C, C1, C2, HH, H2u, mu, NZ, NY, NX);
	//----------------------------------------------------------------------------------
	//扩大模型，pml边界
	vs_ext = extmodel(vs, NZ, NY, NX, NP);
	vp_ext = extmodel(vp, NZ, NY, NX, NP);
	vf_ext = extmodel(vf, NZ, NY, NX, NP);
	rho_ext = extmodel(rho, NZ, NY, NX, NP);
	rhof_ext = extmodel(rhof, NZ, NY, NX, NP);
	rhos_ext = extmodel(rhos, NZ, NY, NX, NP);
	mu_ext = extmodel(mu, NZ, NY, NX, NP);
	HH_ext = extmodel(HH, NZ, NY, NX, NP);
	H2u_ext = extmodel(H2u, NZ, NY, NX, NP);
	C1_ext = extmodel(C1, NZ, NY, NX, NP);
	C2_ext = extmodel(C2, NZ, NY, NX, NP);
	C_ext = extmodel(C, NZ, NY, NX, NP);
	M_ext = extmodel(M, NZ, NY, NX, NP);
	//----------------------------------------------------------------------------------
	//应力和速度分量内存开辟
	float*** vux;
	float*** vuy;
	float*** vuz;
	float*** txx;
	float*** tyy;
	float*** tzz;
	float*** txz;
	float*** txy;
	float*** tyz;
	float*** vwx;
	float*** vwy;
	float*** vwz;
	float*** ss;
	vux = space3d(NZ_ext, NY_ext, NX_ext);
	vuy = space3d(NZ_ext, NY_ext, NX_ext);
	vuz = space3d(NZ_ext, NY_ext, NX_ext);
	txx = space3d(NZ_ext, NY_ext, NX_ext);
	tyy = space3d(NZ_ext, NY_ext, NX_ext);
	tzz = space3d(NZ_ext, NY_ext, NX_ext);
	txy = space3d(NZ_ext, NY_ext, NX_ext);
	txz = space3d(NZ_ext, NY_ext, NX_ext);
	tyz = space3d(NZ_ext, NY_ext, NX_ext);
	vwx = space3d(NZ_ext, NY_ext, NX_ext);
	vwy = space3d(NZ_ext, NY_ext, NX_ext);
	vwz = space3d(NZ_ext, NY_ext, NX_ext);
	ss = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx50 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx100 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx150 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx200 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx250 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** txx300 = space3d(NZ_ext, NY_ext, NX_ext);

	float*** vuz50 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** vwz50 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** vux50 = space3d(NZ_ext, NY_ext, NX_ext);
	float*** vwx50 = space3d(NZ_ext, NY_ext, NX_ext);
	//------------------------------------------------------------------------------------------------
	//定义密度，粘度系数分量
	float*** rho_tempx;
	float*** rho_tempy;
	float*** rho_tempz;
	float*** rhof_extx;
	float*** rhof_exty;
	float*** rhof_extz;

	float*** muxy, *** muxz, *** muyz;
	float*** C1x, *** C1y, *** C1z;
	float*** C2x, *** C2y, *** C2z;
	muxy = space3d(NZ_ext, NY_ext, NX_ext);
	muxz = space3d(NZ_ext, NY_ext, NX_ext);
	muyz = space3d(NZ_ext, NY_ext, NX_ext);
	rho_tempx = space3d(NZ_ext, NY_ext, NX_ext);
	rho_tempy = space3d(NZ_ext, NY_ext, NX_ext);
	rho_tempz = space3d(NZ_ext, NY_ext, NX_ext);
	rhof_extx = space3d(NZ_ext, NY_ext, NX_ext);
	rhof_exty = space3d(NZ_ext, NY_ext, NX_ext);
	rhof_extz = space3d(NZ_ext, NY_ext, NX_ext);

	C1x = space3d(NZ_ext, NY_ext, NX_ext);
	C1y = space3d(NZ_ext, NY_ext, NX_ext);
	C1z = space3d(NZ_ext, NY_ext, NX_ext);
	C2x = space3d(NZ_ext, NY_ext, NX_ext);
	C2y = space3d(NZ_ext, NY_ext, NX_ext);
	C2z = space3d(NZ_ext, NY_ext, NX_ext);
	//缩放因子
	float Cv = 1.0 * pow(10.0, 8);
	float Cvwp1 = 1.0 * pow(10.0, -6);


	int ix, iy, iz, it;
	for (iz = 1; iz < NZ_ext - 1; iz++)
	{
		for (iy = 1; iy < NY_ext - 1; iy++)
		{
			for (ix = 1; ix < NX_ext - 1; ix++)
			{
				//计算参数mu
				if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy][ix + 1] == 0 || mu_ext[iz + 1][iy][ix] == 0 || mu_ext[iz + 1][iy][ix + 1] == 0)
					muxz[iz][iy][ix] = 0.0;
				else
					muxz[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy][ix + 1] + 1 / mu_ext[iz + 1][iy][ix] + 1 / mu_ext[iz + 1][iy][ix + 1]));
				if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy + 1][ix] == 0 || mu_ext[iz + 1][iy][ix] == 0 || mu_ext[iz + 1][iy + 1][ix] == 0)
					muyz[iz][iy][ix] = 0.0;
				else
					muyz[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy + 1][ix] + 1 / mu_ext[iz + 1][iy][ix] + 1 / mu_ext[iz + 1][iy + 1][ix]));
				if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy][ix + 1] == 0 || mu_ext[iz][iy + 1][ix] == 0 || mu_ext[iz][iy + 1][ix + 1] == 0)
					muxy[iz][iy][ix] = 0.0;
				else
					muxy[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy][ix + 1] + 1 / mu_ext[iz][iy + 1][ix] + 1 / mu_ext[iz][iy + 1][ix + 1]));


				//计算参数
				rhof_extx[iz][iy][ix] = 0.5 * (rhof_ext[iz][iy][ix] + rhof_ext[iz][iy][ix + 1]);
				rhof_exty[iz][iy][ix] = 0.5 * (rhof_ext[iz][iy][ix] + rhof_ext[iz][iy + 1][ix]);
				rhof_extz[iz][iy][ix] = 0.5 * (rhof_ext[iz][iy][ix] + rhof_ext[iz + 1][iy][ix]);

				rho_tempx[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz][iy][ix + 1]);
				rho_tempy[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz][iy + 1][ix]);
				rho_tempz[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz + 1][iy][ix]);

				C1x[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz][iy][ix + 1]);
				C1y[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz][iy + 1][ix]);
				C1z[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz + 1][iy][ix]);

				C2x[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz][iy][ix + 1]);
				C2y[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz][iy + 1][ix]);
				C2z[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz + 1][iy][ix]);

			}
		}
	}
	// char rho_tempxname[] = "rho_tempx.dat";
	// wfile3d(rho_tempxname, rho_tempx, NZ_ext, NY_ext, NX_ext);
	// char rho_tempyname[] = "rho_tempy.dat";
	// wfile3d(rho_tempyname, rho_tempy, NZ_ext, NY_ext, NX_ext);
	// char rho_tempzname[] = "rho_tempz.dat";
	// wfile3d(rho_tempzname, rho_tempz, NZ_ext, NY_ext, NX_ext);
	// char muxzname[] = "muxz.dat";
	// wfile3d(muxzname, muxz, NZ_ext, NY_ext, NX_ext);
	// char muxyname[] = "muxy.dat";
	// wfile3d(muxyname, muxy, NZ_ext, NY_ext, NX_ext);
	// char muyzname[] = "muyz.dat";
	// wfile3d(muyzname, muyz, NZ_ext, NY_ext, NX_ext);
	// char C1xname[] = "C1x.dat";
	// wfile3d(C1xname, C1x, NZ_ext, NY_ext, NX_ext);
	// char C1yname[] = "C1y.dat";
	// wfile3d(C1yname, C1y, NZ_ext, NY_ext, NX_ext);
	// char C1zname[] = "C1z.dat";
	// wfile3d(C1zname, C1z, NZ_ext, NY_ext, NX_ext);
	// char C2xname[] = "C2x.dat";
	// wfile3d(C2xname, C2x, NZ_ext, NY_ext, NX_ext);
	// char C2yname[] = "C2y.dat";
	// wfile3d(C2yname, C2y, NZ_ext, NY_ext, NX_ext);
	// char C2zname[] = "C2z.dat";
	// wfile3d(C2zname, C2z, NZ_ext, NY_ext, NX_ext);
	// char cname[] = "C.dat";
	// wfile3d(cname, C_ext, NZ_ext, NY_ext, NX_ext);
	// char Mname[] = "M.dat";
	// wfile3d(Mname, M_ext, NZ_ext, NY_ext, NX_ext);
	// char C1name[] = "C1.dat";
	// wfile3d(C1name, C1_ext, NZ_ext, NY_ext, NX_ext);
	// char C2name[] = "C2.dat";
	// wfile3d(C2name, C2_ext, NZ_ext, NY_ext, NX_ext);
	// char HHname[] = "HH.dat";
	// wfile3d(HHname, HH_ext, NZ_ext, NY_ext, NX_ext);
	// char H2Uname[] = "H2U.dat";
	// wfile3d(H2Uname, H2u_ext, NZ_ext, NY_ext, NX_ext);
	//---------------------------------------------------------------------
	//pml吸收边界设置，开辟内存，赋值
	float*** dxi, *** dxi2;
	float*** dyj, *** dyj2;
	float*** dzk, *** dzk2;
	float*** e_dxi, *** e_dxi2;
	float*** e_dyj, *** e_dyj2;
	float*** e_dzk, *** e_dzk2;
	dxi = space3d(NZ_ext, NY_ext, NX_ext);
	dxi2 = space3d(NZ_ext, NY_ext, NX_ext);
	dyj = space3d(NZ_ext, NY_ext, NX_ext);
	dyj2 = space3d(NZ_ext, NY_ext, NX_ext);
	dzk = space3d(NZ_ext, NY_ext, NX_ext);
	dzk2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dxi = space3d(NZ_ext, NY_ext, NX_ext);
	e_dxi2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dyj = space3d(NZ_ext, NY_ext, NX_ext);
	e_dyj2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dzk = space3d(NZ_ext, NY_ext, NX_ext);
	e_dzk2 = space3d(NZ_ext, NY_ext, NX_ext);
	float tt, x, y, z, xoleft, xoright, yoleft, yoright, zoleft, zoright, d0, best_dt, v0;
	xoleft = DP;
	xoright = (NX_ext - 1) * H - DP;
	yoleft = DP;
	yoright = (NY_ext - 1) * H - DP;
	zoleft = DP;
	zoright = (NZ_ext - 1) * H - DP;
	// space3d 使用 malloc：物理区内若下面各 if 均未命中，系数必须为 d=0、e=1，否则未初始化值会拷到 GPU，PML 分裂项污染速度与应力
	for (iz = 0; iz < NZ_ext; iz++)
		for (iy = 0; iy < NY_ext; iy++)
			for (ix = 0; ix < NX_ext; ix++)
			{
				dxi[iz][iy][ix] = 0.0f; dxi2[iz][iy][ix] = 0.0f;
				dyj[iz][iy][ix] = 0.0f; dyj2[iz][iy][ix] = 0.0f;
				dzk[iz][iy][ix] = 0.0f; dzk2[iz][iy][ix] = 0.0f;
				e_dxi[iz][iy][ix] = 1.0f; e_dxi2[iz][iy][ix] = 1.0f;
				e_dyj[iz][iy][ix] = 1.0f; e_dyj2[iz][iy][ix] = 1.0f;
				e_dzk[iz][iy][ix] = 1.0f; e_dzk2[iz][iy][ix] = 1.0f;
			}
	//用于对vx_x[iz][ix]等的求解，加入吸收边界，使数值衰减
	for (iz = 0; iz < NZ_ext; iz++)
	{
		for (iy = 0; iy < NY_ext; iy++)
		{
			for (ix = 0; ix < NX_ext; ix++)
			{
				x = ix * H; y = iy * H; z = iz * H;
				if ((x >= 0 && x < xoleft) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
					dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
					e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
					e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
				}
				if ((x >= xoright && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
					dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
					e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
					e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
					dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
					e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
					e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= yoright && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
					dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
					e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
					e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < zoleft))
				{

					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
					dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
					e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
					e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= zoright && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
					dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
					e_dzk[iz][iy][ix] = exp((-dzk[iz][iy][ix]) * DT);
					e_dzk2[iz][iy][ix] = exp((-dzk2[iz][iy][ix]) * DT);
				}
			}
		}
	}
	// char dxiname[] = "dxi.dat";
	// wfile3d(dxiname, dxi, NZ_ext, NY_ext, NX_ext);
	// char dxi2name[] = "dxi2.dat";
	// wfile3d(dxi2name, dxi2, NZ_ext, NY_ext, NX_ext);
	// char dyjname[] = "dyj.dat";
	// wfile3d(dyjname, dyj, NZ_ext, NY_ext, NX_ext);
	// char dyj2name[] = "dyj2.dat";
	// wfile3d(dyj2name, dyj2, NZ_ext, NY_ext, NX_ext);
	// char dzkname[] = "dzk.dat";
	// wfile3d(dzkname, dzk, NZ_ext, NY_ext, NX_ext);
	// char dzk2name[] = "dzk2.dat";
	// wfile3d(dzk2name, dzk2, NZ_ext, NY_ext, NX_ext);
	//-----------------------------------------------并行计算-------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------
	//在主机端CPU定义参数，分配内存
	//地层参数
	float* h_rhof_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_HH_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_H2u_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_M_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C1x = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C1y = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C1z = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C2x = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C2y = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_C2z = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_rho_tempx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_rho_tempy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_rho_tempz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_muxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_muyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_muxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//速度应力
	float* h_vwx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vwy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vwz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_ss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_txx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_tyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_tzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_txz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_txy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_tyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//前一时刻的速度
	float* h_vwx2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vwy2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_vwz2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//pml内的差分值
	float* h_pmlxSxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlySxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzSxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxSxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlySyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzSyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxSxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlySyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzSzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlyVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlyVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlyVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxVwx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlyVwy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzVwz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlxss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlyss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_pmlzss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//前一时刻差分值
	float* h_SXxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SXxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SXxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SYxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SYyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SYyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SZxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SZyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SZzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SXss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SYss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_SZss = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuyx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuzy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vuzx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vwxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vwyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_Vwzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//pml参数
	float* h_dxi = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dyj = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dzk = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dxi2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dyj2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dzk2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dxi = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dyj = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dzk = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dxi2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dyj2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dzk2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	//---------------------------------------------------------------------------------
	//在主机端将三维参数转化为一维参数
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				h_rhof_ext[i + j * NX_ext + k * NX_ext * NY_ext] = rhof_ext[k][j][i];
				h_HH_ext[i + j * NX_ext + k * NX_ext * NY_ext] = HH_ext[k][j][i];
				h_H2u_ext[i + j * NX_ext + k * NX_ext * NY_ext] = H2u_ext[k][j][i];
				h_C_ext[i + j * NX_ext + k * NX_ext * NY_ext] = C_ext[k][j][i];
				h_M_ext[i + j * NX_ext + k * NX_ext * NY_ext] = M_ext[k][j][i];
				h_C1x[i + j * NX_ext + k * NX_ext * NY_ext] = C1x[k][j][i];
				h_C1y[i + j * NX_ext + k * NX_ext * NY_ext] = C1y[k][j][i];
				h_C1z[i + j * NX_ext + k * NX_ext * NY_ext] = C1z[k][j][i];
				h_C2x[i + j * NX_ext + k * NX_ext * NY_ext] = C2x[k][j][i];
				h_C2y[i + j * NX_ext + k * NX_ext * NY_ext] = C2y[k][j][i];
				h_C2z[i + j * NX_ext + k * NX_ext * NY_ext] = C2z[k][j][i];
				h_rho_tempx[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempx[k][j][i];
				h_rho_tempy[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempy[k][j][i];
				h_rho_tempz[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempz[k][j][i];
				h_muxz[i + j * NX_ext + k * NX_ext * NY_ext] = muxz[k][j][i];
				h_muxy[i + j * NX_ext + k * NX_ext * NY_ext] = muxy[k][j][i];
				h_muyz[i + j * NX_ext + k * NX_ext * NY_ext] = muyz[k][j][i];
				h_dxi[i + j * NX_ext + k * NX_ext * NY_ext] = dxi[k][j][i];
				h_dyj[i + j * NX_ext + k * NX_ext * NY_ext] = dyj[k][j][i];
				h_dzk[i + j * NX_ext + k * NX_ext * NY_ext] = dzk[k][j][i];
				h_dxi2[i + j * NX_ext + k * NX_ext * NY_ext] = dxi2[k][j][i];
				h_dyj2[i + j * NX_ext + k * NX_ext * NY_ext] = dyj2[k][j][i];
				h_dzk2[i + j * NX_ext + k * NX_ext * NY_ext] = dzk2[k][j][i];
				h_e_dxi[i + j * NX_ext + k * NX_ext * NY_ext] = e_dxi[k][j][i];
				h_e_dyj[i + j * NX_ext + k * NX_ext * NY_ext] = e_dyj[k][j][i];
				h_e_dzk[i + j * NX_ext + k * NX_ext * NY_ext] = e_dzk[k][j][i];
				h_e_dxi2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dxi2[k][j][i];
				h_e_dyj2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dyj2[k][j][i];
				h_e_dzk2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dzk2[k][j][i];
			}
		}
	}



	//------------------------------------------------------------------------------------------------------------
	//在设备端GPU定义参数，分配内存
	float* d_rho_tempx, * d_rho_tempy, * d_rho_tempz, * d_muxz, * d_muxy, * d_muyz, * d_rhof_ext, * d_HH_ext, * d_H2u_ext, * d_C_ext, * d_M_ext;	   //初始模型密度
	float* d_vux, * d_vuy, * d_vuz;
	float* d_vwx, * d_vwy, * d_vwz;
	float* d_vwx2, * d_vwy2, * d_vwz2;
	float* d_txx, * d_tyy, * d_tzz, * d_txy, * d_tyz, * d_txz, * d_ss;
	float* d_SXxx, * d_SXxy, * d_SXxz, * d_SYxy, * d_SYyy, * d_SYyz, * d_SZxz, * d_SZyz, * d_SZzz;
	float* d_Vuxx, * d_Vuyy, * d_Vuzz, * d_Vuxy, * d_Vuyx, * d_Vuxz, * d_Vuzx, * d_Vuyz, * d_Vuzy;
	float* d_pmlxSxx, * d_pmlySxy, * d_pmlzSxz, * d_pmlxSxy, * d_pmlySyy, * d_pmlzSyz, * d_pmlxSxz, * d_pmlySyz, * d_pmlzSzz;
	float* d_pmlxVux, * d_pmlyVuy, * d_pmlzVuz, * d_pmlxVuy, * d_pmlyVux, * d_pmlxVuz, * d_pmlzVux, * d_pmlyVuz, * d_pmlzVuy;
	float* d_pmlxVwx, * d_pmlyVwy, * d_pmlzVwz, * d_pmlxss, * d_pmlyss, * d_pmlzss;
	float* d_SXss, * d_SYss, * d_SZss, * d_Vwxx, * d_Vwyy, * d_Vwzz;
	float* d_dxi;
	float* d_dyj;
	float* d_dzk;
	float* d_dxi2;
	float* d_dyj2;
	float* d_dzk2;
	float* d_e_dxi;
	float* d_e_dyj;
	float* d_e_dzk;
	float* d_e_dxi2;
	float* d_e_dyj2;
	float* d_e_dzk2;
	float* d_C1x, * d_C1y, * d_C1z, * d_C2x, * d_C2y, * d_C2z;

	cudaMalloc(&d_rhof_ext, mem_size);
	cudaMalloc(&d_HH_ext, mem_size);
	cudaMalloc(&d_H2u_ext, mem_size);
	cudaMalloc(&d_C_ext, mem_size);
	cudaMalloc(&d_M_ext, mem_size);
	cudaMalloc(&d_C1x, mem_size);
	cudaMalloc(&d_C1y, mem_size);
	cudaMalloc(&d_C1z, mem_size);
	cudaMalloc(&d_C2x, mem_size);
	cudaMalloc(&d_C2y, mem_size);
	cudaMalloc(&d_C2z, mem_size);
	cudaMalloc(&d_rho_tempx, mem_size);
	cudaMalloc(&d_rho_tempy, mem_size);
	cudaMalloc(&d_rho_tempz, mem_size);
	cudaMalloc(&d_muxz, mem_size);
	cudaMalloc(&d_muyz, mem_size);
	cudaMalloc(&d_muxy, mem_size);
	cudaMalloc(&d_vux, mem_size);
	cudaMalloc(&d_vuy, mem_size);
	cudaMalloc(&d_vuz, mem_size);
	cudaMalloc(&d_txx, mem_size);
	cudaMalloc(&d_tyy, mem_size);
	cudaMalloc(&d_tzz, mem_size);
	cudaMalloc(&d_txz, mem_size);
	cudaMalloc(&d_txy, mem_size);
	cudaMalloc(&d_tyz, mem_size);
	cudaMalloc(&d_ss, mem_size);
	cudaMalloc(&d_vwx, mem_size);
	cudaMalloc(&d_vwy, mem_size);
	cudaMalloc(&d_vwz, mem_size);
	cudaMalloc(&d_vwx2, mem_size);
	cudaMalloc(&d_vwy2, mem_size);
	cudaMalloc(&d_vwz2, mem_size);
	cudaMalloc(&d_pmlxSxx, mem_size);
	cudaMalloc(&d_pmlySxy, mem_size);
	cudaMalloc(&d_pmlzSxz, mem_size);
	cudaMalloc(&d_pmlxSxy, mem_size);
	cudaMalloc(&d_pmlySyy, mem_size);
	cudaMalloc(&d_pmlzSyz, mem_size);
	cudaMalloc(&d_pmlxSxz, mem_size);
	cudaMalloc(&d_pmlySyz, mem_size);
	cudaMalloc(&d_pmlzSzz, mem_size);
	cudaMalloc(&d_pmlxVux, mem_size);
	cudaMalloc(&d_pmlyVuy, mem_size);
	cudaMalloc(&d_pmlzVuz, mem_size);
	cudaMalloc(&d_pmlxVuy, mem_size);
	cudaMalloc(&d_pmlyVux, mem_size);
	cudaMalloc(&d_pmlxVuz, mem_size);
	cudaMalloc(&d_pmlzVux, mem_size);
	cudaMalloc(&d_pmlyVuz, mem_size);
	cudaMalloc(&d_pmlzVuy, mem_size);
	cudaMalloc(&d_SXxx, mem_size);
	cudaMalloc(&d_SXxy, mem_size);
	cudaMalloc(&d_SXxz, mem_size);
	cudaMalloc(&d_SYxy, mem_size);
	cudaMalloc(&d_SYyy, mem_size);
	cudaMalloc(&d_SYyz, mem_size);
	cudaMalloc(&d_SZxz, mem_size);
	cudaMalloc(&d_SZyz, mem_size);
	cudaMalloc(&d_SZzz, mem_size);
	cudaMalloc(&d_Vuxx, mem_size);
	cudaMalloc(&d_Vuyy, mem_size);
	cudaMalloc(&d_Vuzz, mem_size);
	cudaMalloc(&d_Vuxy, mem_size);
	cudaMalloc(&d_Vuyx, mem_size);
	cudaMalloc(&d_Vuxz, mem_size);
	cudaMalloc(&d_Vuzx, mem_size);
	cudaMalloc(&d_Vuyz, mem_size);
	cudaMalloc(&d_Vuzy, mem_size);
	cudaMalloc(&d_pmlxVwx, mem_size);
	cudaMalloc(&d_pmlyVwy, mem_size);
	cudaMalloc(&d_pmlzVwz, mem_size);
	cudaMalloc(&d_pmlxss, mem_size);
	cudaMalloc(&d_pmlyss, mem_size);
	cudaMalloc(&d_pmlzss, mem_size);
	cudaMalloc(&d_SXss, mem_size);
	cudaMalloc(&d_SYss, mem_size);
	cudaMalloc(&d_SZss, mem_size);
	cudaMalloc(&d_Vwxx, mem_size);
	cudaMalloc(&d_Vwyy, mem_size);
	cudaMalloc(&d_Vwzz, mem_size);
	cudaMalloc(&d_dxi, mem_size);
	cudaMalloc(&d_dyj, mem_size);
	cudaMalloc(&d_dzk, mem_size);
	cudaMalloc(&d_dxi2, mem_size);
	cudaMalloc(&d_dyj2, mem_size);
	cudaMalloc(&d_dzk2, mem_size);
	cudaMalloc(&d_e_dxi, mem_size);
	cudaMalloc(&d_e_dyj, mem_size);
	cudaMalloc(&d_e_dzk, mem_size);
	cudaMalloc(&d_e_dxi2, mem_size);
	cudaMalloc(&d_e_dyj2, mem_size);
	cudaMalloc(&d_e_dzk2, mem_size);
	//---------------------------------------------------------------------------------------------
	//开始时间递推
	//这里的代码应该修正一下，应该从数据里面提取最大最小速度
	//检查稳定性条件
	best_dt = 6.0 * H / (7.0 * sqrt(2.0) * Vpmax);
	if (DT >= best_dt)
		printf("时间步长过大，应该小于 %f\n", best_dt);
	//控制网格频散
	if (Vsmin / (F0 * H) < 15)
		printf("空间步长太大,可能引起明显的网格频散\n");
	//--------------------------------------------------------------------------------------------
	//进行参数传递，将主机端CPU参数传递到设备端GPU
	cudaMemcpy(d_rho_tempx, h_rho_tempx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho_tempy, h_rho_tempy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho_tempz, h_rho_tempz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muxz, h_muxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muxy, h_muxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muyz, h_muyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vux, h_vux, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuy, h_vuy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuz, h_vuz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txx, h_txx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyy, h_tyy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tzz, h_tzz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txy, h_txy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txz, h_txz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyz, h_tyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxx, h_pmlxSxx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySxy, h_pmlySxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSxz, h_pmlzSxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxy, h_pmlxSxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyy, h_pmlySyy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSyz, h_pmlzSyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxz, h_pmlxSxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyz, h_pmlySyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSzz, h_pmlzSzz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVux, h_pmlxVux, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuy, h_pmlyVuy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuz, h_pmlzVuz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuy, h_pmlxVuy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVux, h_pmlyVux, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuz, h_pmlxVuz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVux, h_pmlzVux, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuz, h_pmlyVuz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuy, h_pmlzVuy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxx, h_SXxx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxy, h_SXxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxz, h_SXxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYxy, h_SYxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyy, h_SYyy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyz, h_SYyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZxz, h_SZxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZyz, h_SZyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZzz, h_SZzz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxx, h_Vuxx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyy, h_Vuyy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzz, h_Vuzz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxy, h_Vuxy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyx, h_Vuyx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxz, h_Vuxz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzx, h_Vuzx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyz, h_Vuyz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzy, h_Vuzy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxi, h_dxi, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj, h_dyj, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk, h_dzk, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxi2, h_dxi2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj2, h_dyj2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk2, h_dzk2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi, h_e_dxi, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj, h_e_dyj, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk, h_e_dzk, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi2, h_e_dxi2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj2, h_e_dyj2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk2, h_e_dzk2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVwx, h_pmlxVwx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVwy, h_pmlyVwy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVwz, h_pmlzVwz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxss, h_pmlxss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyss, h_pmlyss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzss, h_pmlzss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXss, h_SXss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYss, h_SYss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZss, h_SZss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwxx, h_Vwxx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwyy, h_Vwyy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwzz, h_Vwzz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ss, h_ss, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx, h_vwx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy, h_vwy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz, h_vwz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx2, h_vwx2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy2, h_vwy2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz2, h_vwz2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhof_ext, h_rhof_ext, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_HH_ext, h_HH_ext, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_H2u_ext, h_H2u_ext, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_ext, h_C_ext, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_ext, h_M_ext, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1x, h_C1x, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1y, h_C1y, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1z, h_C1z, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2x, h_C2x, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2y, h_C2y, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2z, h_C2z, mem_size, cudaMemcpyHostToDevice);
	//---------------------------------------------------------------------------------------------------------------------
	//在时间上进行迭代
	//for (it = 0; it < NT1; it++)
	for (it = 0; it < NT; it++)
	{
		float tt = it * DT;
		if (it % 100 == 0)
		{
			printf("it=%d\n", it);
		}
		if (it % MM == 0)
		{
			//加震源
			float I_sou;
			I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
			Source << <Gridsize, Blocksize >> > (d_txx, d_tyy, d_tzz, I_sou, sn, NX_ext, NY_ext, NZ_ext);
			cudaDeviceSynchronize();
		}
		//----------------------------------------------------------------------------------------------------
		////////计算速度
		////FD_V <BLOCK_X, BLOCK_Y> << <Gridsize1, Blocksize1 >> > (d_vux, d_vuy, d_vuz, d_rho_tempx, d_rho_tempy, d_rho_tempz, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_SXxx, d_SXxy, d_SXxz, d_SYxy, d_SYyy, d_SYyz, d_SZxz, d_SZyz, d_SZzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj, d_dyj, d_e_dyj2, d_dyj2, d_e_dzk, d_dzk, d_dzk2, d_e_dzk2,d_C1_ext,d_C2_ext,d_rhof_ext,d_ss,d_vwx, d_vwy, d_vwz, d_vwx2, d_vwy2, d_vwz2, d_SXss, d_SYss, d_SZss, d_pmlxss, d_pmlyss, d_pmlzss);
		FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_rho_tempx, d_rho_tempy, d_rho_tempz, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_SXxx, d_SXxy, d_SXxz, d_SYxy, d_SYyy, d_SYyz, d_SZxz, d_SZyz, d_SZzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj, d_dyj, d_e_dyj2, d_dyj2, d_e_dzk, d_dzk, d_dzk2, d_e_dzk2, d_rhof_ext, d_ss, d_vwx, d_vwy, d_vwz, d_vwx2, d_vwy2, d_vwz2, d_SXss, d_SYss, d_SZss, d_pmlxss, d_pmlyss, d_pmlzss, d_C1x, d_C1y, d_C1z, d_C2x, d_C2y, d_C2z, DT);
		cudaDeviceSynchronize();
		////----------------------------------------------------------------------------------------------------------
		////计算应力
		////FD_T  <BLOCK_X, BLOCK_Y> << <Gridsize1, Blocksize1 >> > (d_vux, d_vuy, d_vuz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_lamda2u_ext, d_lamda_ext, d_pmlxVux, d_muxy, d_muxz, d_muyz, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_Vuxx, d_Vuxy, d_Vuxz, d_Vuyx, d_Vuyy, d_Vuyz, d_Vuzx, d_Vuzy, d_Vuzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj2, d_dyj2, d_e_dyj, d_dyj, d_dzk2, d_e_dzk2, d_e_dzk, d_dzk, d_Vwxx, d_Vwyy, d_Vwzz, d_vwx, d_vwy, d_vwz, d_C_ext, d_M_ext, d_HH_ext, d_H2u_ext, d_ss, d_pmlxVwx, d_pmlyVwy, d_pmlzVwz);
		FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_pmlxVux, d_muxy, d_muxz, d_muyz, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_Vuxx, d_Vuxy, d_Vuxz, d_Vuyx, d_Vuyy, d_Vuyz, d_Vuzx, d_Vuzy, d_Vuzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj2, d_dyj2, d_e_dyj, d_dyj, d_dzk2, d_e_dzk2, d_e_dzk, d_dzk, d_Vwxx, d_Vwyy, d_Vwzz, d_vwx, d_vwy, d_vwz, d_C_ext, d_M_ext, d_HH_ext, d_H2u_ext, d_ss, d_pmlxVwx, d_pmlyVwy, d_pmlzVwz, DT);
		cudaDeviceSynchronize();
		//----------------------------------------------------------------------------------------------------------
		//将速度，应力从设备端拷贝到主机
		cudaMemcpy(h_txx, d_txx, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_tzz, d_tzz, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ss, d_ss, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vux, d_vux, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vwx, d_vwx, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vuz, d_vuz, mem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vwz, d_vwz, mem_size, cudaMemcpyDeviceToHost);

		//记录地震记录

		for (iz = 0; iz < NZ_ext; iz++)
		{
			sis_x[iz][it] = h_txx[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			sis_z[iz][it] = h_tzz[iz * NX_ext * NY_ext + sx + sy * NX_ext];

			sis_vu[iz][it] = h_vux[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			sis_vw[iz][it] = h_vwx[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			sis_p[iz][it] = h_ss[iz * NX_ext * NY_ext + sx + sy * NX_ext];
		}
		if (it == 500)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx50[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
						vux50[k][j][i] = h_vux[i + j * NX_ext + k * NX_ext * NY_ext];
						vwx50[k][j][i] = h_vwx[i + j * NX_ext + k * NX_ext * NY_ext];
						vuz50[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						vwz50[k][j][i] = h_vwz[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
		if (it == 1000)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx100[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
		if (it == 1500)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx150[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
		if (it == 2000)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx200[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
		if (it == 2500)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx250[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
		if (it == 3000)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx300[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
					}
				}
			}
	}
	//波场图参数设置
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				txx[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
			}
		}
	}
	finish = clock();
	printf("%f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);
	//--------------------------------------------------------------------------------------------------------------
	//输出波场快照
	char txxname[] = "txx.dat";
	printf("NZ_ext = %d NY_ext = %d NX_ext = %d\n", NZ_ext, NY_ext, NX_ext);
	printf("NZ_ext = %d  NT = %d\n", NZ_ext, NT);
	wfile3d(txxname, txx, NZ_ext, NY_ext, NX_ext);
	char txx50name[] = "txx500.dat";
	wfile3d(txx50name, txx50, NZ_ext, NY_ext, NX_ext);
	char txx100name[] = "txx1000.dat";
	wfile3d(txx100name, txx100, NZ_ext, NY_ext, NX_ext);
	char txx150name[] = "txx1500.dat";
	wfile3d(txx150name, txx150, NZ_ext, NY_ext, NX_ext);
	char txx200name[] = "txx2000.dat";
	wfile3d(txx200name, txx200, NZ_ext, NY_ext, NX_ext);
	char txx250name[] = "txx2500.dat";
	wfile3d(txx250name, txx250, NZ_ext, NY_ext, NX_ext);
	char txx300name[] = "txx3000.dat";
	wfile3d(txx300name, txx300, NZ_ext, NY_ext, NX_ext);

	char vux50name[] = "vux500.dat";
	wfile3d(vux50name, vux50, NZ_ext, NY_ext, NX_ext);
	char vwx50name[] = "vwx500.dat";
	wfile3d(vwx50name, vwx50, NZ_ext, NY_ext, NX_ext);
	char vuz50name[] = "vuz500.dat";
	wfile3d(vuz50name, vuz50, NZ_ext, NY_ext, NX_ext);
	char vwz50name[] = "vwz500.dat";
	wfile3d(vwz50name, vwz50, NZ_ext, NY_ext, NX_ext);
	//输出地震记录
	char siszname[] = "sisz.dat";
	char sisxname[] = "sisx.dat";
	char sisvuname[] = "sisvu.dat";
	char sisvwname[] = "sisvw.dat";
	char sispname[] = "sisp.dat";
	wfile(sisxname, sis_x, NZ_ext, NT);
	wfile(siszname, sis_z, NZ_ext, NT);
	wfile(sisvuname, sis_vu, NZ_ext, NT);
	wfile(sisvwname, sis_vw, NZ_ext, NT);
	wfile(sispname, sis_p, NZ_ext, NT);
	//---------------------------------------------------------------------------------------------------------------
	//主机端释放内存
	free(h_HH_ext);
	free(h_H2u_ext);
	free(h_C1x);
	free(h_C1y);
	free(h_C1z);
	free(h_C2x);
	free(h_C2y);
	free(h_C2z);
	free(h_C_ext);
	free(h_M_ext);
	free(h_dxi);
	free(h_dxi2);
	free(h_dyj);
	free(h_dyj2);
	free(h_dzk);
	free(h_dzk2);
	free(h_e_dxi);
	free(h_e_dxi2);
	free(h_e_dyj);
	free(h_e_dyj2);
	free(h_e_dzk);
	free(h_e_dzk2);
	free(h_rhof_ext);
	free(h_muxy);
	free(h_muxz);
	free(h_muyz);
	free(h_pmlxSxx);
	free(h_pmlxSxy);
	free(h_pmlxSxz);
	free(h_pmlxVux);
	free(h_pmlxVuy);
	free(h_pmlxVuz);
	free(h_pmlySxy);
	free(h_pmlySyy);
	free(h_pmlySyz);
	free(h_pmlyVux);
	free(h_pmlyVuy);
	free(h_pmlyVuz);
	free(h_pmlzSxz);
	free(h_pmlzSyz);
	free(h_pmlzSzz);
	free(h_pmlzVux);
	free(h_pmlzVuy);
	free(h_pmlzVuz);
	free(h_rho_tempx);
	free(h_rho_tempy);
	free(h_rho_tempz);
	free(h_SXxx);
	free(h_SXxy);
	free(h_SXxz);
	free(h_SYxy);
	free(h_SYyy);
	free(h_SYyz);
	free(h_SZxz);
	free(h_SZyz);
	free(h_SZzz);
	free(h_txx);
	free(h_tyy);
	free(h_tzz);
	free(h_txy);
	free(h_txz);
	free(h_tyz);
	free(h_vux);
	free(h_vuy);
	free(h_vuz);
	free(h_Vuxx);
	free(h_Vuxy);
	free(h_Vuxz);
	free(h_Vuyx);
	free(h_Vuyy);
	free(h_Vuyz);
	free(h_Vuzx);
	free(h_Vuzy);
	free(h_Vuzz);
	free_space3d(M, NZ, NY);
	free_space3d(C, NZ, NY);
	free_space3d(HH, NZ, NY);
	free_space3d(H2u, NZ, NY);
	free_space3d(C1, NZ, NY);
	free_space3d(C2, NZ, NY);
	free_space3d(M_ext, NZ_ext, NY_ext);
	free_space3d(C_ext, NZ_ext, NY_ext);
	free_space3d(HH_ext, NZ_ext, NY_ext);
	free_space3d(H2u_ext, NZ_ext, NY_ext);
	free_space3d(C1_ext, NZ_ext, NY_ext);
	free_space3d(C2_ext, NZ_ext, NY_ext);
	free_space3d(rhos_ext, NZ_ext, NY_ext);
	free_space3d(vwx, NZ_ext, NY_ext);
	free_space3d(vwy, NZ_ext, NY_ext);
	free_space3d(vwz, NZ_ext, NY_ext);
	free_space3d(ss, NZ_ext, NY_ext);
	free_space3d(C1x, NZ_ext, NY_ext);
	free_space3d(C1y, NZ_ext, NY_ext);
	free_space3d(C1z, NZ_ext, NY_ext);
	free_space3d(C2x, NZ_ext, NY_ext);
	free_space3d(C2y, NZ_ext, NY_ext);
	free_space3d(C2z, NZ_ext, NY_ext);
	free_space3d(rhof_extx, NZ_ext, NY_ext);
	free_space3d(rhof_exty, NZ_ext, NY_ext);
	free_space3d(rhof_extz, NZ_ext, NY_ext);
	free_space3d(rho, NZ, NY);
	free_space3d(rhof, NZ, NY);
	free_space3d(mu, NZ, NY);
	free_space3d(vs, NZ, NY);
	free_space3d(vp, NZ, NY);
	free_space3d(tzz, NZ_ext, NY_ext);
	free_space3d(tyy, NZ_ext, NY_ext);
	free_space3d(txx, NZ_ext, NY_ext);
	free_space3d(txy, NZ_ext, NY_ext);
	free_space3d(txz, NZ_ext, NY_ext);
	free_space3d(tyz, NZ_ext, NY_ext);
	free_space3d(vux, NZ_ext, NY_ext);
	free_space3d(vuy, NZ_ext, NY_ext);
	free_space3d(vuz, NZ_ext, NY_ext);

	free_space3d(vuz50, NZ_ext, NY_ext);
	free_space3d(vwz50, NZ_ext, NY_ext);
	free_space3d(vux50, NZ_ext, NY_ext);
	free_space3d(vwx50, NZ_ext, NY_ext);
	free_space3d(txx50, NZ_ext, NY_ext);
	free_space3d(txx100, NZ_ext, NY_ext);
	free_space3d(txx150, NZ_ext, NY_ext);
	free_space3d(txx200, NZ_ext, NY_ext);
	free_space3d(txx250, NZ_ext, NY_ext);
	free_space3d(txx300, NZ_ext, NY_ext);
	free_space3d(rho_tempx, NZ_ext, NY_ext);
	free_space3d(rho_tempy, NZ_ext, NY_ext);
	free_space3d(rho_tempz, NZ_ext, NY_ext);
	free_space3d(muxy, NZ_ext, NY_ext);
	free_space3d(muxz, NZ_ext, NY_ext);
	free_space3d(muyz, NZ_ext, NY_ext);
	free_space3d(vs_ext, NZ_ext, NY_ext);
	free_space3d(vp_ext, NZ_ext, NY_ext);
	free_space3d(vf_ext, NZ_ext, NY_ext);
	free_space3d(rho_ext, NZ_ext, NY_ext);
	free_space3d(rhof_ext, NZ_ext, NY_ext);
	free_space3d(mu_ext, NZ_ext, NY_ext);
	free_space3d(dxi, NZ_ext, NY_ext);
	free_space3d(e_dxi, NZ_ext, NY_ext);
	free_space3d(dxi2, NZ_ext, NY_ext);
	free_space3d(e_dxi2, NZ_ext, NY_ext);
	free_space3d(dyj, NZ_ext, NY_ext);
	free_space3d(e_dyj, NZ_ext, NY_ext);
	free_space3d(dyj2, NZ_ext, NY_ext);
	free_space3d(e_dyj2, NZ_ext, NY_ext);
	free_space3d(dzk, NZ_ext, NY_ext);
	free_space3d(e_dzk, NZ_ext, NY_ext);
	free_space3d(dzk2, NZ_ext, NY_ext);
	free_space3d(e_dzk2, NZ_ext, NY_ext);
	//设备端释放内存
	cudaFree(d_HH_ext);
	cudaFree(d_H2u_ext);
	cudaFree(d_C_ext);
	cudaFree(d_M_ext);
	cudaFree(d_M_ext);
	cudaFree(d_C1x);
	cudaFree(d_C1y);
	cudaFree(d_C1z);
	cudaFree(d_C2x);
	cudaFree(d_C2y);
	cudaFree(d_C2z);
	cudaFree(d_dxi);
	cudaFree(d_dxi2);
	cudaFree(d_dyj);
	cudaFree(d_dyj2);
	cudaFree(d_dzk);
	cudaFree(d_dzk2);
	cudaFree(d_e_dxi);
	cudaFree(d_e_dxi2);
	cudaFree(d_e_dyj);
	cudaFree(d_e_dyj2);
	cudaFree(d_e_dzk);
	cudaFree(d_e_dzk2);
	cudaFree(d_rhof_ext);
	cudaFree(d_muxy);
	cudaFree(d_muxz);
	cudaFree(d_muyz);
	cudaFree(d_pmlxSxx);
	cudaFree(d_pmlxSxy);
	cudaFree(d_pmlxSxz);
	cudaFree(d_pmlxVux);
	cudaFree(d_pmlxVuy);
	cudaFree(d_pmlxVuz);
	cudaFree(d_pmlySxy);
	cudaFree(d_pmlySyy);
	cudaFree(d_pmlySyz);
	cudaFree(d_pmlyVux);
	cudaFree(d_pmlyVuy);
	cudaFree(d_pmlyVuz);
	cudaFree(d_pmlzSxz);
	cudaFree(d_pmlzSyz);
	cudaFree(d_pmlzSzz);
	cudaFree(d_pmlzVux);
	cudaFree(d_pmlzVuy);
	cudaFree(d_pmlzVuz);
	cudaFree(d_rho_tempx);
	cudaFree(d_rho_tempy);
	cudaFree(d_rho_tempz);
	cudaFree(d_SXxx);
	cudaFree(d_SXxy);
	cudaFree(d_SXxz);
	cudaFree(d_SYxy);
	cudaFree(d_SYyy);
	cudaFree(d_SYyz);
	cudaFree(d_SZxz);
	cudaFree(d_SZyz);
	cudaFree(d_SZzz);
	cudaFree(d_txx);
	cudaFree(d_tyy);
	cudaFree(d_tzz);
	cudaFree(d_txy);
	cudaFree(d_txz);
	cudaFree(d_tyz);
	cudaFree(d_vux);
	cudaFree(d_vuy);
	cudaFree(d_vuz);
	cudaFree(d_ss);
	cudaFree(d_vwx);
	cudaFree(d_vwy);
	cudaFree(d_vwz);
	cudaFree(d_vwx2);
	cudaFree(d_vwy2);
	cudaFree(d_vwz2);
	cudaFree(d_Vuxx);
	cudaFree(d_Vuxy);
	cudaFree(d_Vuxz);
	cudaFree(d_Vuyx);
	cudaFree(d_Vuyy);
	cudaFree(d_Vuyz);
	cudaFree(d_Vuzx);
	cudaFree(d_Vuzy);
	cudaFree(d_Vuzz);
	cudaFree(d_pmlxVwx);
	cudaFree(d_pmlyVwy);
	cudaFree(d_pmlzVwz);
	cudaFree(d_pmlxss);
	cudaFree(d_pmlyss);
	cudaFree(d_pmlzss);
	cudaFree(d_SXss);
	cudaFree(d_SYss);
	cudaFree(d_SZss);
	cudaFree(d_Vwxx);
	cudaFree(d_Vwyy);
	cudaFree(d_Vwzz);
	free_space2d(sis_p, NZ_ext);
	free_space2d(sis_x, NZ_ext);
	free_space2d(sis_y, NZ_ext);
	free_space2d(sis_z, NZ_ext);
	free_space2d(sis_vu, NZ_ext);
	free_space2d(sis_vw, NZ_ext);
	printf("\Press any key to exit program...");
	return 0;
}

//申请二维动态数组
float** space2d(int nr, int nc)
{
	float** a;
	int i;
	a = (float**)calloc(nr, sizeof(float*));
	for (i = 0; i < nr; i++)
		a[i] = (float*)calloc(nc, sizeof(float));

	return a;
}

//释放二维动态数组
void free_space2d(float** a, int nr)
{
	int i;
	for (i = 0; i < nr; i++)
		free(a[i]);
	free(a);
}

//将二进制数据写入文件
void wfile(char filename[], float** data, int nr, int nc)
{
	int i, j;
	FILE* fp = fopen(filename, "wt");
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nc; j++)
		{
			fprintf(fp, "%e ", data[i][j]);
			if ((j + 1) % nc == 0)
				fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

//申请三维动态数组
float*** space3d(int nr, int ny, int nc)
{
	float*** a;
	int i, j, k;
	a = (float***)malloc(sizeof(float**) * nr);
	for (i = 0; i < nr; i++)
	{
		a[i] = (float**)malloc(sizeof(float*) * ny);
	}
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < ny; j++)
		{
			a[i][j] = (float*)malloc(sizeof(float) * nc);//sizeof(float) nc改为sizeof(float)*nc
		}
	}
	for (i = 0; i < nr; i++)
		for (j = 0; j < ny; j++)
			for (k = 0; k < nc; k++)
			{
				a[i][j][k] = 0.0f;
			}
	return a;
}

//释放三维动态数组
void free_space3d(float*** a, int nr, int ny)
{
	int i, j;
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < ny; j++)
		{
			free(a[i][j]);
		}
	}
	for (i = 0; i < nr; i++)
	{
		free(a[i]);
	}
	free(a);
}

//将二进制数据写入文件—三维
void wfile3d(char filename[], float*** data, int nr, int ny, int nc)
{
	int i, j, k;
	FILE* fp = fopen(filename, "wt");
	/*
	for (int i = 0; i<nr; i++)
	{
	fwrite(data[i], sizeof(float), nc, fp);
	}
	*/
	{
		for (i = 0; i < nr; i++)
		{
			for (k = 0; k < nc; k++)
			{
				j = ny / 2;//y轴切片
				fprintf(fp, "%e ", data[i][j][k]);
				if ((k + 1) % nc == 0)
					fprintf(fp, "\n");
			}
			fprintf(fp, "\n");
		}
	}
	//          fwrite(&data[i][j],1,sizeof(float),fp);
	fclose(fp);
}

void create_model(float*** vp, float*** vs, float*** rhos, float*** vf, float*** rho, float*** rhof, float*** M, float*** C, float*** C1,
	float*** C2, float*** HH, float*** H2u, float*** mu, int nr, int ny, int nc)
{
	//这里进行修改，最好是可视化设计或从文件读入
	int ix, iy, iz;
	double Ks1, Kb1, Kf1, a1, D1, tao, eta, porousm, perm, por;
	float vps, vss, rhoss; //固体颗粒的速度
	float acr = sqrt(3);
	vps = 6500; vss = 4000, rhoss = 3200;
	porousm = 8; eta = 1.0 * pow(10.0, -3);
	for (iz = 0; iz < nr; iz++)
	{
		for (iy = 0; iy < ny; iy++)
		{
			for (ix = 0; ix < nc; ix++)
			{
				if (((ix - nc / 2) * (ix - nc / 2) + (iy - ny / 2) * (iy - ny / 2)) <= 100)
					//竖直井孔%%%%%%\// if(iz<(nr-14-ix)||iz>=(nr+14-ix))//45度倾斜井孔//if (iz>= 50)//if(ix>=50)
				{
					vp[iz][iy][ix] = 1500.0f;
					vs[iz][iy][ix] = 0.0f;
					vf[iz][iy][ix] = 1500.0f;
					rhos[iz][iy][ix] = 1000.0f;
					rhof[iz][iy][ix] = 1000.0f;
					rho[iz][iy][ix] = 1000.0f;
					por = 1.0;
					mu[iz][iy][ix] = 0.0f;
					Kb1 = 0.0;//骨架压缩模量，声波测井原理与应用，P39
					Ks1 = rhof[iz][iy][ix] * vp[iz][iy][ix] * vp[iz][iy][ix];//岩石固态颗粒的体积模量
					Kf1 = rhof[iz][iy][ix] * vf[iz][iy][ix] * vf[iz][iy][ix];//孔隙流体的体积压缩模量
					C[iz][iy][ix] = Kf1;
					M[iz][iy][ix] = Kf1;
					HH[iz][iy][ix] = Kf1;
					H2u[iz][iy][ix] = Kf1;
					C1[iz][iy][ix] = 0.0f;
					C2[iz][iy][ix] = rhof[iz][iy][ix];
				}
				else
				{
					vp[iz][iy][ix] = 4000.0f;
					vs[iz][iy][ix] = 2700.0f;
					rhos[iz][iy][ix] = 2650.0f;
					rhof[iz][iy][ix] = 1000.0f;
					vf[iz][iy][ix] = 1500.0f;
					por = 0.1;
					rho[iz][iy][ix] = (1 - por) * rhos[iz][iy][ix] + por * rhof[iz][iy][ix]; //地层的密度
					perm = 2 * pow(10.0, -12);
					mu[iz][iy][ix] = (1 - por) * rhos[iz][iy][ix] * vs[iz][iy][ix] * vs[iz][iy][ix];
					Kb1 = rhos[iz][iy][ix] * (1 - por) * (vp[iz][iy][ix] * vp[iz][iy][ix] - vs[iz][iy][ix] * vs[iz][iy][ix] * 4.0 / 3.0);//干岩石体积弹性模量，声波测井原理与应用，P39
					Ks1 = rhos[iz][iy][ix] * (vp[iz][iy][ix] * vp[iz][iy][ix] - vs[iz][iy][ix] * vs[iz][iy][ix] * 4.0 / 3.0);//岩石固态颗粒的体积模量
					Kf1 = rhof[iz][iy][ix] * vf[iz][iy][ix] * vf[iz][iy][ix];//孔隙流体的体积压缩模量

					tao = 3.0;
					a1 = 1 - Kb1 / Ks1;
					M[iz][iy][ix] = Kf1 * Ks1 / (por * Ks1 + (a1 - por) * Kf1);
					HH[iz][iy][ix] = a1 * a1 * M[iz][iy][ix] + Kb1 + mu[iz][iy][ix] * 4.0 / 3.0;
					H2u[iz][iy][ix] = HH[iz][iy][ix] - 2 * mu[iz][iy][ix];
					C[iz][iy][ix] = M[iz][iy][ix] * a1;

					C1[iz][iy][ix] = eta / perm;
					C2[iz][iy][ix] = (1 + 2 / porousm) * tao * rhof[iz][iy][ix] / por;
				}
			}
		}
	}
}



//将模型扩边,用于PML
//具体的操作过程是将实际模型参数放置在扩边后的数据中央，四周的数据用
//最外缘的数据填充_3D
float*** extmodel(float*** init_model, int nz, int ny, int nx, int np)
{
	float*** p;
	int i, j, k;
	int nx2 = nx + 2 * np;
	int ny2 = ny + 2 * np;
	int nz2 = nz + 2 * np;
	p = space3d(nz2, ny2, nx2);


	for (i = 0; i < np; i++)
		for (k = 0; k < np; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[0][0][0];
	for (i = 0; i < np; i++)
		for (k = np; k < np + ny; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[0][k - np][0];
	for (i = 0; i < np; i++)
		for (k = np + ny; k < ny2; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[0][ny - 1][0];

	for (i = np; i < nz + np; i++)
		for (k = 0; k < np; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[i - np][0][0];
	for (i = np; i < nz + np; i++)
		for (k = np; k < np + ny; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[i - np][k - np][0];
	for (i = np; i < nz + np; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[i - np][ny - 1][0];

	for (i = nz + np; i < nz2; i++)
		for (k = 0; k < np; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[nz - 1][0][0];
	for (i = nz + np; i < nz2; i++)
		for (k = np; k < np + ny; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[nz - 1][k - np][0];
	for (i = nz + np; i < nz2; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = 0; j < np; j++)
				p[i][k][j] = init_model[nz - 1][ny - 1][0];



	for (i = 0; i < np; i++)
		for (k = 0; k < np; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[0][0][j - np];
	for (i = 0; i < np; i++)
		for (k = np; k < np + ny; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[0][k - np][j - np];
	for (i = 0; i < np; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[0][ny - 1][j - np];

	for (i = np; i < nz + np; i++)
		for (k = 0; k < np; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[i - np][0][j - np];
	for (i = np; i < nz + np; i++)
		for (k = np; k < np + ny; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[i - np][k - np][j - np];
	for (i = np; i < nz + np; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[i - np][ny - 1][j - np];

	for (i = nz + np; i < nz2; i++)
		for (k = 0; k < np; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[nz - 1][0][j - np];
	for (i = nz + np; i < nz2; i++)
		for (k = np; k < np + ny; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[nz - 1][k - np][j - np];
	for (i = nz + np; i < nz2; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np; j < np + nx; j++)
				p[i][k][j] = init_model[nz - 1][ny - 1][j - np];


	for (i = 0; i < np; i++)
		for (k = 0; k < np; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[0][0][nx - 1];
	for (i = 0; i < np; i++)
		for (k = np; k < np + ny; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[0][k - np][nx - 1];
	for (i = 0; i < np; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[0][ny - 1][nx - 1];

	for (i = np; i < nz + np; i++)
		for (k = 0; k < np; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[i - np][0][nx - 1];
	for (i = np; i < nz + np; i++)
		for (k = np; k < np + ny; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[i - np][k - np][nx - 1];
	for (i = np; i < nz + np; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[i - np][ny - 1][nx - 1];

	for (i = nz + np; i < nz2; i++)
		for (k = 0; k < np; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[nz - 1][0][nx - 1];
	for (i = nz + np; i < nz2; i++)
		for (k = np; k < np + ny; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[nz - 1][k - np][nx - 1];
	for (i = nz + np; i < nz2; i++)
		for (k = ny + np; k < ny2; k++)
			for (j = np + nx; j < nx2; j++)
				p[i][k][j] = init_model[nz - 1][ny - 1][nx - 1];


	return p;
}
