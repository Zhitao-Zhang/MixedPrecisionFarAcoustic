#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"

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

__device__ __forceinline__ float h2f(half v)
{
	return __half2float(v);
}

__device__ __forceinline__ void store_h(half* data, int offset, float value)
{
	data[offset] = __float2half(value);
}
//-------------------------------------------------------------------------------------------------------------------------------
//计算震源
__global__ void Source(half* txx, half* tyy, half* tzz, float I_sou, int sn, int NX_ext, int NY_ext, int NZ_ext)
{
	//加震源
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix >= NX_ext || iy >= NY_ext || iz >= NZ_ext)
		return;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	if (offset == sn)
	{
		store_h(txx, offset, h2f(txx[offset]) + I_sou);
		store_h(tyy, offset, h2f(tyy[offset]) + I_sou);
		store_h(tzz, offset, h2f(tzz[offset]) + I_sou);

	}
}
//------------------------------------------------------------------------------------------------------------------



__global__ void FD_V(half* vux, half* vuy, half* vuz, half* rho_inv_tildex, half* rho_inv_tildey, half* rho_inv_tildez,
	half* txx, half* tyy, half* tzz, half* txz, half* txy, half* tyz,
	half* pmlxSxx, half* pmlySxy, half* pmlzSxz, half* pmlxSxy, half* pmlySyy, half* pmlzSyz, half* pmlxSxz, half* pmlySyz, half* pmlzSzz,
	half* SXxx, half* SXxy, half* SXxz, half* SYxy, half* SYyy, half* SYyz, half* SZxz, half* SZyz, half* SZzz,
	float* e_dxi, float* dxi, float* e_dxi2, float* dxi2, float* e_dyj, float* dyj, float* e_dyj2, float* dyj2, float* e_dzk, float* dzk, float* dzk2, float* e_dzk2,
	half* ss, half* vwx, half* vwy, half* vwz, half* vwx2, half* vwy2, half* vwz2, half* SXss, half* SYss, half* SZss, half* pmlxss, half* pmlyss, half* pmlzss,
	half* Awx, half* Awy, half* Awz, half* Bpx, half* Bpy, half* Bpz, half* Btx, half* Bty, half* Btz, half* rhof_rho_invx, half* rhof_rho_invy, half* rhof_rho_invz, float DT)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float H = 0.01f;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	if (ix >= NX_ext || iy >= NY_ext || iz >= NZ_ext)
		return;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下
	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
		x1 = (h2f(txx[offset_r]) - h2f(txx[offset])) / H;
		x2 = (h2f(txy[offset]) - h2f(txy[offset_h])) / H;
		x3 = (h2f(txz[offset]) - h2f(txz[offset_b])) / H;
		s1 = (h2f(ss[offset_r]) - h2f(ss[offset])) / H;

		y1 = (h2f(tyy[offset_q]) - h2f(tyy[offset])) / H;
		y2 = (h2f(txy[offset]) - h2f(txy[offset_l])) / H;
		y3 = (h2f(tyz[offset]) - h2f(tyz[offset_b])) / H;
		s2 = (h2f(ss[offset_q]) - h2f(ss[offset])) / H;

		z1 = (h2f(tzz[offset_u]) - h2f(tzz[offset])) / H;
		z2 = (h2f(txz[offset]) - h2f(txz[offset_l])) / H;
		z3 = (h2f(tyz[offset]) - h2f(tyz[offset_h])) / H;
		s3 = (h2f(ss[offset_u]) - h2f(ss[offset])) / H;

		float pmlxSxx_new = h2f(pmlxSxx[offset]) * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5f) * (e_dxi2[offset] * h2f(SXxx[offset]) + x1);
		float pmlySxy_new = h2f(pmlySxy[offset]) * e_dyj[offset] + (-DT * dyj[offset] * 0.5f) * (e_dyj[offset] * h2f(SXxy[offset]) + x2);
		float pmlzSxz_new = h2f(pmlzSxz[offset]) * e_dzk[offset] + (-DT * dzk[offset] * 0.5f) * (e_dzk[offset] * h2f(SXxz[offset]) + x3);
		float pmlxss_new = h2f(pmlxss[offset]) * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5f) * (e_dxi2[offset] * h2f(SXss[offset]) + s1);
		store_h(pmlxSxx, offset, pmlxSxx_new);
		store_h(pmlySxy, offset, pmlySxy_new);
		store_h(pmlzSxz, offset, pmlzSxz_new);
		store_h(pmlxss, offset, pmlxss_new);
		store_h(SXxx, offset, x1); store_h(SXxy, offset, x2); store_h(SXxz, offset, x3); store_h(SXss, offset, s1);
		x1 = x1 + pmlxSxx_new;
		x2 = x2 + pmlySxy_new;
		x3 = x3 + pmlzSxz_new;
		s1 = s1 + pmlxss_new;

		float pmlxSxy_new = h2f(pmlxSxy[offset]) * e_dxi[offset] + (-DT * dxi[offset] * 0.5f) * (e_dxi[offset] * h2f(SYxy[offset]) + y2);
		float pmlySyy_new = h2f(pmlySyy[offset]) * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5f) * (e_dyj2[offset] * h2f(SYyy[offset]) + y1);
		float pmlzSyz_new = h2f(pmlzSyz[offset]) * e_dzk[offset] + (-DT * dzk[offset] * 0.5f) * (e_dzk[offset] * h2f(SYyz[offset]) + y3);
		float pmlyss_new = h2f(pmlyss[offset]) * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5f) * (e_dyj2[offset] * h2f(SYss[offset]) + s2);
		store_h(pmlxSxy, offset, pmlxSxy_new);
		store_h(pmlySyy, offset, pmlySyy_new);
		store_h(pmlzSyz, offset, pmlzSyz_new);
		store_h(pmlyss, offset, pmlyss_new);
		store_h(SYxy, offset, y2); store_h(SYyy, offset, y1); store_h(SYyz, offset, y3); store_h(SYss, offset, s2);
		y2 = y2 + pmlxSxy_new;
		y1 = y1 + pmlySyy_new;
		y3 = y3 + pmlzSyz_new;
		s2 = s2 + pmlyss_new;

		float pmlxSxz_new = h2f(pmlxSxz[offset]) * e_dxi[offset] + (-DT * dxi[offset] * 0.5f) * (e_dxi[offset] * h2f(SZxz[offset]) + z2);
		float pmlySyz_new = h2f(pmlySyz[offset]) * e_dyj[offset] + (-DT * dyj[offset] * 0.5f) * (e_dyj[offset] * h2f(SZyz[offset]) + z3);
		float pmlzSzz_new = h2f(pmlzSzz[offset]) * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5f) * (e_dzk2[offset] * h2f(SZzz[offset]) + z1);
		float pmlzss_new = h2f(pmlzss[offset]) * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5f) * (e_dzk2[offset] * h2f(SZss[offset]) + s3);
		store_h(pmlxSxz, offset, pmlxSxz_new);
		store_h(pmlySyz, offset, pmlySyz_new);
		store_h(pmlzSzz, offset, pmlzSzz_new);
		store_h(pmlzss, offset, pmlzss_new);
		store_h(SZxz, offset, z2); store_h(SZyz, offset, z3); store_h(SZzz, offset, z1); store_h(SZss, offset, s3);
		z2 = z2 + pmlxSxz_new;
		z3 = z3 + pmlySyz_new;
		z1 = z1 + pmlzSzz_new;
		s3 = s3 + pmlzss_new;

		float div_tau_x = x1 + x2 + x3;
		float div_tau_y = y1 + y2 + y3;
		float div_tau_z = z1 + z2 + z3;
		float vwx_old = h2f(vwx[offset]);
		float vwy_old = h2f(vwy[offset]);
		float vwz_old = h2f(vwz[offset]);
		float vwx_prev = h2f(vwx2[offset]);
		float vwy_prev = h2f(vwy2[offset]);
		float vwz_prev = h2f(vwz2[offset]);

		float awx = h2f(Awx[offset]), bpx = h2f(Bpx[offset]), btx = h2f(Btx[offset]);
		float awy = h2f(Awy[offset]), bpy = h2f(Bpy[offset]), bty = h2f(Bty[offset]);
		float awz = h2f(Awz[offset]), bpz = h2f(Bpz[offset]), btz = h2f(Btz[offset]);
		float vwx_new = (awx == 0.0f && bpx == 0.0f && btx == 0.0f) ? 0.0f : awx * vwx_old - btx * div_tau_x - bpx * s1;
		float vwy_new = (awy == 0.0f && bpy == 0.0f && bty == 0.0f) ? 0.0f : awy * vwy_old - bty * div_tau_y - bpy * s2;
		float vwz_new = (awz == 0.0f && bpz == 0.0f && btz == 0.0f) ? 0.0f : awz * vwz_old - btz * div_tau_z - bpz * s3;

		float vux_new = h2f(vux[offset]) + h2f(rho_inv_tildex[offset]) * div_tau_x - h2f(rhof_rho_invx[offset]) * (vwx_new - vwx_prev);
		float vuy_new = h2f(vuy[offset]) + h2f(rho_inv_tildey[offset]) * div_tau_y - h2f(rhof_rho_invy[offset]) * (vwy_new - vwy_prev);
		float vuz_new = h2f(vuz[offset]) + h2f(rho_inv_tildez[offset]) * div_tau_z - h2f(rhof_rho_invz[offset]) * (vwz_new - vwz_prev);

		store_h(vwx, offset, vwx_new); store_h(vwy, offset, vwy_new); store_h(vwz, offset, vwz_new);
		store_h(vux, offset, vux_new); store_h(vuy, offset, vuy_new); store_h(vuz, offset, vuz_new);
		store_h(vwx2, offset, vwx_new); store_h(vwy2, offset, vwy_new); store_h(vwz2, offset, vwz_new);
	}
}

__global__ void FD_T(half* vux, half* vuy, half* vuz, half* txx, half* tzz, half* tyy, half* txz, half* txy, half* tyz,
	half* pmlxVux, half* muxy_tilde, half* muxz_tilde, half* muyz_tilde,
	half* pmlyVuy, half* pmlzVuz, half* pmlxVuy, half* pmlyVux, half* pmlyVuz, half* pmlzVuy, half* pmlzVux, half* pmlxVuz,
	half* Vuxx, half* Vuxy, half* Vuxz, half* Vuyx, half* Vuyy, half* Vuyz, half* Vuzx, half* Vuzy, half* Vuzz,
	float* e_dxi, float* dxi, float* e_dxi2, float* dxi2, float* e_dyj2, float* dyj2, float* e_dyj, float* dyj, float* dzk2, float* e_dzk2, float* e_dzk, float* dzk,
	half* Vwxx, half* Vwyy, half* Vwzz, half* vwx, half* vwy, half* vwz, half* C_tilde, half* M_tilde, half* HH_tilde, half* H2u_tilde, half* ss, half* pmlxVwx, half* pmlyVwy, half* pmlzVwz, float DT)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	float H = 0.01f;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix <= 0 || iy <= 0 || iz <= 0 || ix >= (NX_ext - 1) || iy >= (NY_ext - 1) || iz >= (NZ_ext - 1))
		return;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下

	uxx = (h2f(vux[offset]) - h2f(vux[offset_l])) / H;
	uyy = (h2f(vuy[offset]) - h2f(vuy[offset_h])) / H;
	uzz = (h2f(vuz[offset]) - h2f(vuz[offset_b])) / H;

	wx = (h2f(vwx[offset]) - h2f(vwx[offset_l])) / H;
	wy = (h2f(vwy[offset]) - h2f(vwy[offset_h])) / H;
	wz = (h2f(vwz[offset]) - h2f(vwz[offset_b])) / H;

	uxy = (h2f(vux[offset_q]) - h2f(vux[offset])) / H;
	uyx = (h2f(vuy[offset_r]) - h2f(vuy[offset])) / H;

	uxz = (h2f(vux[offset_u]) - h2f(vux[offset])) / H;
	uzx = (h2f(vuz[offset_r]) - h2f(vuz[offset])) / H;

	uyz = (h2f(vuy[offset_u]) - h2f(vuy[offset])) / H;
	uzy = (h2f(vuz[offset_q]) - h2f(vuz[offset])) / H;


	float pmlxVux_new = h2f(pmlxVux[offset]) * e_dxi[offset] + (-DT * dxi[offset] * 0.5f) * (e_dxi[offset] * h2f(Vuxx[offset]) + uxx);
	float pmlyVuy_new = h2f(pmlyVuy[offset]) * e_dyj[offset] + (-DT * dyj[offset] * 0.5f) * (e_dyj[offset] * h2f(Vuyy[offset]) + uyy);
	float pmlzVuz_new = h2f(pmlzVuz[offset]) * e_dzk[offset] + (-DT * dzk[offset] * 0.5f) * (e_dzk[offset] * h2f(Vuzz[offset]) + uzz);
	store_h(pmlxVux, offset, pmlxVux_new);
	store_h(pmlyVuy, offset, pmlyVuy_new);
	store_h(pmlzVuz, offset, pmlzVuz_new);
	store_h(Vuxx, offset, uxx); store_h(Vuyy, offset, uyy); store_h(Vuzz, offset, uzz);
	uxx = uxx + pmlxVux_new;
	uyy = uyy + pmlyVuy_new;
	uzz = uzz + pmlzVuz_new;


	float pmlxVwx_new = h2f(pmlxVwx[offset]) * e_dxi[offset] + (-DT * dxi[offset] * 0.5f) * (e_dxi[offset] * h2f(Vwxx[offset]) + wx);
	float pmlyVwy_new = h2f(pmlyVwy[offset]) * e_dyj[offset] + (-DT * dyj[offset] * 0.5f) * (e_dyj[offset] * h2f(Vwyy[offset]) + wy);
	float pmlzVwz_new = h2f(pmlzVwz[offset]) * e_dzk[offset] + (-DT * dzk[offset] * 0.5f) * (e_dzk[offset] * h2f(Vwzz[offset]) + wz);
	store_h(pmlxVwx, offset, pmlxVwx_new);
	store_h(pmlyVwy, offset, pmlyVwy_new);
	store_h(pmlzVwz, offset, pmlzVwz_new);
	store_h(Vwxx, offset, wx); store_h(Vwyy, offset, wy); store_h(Vwzz, offset, wz);
	wx = wx + pmlxVwx_new;
	wy = wy + pmlyVwy_new;
	wz = wz + pmlzVwz_new;

	float pmlxVuy_new = h2f(pmlxVuy[offset]) * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5f) * (e_dyj2[offset] * h2f(Vuxy[offset]) + uxy);
	float pmlyVux_new = h2f(pmlyVux[offset]) * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5f) * (e_dxi2[offset] * h2f(Vuyx[offset]) + uyx);
	store_h(pmlxVuy, offset, pmlxVuy_new);
	store_h(pmlyVux, offset, pmlyVux_new);
	store_h(Vuxy, offset, uxy); store_h(Vuyx, offset, uyx);
	uxy = uxy + pmlxVuy_new;
	uyx = uyx + pmlyVux_new;

	float pmlxVuz_new = h2f(pmlxVuz[offset]) * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5f) * (e_dzk2[offset] * h2f(Vuxz[offset]) + uxz);
	float pmlzVux_new = h2f(pmlzVux[offset]) * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5f) * (e_dxi2[offset] * h2f(Vuzx[offset]) + uzx);
	store_h(pmlxVuz, offset, pmlxVuz_new);
	store_h(pmlzVux, offset, pmlzVux_new);
	store_h(Vuxz, offset, uxz); store_h(Vuzx, offset, uzx);
	uxz = uxz + pmlxVuz_new;
	uzx = uzx + pmlzVux_new;

	float pmlyVuz_new = h2f(pmlyVuz[offset]) * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5f) * (e_dzk2[offset] * h2f(Vuyz[offset]) + uyz);
	float pmlzVuy_new = h2f(pmlzVuy[offset]) * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5f) * (e_dyj2[offset] * h2f(Vuzy[offset]) + uzy);
	store_h(pmlyVuz, offset, pmlyVuz_new);
	store_h(pmlzVuy, offset, pmlzVuy_new);
	store_h(Vuzy, offset, uzy); store_h(Vuyz, offset, uyz);
	uyz = uyz + pmlyVuz_new;
	uzy = uzy + pmlzVuy_new;

	float div_u = uxx + uyy + uzz;
	float div_w = wx + wy + wz;
	store_h(ss, offset, h2f(ss[offset]) - h2f(C_tilde[offset]) * div_u - h2f(M_tilde[offset]) * div_w);
	store_h(txx, offset, h2f(txx[offset]) + h2f(H2u_tilde[offset]) * (uyy + uzz) + h2f(HH_tilde[offset]) * uxx + h2f(C_tilde[offset]) * div_w);
	store_h(tyy, offset, h2f(tyy[offset]) + h2f(H2u_tilde[offset]) * (uxx + uzz) + h2f(HH_tilde[offset]) * uyy + h2f(C_tilde[offset]) * div_w);
	store_h(tzz, offset, h2f(tzz[offset]) + h2f(H2u_tilde[offset]) * (uxx + uyy) + h2f(HH_tilde[offset]) * uzz + h2f(C_tilde[offset]) * div_w);
	store_h(txy, offset, h2f(txy[offset]) + h2f(muxy_tilde[offset]) * (uxy + uyx));
	store_h(tyz, offset, h2f(tyz[offset]) + h2f(muyz_tilde[offset]) * (uyz + uzy));
	store_h(txz, offset, h2f(txz[offset]) + h2f(muxz_tilde[offset]) * (uxz + uzx));
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
	const size_t elem_count = (size_t)NZ_ext * NY_ext * NX_ext;
	size_t mem_size = elem_count * sizeof(float);     //float内存大小
	size_t mem_size_half = elem_count * sizeof(half); //half内存大小
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
	//指数缩放因子和缩放后系数
	float Kmax = 0.0f;
	float Rmax = 0.0f;
	const float Q_EPS = 1.0e-12f;
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				Kmax = fmaxf(Kmax, fabsf(HH_ext[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(H2u_ext[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(C_ext[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(M_ext[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(mu_ext[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(muxy[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(muxz[k][j][i]));
				Kmax = fmaxf(Kmax, fabsf(muyz[k][j][i]));

				float rhof0 = rhof_ext[k][j][i];
				float rho_inv_x = rho_tempx[k][j][i];
				float rho_inv_y = rho_tempy[k][j][i];
				float rho_inv_z = rho_tempz[k][j][i];
				if (rho_inv_x > 0.0f) Rmax = fmaxf(Rmax, fabsf(rho_inv_x));
				if (rho_inv_y > 0.0f) Rmax = fmaxf(Rmax, fabsf(rho_inv_y));
				if (rho_inv_z > 0.0f) Rmax = fmaxf(Rmax, fabsf(rho_inv_z));

				float Qx = C2x[k][j][i] - rhof0 * rhof0 * rho_inv_x;
				float Qy = C2y[k][j][i] - rhof0 * rhof0 * rho_inv_y;
				float Qz = C2z[k][j][i] - rhof0 * rhof0 * rho_inv_z;
				if (C1x[k][j][i] != 0.0f && fabsf(Qx) > Q_EPS)
				{
					Rmax = fmaxf(Rmax, 1.0f / fabsf(Qx));
					Rmax = fmaxf(Rmax, fabsf(rhof0 * rho_inv_x / Qx));
				}
				if (C1y[k][j][i] != 0.0f && fabsf(Qy) > Q_EPS)
				{
					Rmax = fmaxf(Rmax, 1.0f / fabsf(Qy));
					Rmax = fmaxf(Rmax, fabsf(rhof0 * rho_inv_y / Qy));
				}
				if (C1z[k][j][i] != 0.0f && fabsf(Qz) > Q_EPS)
				{
					Rmax = fmaxf(Rmax, 1.0f / fabsf(Qz));
					Rmax = fmaxf(Rmax, fabsf(rhof0 * rho_inv_z / Qz));
				}
			}
		}
	}

	int em_int = -23;
	const int es_int = 0;
	if (Kmax > 0.0f && Rmax > 0.0f)
	{
		float er = 1.0f / sqrtf(DT * DT * Kmax * Rmax);
		em_int = (int)lrintf(-log2f(er * DT * Kmax));
	}
	float scale_m = ldexpf(1.0f, em_int);
	float inv_scale_m = ldexpf(1.0f, -em_int);
	float velocity_output_scale = ldexpf(1.0f, em_int - es_int);
	printf("mixed precision exponent scaling: em=%d es=%d scale_m=%e inv_scale_m=%e\n", em_int, es_int, scale_m, inv_scale_m);

	//在主机端CPU定义缩放后参数，分配内存
	half* h_rho_tempx = (half*)calloc(elem_count, sizeof(half));      // rho_inv_tildex
	half* h_rho_tempy = (half*)calloc(elem_count, sizeof(half));
	half* h_rho_tempz = (half*)calloc(elem_count, sizeof(half));
	half* h_rhof_rho_invx = (half*)calloc(elem_count, sizeof(half));
	half* h_rhof_rho_invy = (half*)calloc(elem_count, sizeof(half));
	half* h_rhof_rho_invz = (half*)calloc(elem_count, sizeof(half));
	half* h_C1x = (half*)calloc(elem_count, sizeof(half));            // Awx
	half* h_C1y = (half*)calloc(elem_count, sizeof(half));
	half* h_C1z = (half*)calloc(elem_count, sizeof(half));
	half* h_C2x = (half*)calloc(elem_count, sizeof(half));            // Bpx
	half* h_C2y = (half*)calloc(elem_count, sizeof(half));
	half* h_C2z = (half*)calloc(elem_count, sizeof(half));
	half* h_Btx = (half*)calloc(elem_count, sizeof(half));
	half* h_Bty = (half*)calloc(elem_count, sizeof(half));
	half* h_Btz = (half*)calloc(elem_count, sizeof(half));
	half* h_HH_ext = (half*)calloc(elem_count, sizeof(half));         // HH_tilde
	half* h_H2u_ext = (half*)calloc(elem_count, sizeof(half));
	half* h_C_ext = (half*)calloc(elem_count, sizeof(half));
	half* h_M_ext = (half*)calloc(elem_count, sizeof(half));
	half* h_muxy = (half*)calloc(elem_count, sizeof(half));
	half* h_muyz = (half*)calloc(elem_count, sizeof(half));
	half* h_muxz = (half*)calloc(elem_count, sizeof(half));
	//速度应力
	half* h_vwx = (half*)calloc(elem_count, sizeof(half));
	half* h_vwy = (half*)calloc(elem_count, sizeof(half));
	half* h_vwz = (half*)calloc(elem_count, sizeof(half));
	half* h_ss = (half*)calloc(elem_count, sizeof(half));
	half* h_vux = (half*)calloc(elem_count, sizeof(half));
	half* h_vuy = (half*)calloc(elem_count, sizeof(half));
	half* h_vuz = (half*)calloc(elem_count, sizeof(half));
	half* h_txx = (half*)calloc(elem_count, sizeof(half));
	half* h_tyy = (half*)calloc(elem_count, sizeof(half));
	half* h_tzz = (half*)calloc(elem_count, sizeof(half));
	half* h_txz = (half*)calloc(elem_count, sizeof(half));
	half* h_txy = (half*)calloc(elem_count, sizeof(half));
	half* h_tyz = (half*)calloc(elem_count, sizeof(half));
	//前一时刻的速度
	half* h_vwx2 = (half*)calloc(elem_count, sizeof(half));
	half* h_vwy2 = (half*)calloc(elem_count, sizeof(half));
	half* h_vwz2 = (half*)calloc(elem_count, sizeof(half));
	//pml内的差分值
	half* h_pmlxSxx = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlySxy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzSxz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxSxy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlySyy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzSyz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxSxz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlySyz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzSzz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxVux = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlyVuy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzVuz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxVuy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlyVux = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxVuz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzVux = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlyVuz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzVuy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxVwx = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlyVwy = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzVwz = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlxss = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlyss = (half*)calloc(elem_count, sizeof(half));
	half* h_pmlzss = (half*)calloc(elem_count, sizeof(half));
	//前一时刻差分值
	half* h_SXxx = (half*)calloc(elem_count, sizeof(half));
	half* h_SXxy = (half*)calloc(elem_count, sizeof(half));
	half* h_SXxz = (half*)calloc(elem_count, sizeof(half));
	half* h_SYxy = (half*)calloc(elem_count, sizeof(half));
	half* h_SYyy = (half*)calloc(elem_count, sizeof(half));
	half* h_SYyz = (half*)calloc(elem_count, sizeof(half));
	half* h_SZxz = (half*)calloc(elem_count, sizeof(half));
	half* h_SZyz = (half*)calloc(elem_count, sizeof(half));
	half* h_SZzz = (half*)calloc(elem_count, sizeof(half));
	half* h_SXss = (half*)calloc(elem_count, sizeof(half));
	half* h_SYss = (half*)calloc(elem_count, sizeof(half));
	half* h_SZss = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuxx = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuyy = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuzz = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuxy = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuyx = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuyz = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuzy = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuxz = (half*)calloc(elem_count, sizeof(half));
	half* h_Vuzx = (half*)calloc(elem_count, sizeof(half));
	half* h_Vwxx = (half*)calloc(elem_count, sizeof(half));
	half* h_Vwyy = (half*)calloc(elem_count, sizeof(half));
	half* h_Vwzz = (half*)calloc(elem_count, sizeof(half));
	//pml参数保留float，避免dxi/dyj/dzk超过FP16范围
	float* h_dxi = (float*)calloc(elem_count, sizeof(float));
	float* h_dyj = (float*)calloc(elem_count, sizeof(float));
	float* h_dzk = (float*)calloc(elem_count, sizeof(float));
	float* h_dxi2 = (float*)calloc(elem_count, sizeof(float));
	float* h_dyj2 = (float*)calloc(elem_count, sizeof(float));
	float* h_dzk2 = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dxi = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dyj = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dzk = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dxi2 = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dyj2 = (float*)calloc(elem_count, sizeof(float));
	float* h_e_dzk2 = (float*)calloc(elem_count, sizeof(float));

	//在主机端将三维参数转化为一维缩放参数
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				int idx = i + j * NX_ext + k * NX_ext * NY_ext;
				float rhof0 = rhof_ext[k][j][i];
				float rho_inv_x = rho_tempx[k][j][i];
				float rho_inv_y = rho_tempy[k][j][i];
				float rho_inv_z = rho_tempz[k][j][i];

				h_rho_tempx[idx] = __float2half(inv_scale_m * DT * rho_inv_x);
				h_rho_tempy[idx] = __float2half(inv_scale_m * DT * rho_inv_y);
				h_rho_tempz[idx] = __float2half(inv_scale_m * DT * rho_inv_z);
				h_rhof_rho_invx[idx] = __float2half(rhof0 * rho_inv_x);
				h_rhof_rho_invy[idx] = __float2half(rhof0 * rho_inv_y);
				h_rhof_rho_invz[idx] = __float2half(rhof0 * rho_inv_z);

				float Qx = C2x[k][j][i] - rhof0 * rhof0 * rho_inv_x;
				float Qy = C2y[k][j][i] - rhof0 * rhof0 * rho_inv_y;
				float Qz = C2z[k][j][i] - rhof0 * rhof0 * rho_inv_z;
				float denom_x = Qx + 0.5f * C1x[k][j][i] * DT;
				float denom_y = Qy + 0.5f * C1y[k][j][i] * DT;
				float denom_z = Qz + 0.5f * C1z[k][j][i] * DT;
				float awx = 0.0f, awy = 0.0f, awz = 0.0f;
				float bpx = 0.0f, bpy = 0.0f, bpz = 0.0f;
				float btx = 0.0f, bty = 0.0f, btz = 0.0f;
				if (C1x[k][j][i] != 0.0f && fabsf(Qx) > Q_EPS && fabsf(denom_x) > Q_EPS)
				{
					awx = (Qx - 0.5f * C1x[k][j][i] * DT) / denom_x;
					bpx = inv_scale_m * DT / denom_x;
					btx = inv_scale_m * DT * rhof0 * rho_inv_x / denom_x;
				}
				if (C1y[k][j][i] != 0.0f && fabsf(Qy) > Q_EPS && fabsf(denom_y) > Q_EPS)
				{
					awy = (Qy - 0.5f * C1y[k][j][i] * DT) / denom_y;
					bpy = inv_scale_m * DT / denom_y;
					bty = inv_scale_m * DT * rhof0 * rho_inv_y / denom_y;
				}
				if (C1z[k][j][i] != 0.0f && fabsf(Qz) > Q_EPS && fabsf(denom_z) > Q_EPS)
				{
					awz = (Qz - 0.5f * C1z[k][j][i] * DT) / denom_z;
					bpz = inv_scale_m * DT / denom_z;
					btz = inv_scale_m * DT * rhof0 * rho_inv_z / denom_z;
				}
				h_C1x[idx] = __float2half(awx);
				h_C1y[idx] = __float2half(awy);
				h_C1z[idx] = __float2half(awz);
				h_C2x[idx] = __float2half(bpx);
				h_C2y[idx] = __float2half(bpy);
				h_C2z[idx] = __float2half(bpz);
				h_Btx[idx] = __float2half(btx);
				h_Bty[idx] = __float2half(bty);
				h_Btz[idx] = __float2half(btz);

				h_HH_ext[idx] = __float2half(scale_m * DT * HH_ext[k][j][i]);
				h_H2u_ext[idx] = __float2half(scale_m * DT * H2u_ext[k][j][i]);
				h_C_ext[idx] = __float2half(scale_m * DT * C_ext[k][j][i]);
				h_M_ext[idx] = __float2half(scale_m * DT * M_ext[k][j][i]);
				h_muxz[idx] = __float2half(scale_m * DT * muxz[k][j][i]);
				h_muxy[idx] = __float2half(scale_m * DT * muxy[k][j][i]);
				h_muyz[idx] = __float2half(scale_m * DT * muyz[k][j][i]);

				h_dxi[idx] = dxi[k][j][i];
				h_dyj[idx] = dyj[k][j][i];
				h_dzk[idx] = dzk[k][j][i];
				h_dxi2[idx] = dxi2[k][j][i];
				h_dyj2[idx] = dyj2[k][j][i];
				h_dzk2[idx] = dzk2[k][j][i];
				h_e_dxi[idx] = e_dxi[k][j][i];
				h_e_dyj[idx] = e_dyj[k][j][i];
				h_e_dzk[idx] = e_dzk[k][j][i];
				h_e_dxi2[idx] = e_dxi2[k][j][i];
				h_e_dyj2[idx] = e_dyj2[k][j][i];
				h_e_dzk2[idx] = e_dzk2[k][j][i];
			}
		}
	}

	//------------------------------------------------------------------------------------------------------------
	//在设备端GPU定义参数，分配内存
	half* d_rho_tempx, * d_rho_tempy, * d_rho_tempz, * d_muxz, * d_muxy, * d_muyz, * d_HH_ext, * d_H2u_ext, * d_C_ext, * d_M_ext;
	half* d_rhof_rho_invx, * d_rhof_rho_invy, * d_rhof_rho_invz;
	half* d_Btx, * d_Bty, * d_Btz;
	half* d_vux, * d_vuy, * d_vuz;
	half* d_vwx, * d_vwy, * d_vwz;
	half* d_vwx2, * d_vwy2, * d_vwz2;
	half* d_txx, * d_tyy, * d_tzz, * d_txy, * d_tyz, * d_txz, * d_ss;
	half* d_SXxx, * d_SXxy, * d_SXxz, * d_SYxy, * d_SYyy, * d_SYyz, * d_SZxz, * d_SZyz, * d_SZzz;
	half* d_Vuxx, * d_Vuyy, * d_Vuzz, * d_Vuxy, * d_Vuyx, * d_Vuxz, * d_Vuzx, * d_Vuyz, * d_Vuzy;
	half* d_pmlxSxx, * d_pmlySxy, * d_pmlzSxz, * d_pmlxSxy, * d_pmlySyy, * d_pmlzSyz, * d_pmlxSxz, * d_pmlySyz, * d_pmlzSzz;
	half* d_pmlxVux, * d_pmlyVuy, * d_pmlzVuz, * d_pmlxVuy, * d_pmlyVux, * d_pmlxVuz, * d_pmlzVux, * d_pmlyVuz, * d_pmlzVuy;
	half* d_pmlxVwx, * d_pmlyVwy, * d_pmlzVwz, * d_pmlxss, * d_pmlyss, * d_pmlzss;
	half* d_SXss, * d_SYss, * d_SZss, * d_Vwxx, * d_Vwyy, * d_Vwzz;
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
	half* d_C1x, * d_C1y, * d_C1z, * d_C2x, * d_C2y, * d_C2z;

	cudaMalloc(&d_HH_ext, mem_size_half);
	cudaMalloc(&d_H2u_ext, mem_size_half);
	cudaMalloc(&d_C_ext, mem_size_half);
	cudaMalloc(&d_M_ext, mem_size_half);
	cudaMalloc(&d_C1x, mem_size_half);
	cudaMalloc(&d_C1y, mem_size_half);
	cudaMalloc(&d_C1z, mem_size_half);
	cudaMalloc(&d_C2x, mem_size_half);
	cudaMalloc(&d_C2y, mem_size_half);
	cudaMalloc(&d_C2z, mem_size_half);
	cudaMalloc(&d_Btx, mem_size_half);
	cudaMalloc(&d_Bty, mem_size_half);
	cudaMalloc(&d_Btz, mem_size_half);
	cudaMalloc(&d_rhof_rho_invx, mem_size_half);
	cudaMalloc(&d_rhof_rho_invy, mem_size_half);
	cudaMalloc(&d_rhof_rho_invz, mem_size_half);
	cudaMalloc(&d_rho_tempx, mem_size_half);
	cudaMalloc(&d_rho_tempy, mem_size_half);
	cudaMalloc(&d_rho_tempz, mem_size_half);
	cudaMalloc(&d_muxz, mem_size_half);
	cudaMalloc(&d_muyz, mem_size_half);
	cudaMalloc(&d_muxy, mem_size_half);
	cudaMalloc(&d_vux, mem_size_half);
	cudaMalloc(&d_vuy, mem_size_half);
	cudaMalloc(&d_vuz, mem_size_half);
	cudaMalloc(&d_txx, mem_size_half);
	cudaMalloc(&d_tyy, mem_size_half);
	cudaMalloc(&d_tzz, mem_size_half);
	cudaMalloc(&d_txz, mem_size_half);
	cudaMalloc(&d_txy, mem_size_half);
	cudaMalloc(&d_tyz, mem_size_half);
	cudaMalloc(&d_ss, mem_size_half);
	cudaMalloc(&d_vwx, mem_size_half);
	cudaMalloc(&d_vwy, mem_size_half);
	cudaMalloc(&d_vwz, mem_size_half);
	cudaMalloc(&d_vwx2, mem_size_half);
	cudaMalloc(&d_vwy2, mem_size_half);
	cudaMalloc(&d_vwz2, mem_size_half);
	cudaMalloc(&d_pmlxSxx, mem_size_half);
	cudaMalloc(&d_pmlySxy, mem_size_half);
	cudaMalloc(&d_pmlzSxz, mem_size_half);
	cudaMalloc(&d_pmlxSxy, mem_size_half);
	cudaMalloc(&d_pmlySyy, mem_size_half);
	cudaMalloc(&d_pmlzSyz, mem_size_half);
	cudaMalloc(&d_pmlxSxz, mem_size_half);
	cudaMalloc(&d_pmlySyz, mem_size_half);
	cudaMalloc(&d_pmlzSzz, mem_size_half);
	cudaMalloc(&d_pmlxVux, mem_size_half);
	cudaMalloc(&d_pmlyVuy, mem_size_half);
	cudaMalloc(&d_pmlzVuz, mem_size_half);
	cudaMalloc(&d_pmlxVuy, mem_size_half);
	cudaMalloc(&d_pmlyVux, mem_size_half);
	cudaMalloc(&d_pmlxVuz, mem_size_half);
	cudaMalloc(&d_pmlzVux, mem_size_half);
	cudaMalloc(&d_pmlyVuz, mem_size_half);
	cudaMalloc(&d_pmlzVuy, mem_size_half);
	cudaMalloc(&d_SXxx, mem_size_half);
	cudaMalloc(&d_SXxy, mem_size_half);
	cudaMalloc(&d_SXxz, mem_size_half);
	cudaMalloc(&d_SYxy, mem_size_half);
	cudaMalloc(&d_SYyy, mem_size_half);
	cudaMalloc(&d_SYyz, mem_size_half);
	cudaMalloc(&d_SZxz, mem_size_half);
	cudaMalloc(&d_SZyz, mem_size_half);
	cudaMalloc(&d_SZzz, mem_size_half);
	cudaMalloc(&d_Vuxx, mem_size_half);
	cudaMalloc(&d_Vuyy, mem_size_half);
	cudaMalloc(&d_Vuzz, mem_size_half);
	cudaMalloc(&d_Vuxy, mem_size_half);
	cudaMalloc(&d_Vuyx, mem_size_half);
	cudaMalloc(&d_Vuxz, mem_size_half);
	cudaMalloc(&d_Vuzx, mem_size_half);
	cudaMalloc(&d_Vuyz, mem_size_half);
	cudaMalloc(&d_Vuzy, mem_size_half);
	cudaMalloc(&d_pmlxVwx, mem_size_half);
	cudaMalloc(&d_pmlyVwy, mem_size_half);
	cudaMalloc(&d_pmlzVwz, mem_size_half);
	cudaMalloc(&d_pmlxss, mem_size_half);
	cudaMalloc(&d_pmlyss, mem_size_half);
	cudaMalloc(&d_pmlzss, mem_size_half);
	cudaMalloc(&d_SXss, mem_size_half);
	cudaMalloc(&d_SYss, mem_size_half);
	cudaMalloc(&d_SZss, mem_size_half);
	cudaMalloc(&d_Vwxx, mem_size_half);
	cudaMalloc(&d_Vwyy, mem_size_half);
	cudaMalloc(&d_Vwzz, mem_size_half);
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
	cudaMemcpy(d_rho_tempx, h_rho_tempx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho_tempy, h_rho_tempy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho_tempz, h_rho_tempz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhof_rho_invx, h_rhof_rho_invx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhof_rho_invy, h_rhof_rho_invy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhof_rho_invz, h_rhof_rho_invz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muxz, h_muxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muxy, h_muxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_muyz, h_muyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vux, h_vux, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuy, h_vuy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuz, h_vuz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txx, h_txx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyy, h_tyy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tzz, h_tzz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txy, h_txy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txz, h_txz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyz, h_tyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxx, h_pmlxSxx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySxy, h_pmlySxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSxz, h_pmlzSxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxy, h_pmlxSxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyy, h_pmlySyy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSyz, h_pmlzSyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxz, h_pmlxSxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyz, h_pmlySyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSzz, h_pmlzSzz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVux, h_pmlxVux, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuy, h_pmlyVuy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuz, h_pmlzVuz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuy, h_pmlxVuy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVux, h_pmlyVux, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuz, h_pmlxVuz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVux, h_pmlzVux, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuz, h_pmlyVuz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuy, h_pmlzVuy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxx, h_SXxx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxy, h_SXxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxz, h_SXxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYxy, h_SYxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyy, h_SYyy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyz, h_SYyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZxz, h_SZxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZyz, h_SZyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZzz, h_SZzz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxx, h_Vuxx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyy, h_Vuyy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzz, h_Vuzz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxy, h_Vuxy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyx, h_Vuyx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxz, h_Vuxz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzx, h_Vuzx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyz, h_Vuyz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzy, h_Vuzy, mem_size_half, cudaMemcpyHostToDevice);
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
	cudaMemcpy(d_pmlxVwx, h_pmlxVwx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVwy, h_pmlyVwy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVwz, h_pmlzVwz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxss, h_pmlxss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyss, h_pmlyss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzss, h_pmlzss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXss, h_SXss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYss, h_SYss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZss, h_SZss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwxx, h_Vwxx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwyy, h_Vwyy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwzz, h_Vwzz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ss, h_ss, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx, h_vwx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy, h_vwy, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz, h_vwz, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx2, h_vwx2, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy2, h_vwy2, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz2, h_vwz2, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_HH_ext, h_HH_ext, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_H2u_ext, h_H2u_ext, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_ext, h_C_ext, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_ext, h_M_ext, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1x, h_C1x, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1y, h_C1y, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1z, h_C1z, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2x, h_C2x, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2y, h_C2y, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2z, h_C2z, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Btx, h_Btx, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bty, h_Bty, mem_size_half, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Btz, h_Btz, mem_size_half, cudaMemcpyHostToDevice);
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
		FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_rho_tempx, d_rho_tempy, d_rho_tempz, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_SXxx, d_SXxy, d_SXxz, d_SYxy, d_SYyy, d_SYyz, d_SZxz, d_SZyz, d_SZzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj, d_dyj, d_e_dyj2, d_dyj2, d_e_dzk, d_dzk, d_dzk2, d_e_dzk2, d_ss, d_vwx, d_vwy, d_vwz, d_vwx2, d_vwy2, d_vwz2, d_SXss, d_SYss, d_SZss, d_pmlxss, d_pmlyss, d_pmlzss, d_C1x, d_C1y, d_C1z, d_C2x, d_C2y, d_C2z, d_Btx, d_Bty, d_Btz, d_rhof_rho_invx, d_rhof_rho_invy, d_rhof_rho_invz, DT);
		cudaDeviceSynchronize();
		////----------------------------------------------------------------------------------------------------------
		////计算应力
		FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_pmlxVux, d_muxy, d_muxz, d_muyz, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_Vuxx, d_Vuxy, d_Vuxz, d_Vuyx, d_Vuyy, d_Vuyz, d_Vuzx, d_Vuzy, d_Vuzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj2, d_dyj2, d_e_dyj, d_dyj, d_dzk2, d_e_dzk2, d_e_dzk, d_dzk, d_Vwxx, d_Vwyy, d_Vwzz, d_vwx, d_vwy, d_vwz, d_C_ext, d_M_ext, d_HH_ext, d_H2u_ext, d_ss, d_pmlxVwx, d_pmlyVwy, d_pmlzVwz, DT);
		cudaDeviceSynchronize();
		//----------------------------------------------------------------------------------------------------------
		//将速度，应力从设备端拷贝到主机
		cudaMemcpy(h_txx, d_txx, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_tzz, d_tzz, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ss, d_ss, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vux, d_vux, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vwx, d_vwx, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vuz, d_vuz, mem_size_half, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vwz, d_vwz, mem_size_half, cudaMemcpyDeviceToHost);

		//记录地震记录

		for (iz = 0; iz < NZ_ext; iz++)
		{
			int rec_idx = iz * NX_ext * NY_ext + sx + sy * NX_ext;
			sis_x[iz][it] = __half2float(h_txx[rec_idx]);
			sis_z[iz][it] = __half2float(h_tzz[rec_idx]);

			sis_vu[iz][it] = velocity_output_scale * __half2float(h_vux[rec_idx]);
			sis_vw[iz][it] = velocity_output_scale * __half2float(h_vwx[rec_idx]);
			sis_p[iz][it] = __half2float(h_ss[rec_idx]);
		}
		if (it == 500)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						int idx = i + j * NX_ext + k * NX_ext * NY_ext;
						txx50[k][j][i] = __half2float(h_txx[idx]);
						vux50[k][j][i] = velocity_output_scale * __half2float(h_vux[idx]);
						vwx50[k][j][i] = velocity_output_scale * __half2float(h_vwx[idx]);
						vuz50[k][j][i] = velocity_output_scale * __half2float(h_vuz[idx]);
						vwz50[k][j][i] = velocity_output_scale * __half2float(h_vwz[idx]);
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
						txx100[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
						txx150[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
						txx200[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
						txx250[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
						txx300[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
				txx[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
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
	free(h_Btx);
	free(h_Bty);
	free(h_Btz);
	free(h_rhof_rho_invx);
	free(h_rhof_rho_invy);
	free(h_rhof_rho_invz);
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
	free(h_ss);
	free(h_vux);
	free(h_vuy);
	free(h_vuz);
	free(h_vwx);
	free(h_vwy);
	free(h_vwz);
	free(h_vwx2);
	free(h_vwy2);
	free(h_vwz2);
	free(h_Vuxx);
	free(h_Vuxy);
	free(h_Vuxz);
	free(h_Vuyx);
	free(h_Vuyy);
	free(h_Vuyz);
	free(h_Vuzx);
	free(h_Vuzy);
	free(h_Vuzz);
	free(h_pmlxVwx);
	free(h_pmlyVwy);
	free(h_pmlzVwz);
	free(h_pmlxss);
	free(h_pmlyss);
	free(h_pmlzss);
	free(h_SXss);
	free(h_SYss);
	free(h_SZss);
	free(h_Vwxx);
	free(h_Vwyy);
	free(h_Vwzz);
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
	cudaFree(d_C1x);
	cudaFree(d_C1y);
	cudaFree(d_C1z);
	cudaFree(d_C2x);
	cudaFree(d_C2y);
	cudaFree(d_C2z);
	cudaFree(d_Btx);
	cudaFree(d_Bty);
	cudaFree(d_Btz);
	cudaFree(d_rhof_rho_invx);
	cudaFree(d_rhof_rho_invy);
	cudaFree(d_rhof_rho_invz);
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
	printf("Press any key to exit program...");
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
