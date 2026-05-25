#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_fp16.h"
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
__global__ void Source(half* txx, half* tyy, half* tzz, float I_sou, int sn, int NX_ext, int NY_ext, int NZ_ext)
{
	//加震源
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	if (offset == sn)
	{
		txx[offset] = __float2half(__half2float(txx[offset]) + I_sou);
		tyy[offset] = __float2half(__half2float(tyy[offset]) + I_sou);
		tzz[offset] = __float2half(__half2float(tzz[offset]) + I_sou);
	}
}


//------------------------------------------------------------------------------------------------------------------

//FD_V kernel - 使用半精度混合精度
//使用__half2float和__float2half进行显式转换，保持与main函数一致的缩放因子
__global__ void FD_V(half* vux, half* vuy, half* vuz,
	half* txx, half* tyy, half* tzz, half* txz, half* txy, half* tyz,
	half* pmlxSxx, half* pmlySxy, half* pmlzSxz, half* pmlxSxy, half* pmlySyy, half* pmlzSyz, half* pmlxSxz, half* pmlySyz, half* pmlzSzz,
	half* SXxx, half* SXxy, half* SXxz, half* SYxy, half* SYyy, half* SYyz, half* SZxz, half* SZyz, half* SZzz,
	half* e_dxi, half* dxi, half* e_dxi2, half* dxi2, half* e_dyj, half* dyj, half* e_dyj2, half* dyj2, half* e_dzk, half* dzk, half* dzk2, half* e_dzk2,
	half* ss, half* vwx, half* vwy, half* vwz, half* vwx2, half* vwy2, half* vwz2, half* SXss, half* SYss, half* SZss, half* pmlxss, half* pmlyss, half* pmlzss, float DT,
	half* VelocityWParameter1x, half* VelocityWParameter1y, half* VelocityWParameter1z, half* VelocityWParameter2x, half* VelocityWParameter2y, half* VelocityWParameter2z, half* VelocityWParameter3x, half* VelocityWParameter3y, half* VelocityWParameter3z,
	half* VelocityUParameter1x, half* VelocityUParameter1y, half* VelocityUParameter1z, half* VelocityUParameter2x, half* VelocityUParameter2y, half* VelocityUParameter2z)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float H = 100.0f;  // 差分系数
	//缩放因子 - 与main函数保持一致
	float Cvwp1 = 1.0f;
	float Cvwp2 = 1.0f;
	float Cvwp3 = 1.0f;
	float Cvup1 = 1.0f;
	float Cvup2 = 1.0f;
	float Cpmll = 100.0f;  // PML阻尼d参数缩放因子，与main中的Cpml=0.01配合

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;

	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
	//使用__half2float进行高精度转换
	x1 = (__half2float(txx[offset_r]) - __half2float(txx[offset])) * H;
	x2 = (__half2float(txy[offset]) - __half2float(txy[offset_h])) * H;
	x3 = (__half2float(txz[offset]) - __half2float(txz[offset_b])) * H;
	s1 = (__half2float(ss[offset_r]) - __half2float(ss[offset])) * H;

	y1 = (__half2float(tyy[offset_q]) - __half2float(tyy[offset])) * H;
	y2 = (__half2float(txy[offset]) - __half2float(txy[offset_l])) * H;
	y3 = (__half2float(tyz[offset]) - __half2float(tyz[offset_b])) * H;
	s2 = (__half2float(ss[offset_q]) - __half2float(ss[offset])) * H;

	z1 = (__half2float(tzz[offset_u]) - __half2float(tzz[offset])) * H;
	z2 = (__half2float(txz[offset]) - __half2float(txz[offset_l])) * H;
	z3 = (__half2float(tyz[offset]) - __half2float(tyz[offset_h])) * H;
	s3 = (__half2float(ss[offset_u]) - __half2float(ss[offset])) * H;

	//PML参数
	float local_dxi, local_dxi2, local_dyj, local_dyj2, local_dzk, local_dzk2;
	float local_e_dxi, local_e_dxi2, local_e_dyj, local_e_dyj2, local_e_dzk, local_e_dzk2;
	local_dxi = __half2float(dxi[offset]) * Cpmll;
	local_dxi2 = __half2float(dxi2[offset]) * Cpmll;
	local_dyj = __half2float(dyj[offset]) * Cpmll;
	local_dyj2 = __half2float(dyj2[offset]) * Cpmll;
	local_dzk = __half2float(dzk[offset]) * Cpmll;
	local_dzk2 = __half2float(dzk2[offset]) * Cpmll;
	// e_* = exp(-d*dt) 是无量纲衰减因子，物理区应为1，不能像d_*一样缩放/反缩放。
	local_e_dxi = __half2float(e_dxi[offset]);
	local_e_dxi2 = __half2float(e_dxi2[offset]);
	local_e_dyj = __half2float(e_dyj[offset]);
	local_e_dyj2 = __half2float(e_dyj2[offset]);
	local_e_dzk = __half2float(e_dzk[offset]);
	local_e_dzk2 = __half2float(e_dzk2[offset]);

	//PML计算 - 使用显式转换确保精度
	pmlxSxx[offset] = __float2half(__half2float(pmlxSxx[offset]) * local_e_dxi2 + (-DT * local_dxi2 * 0.5f) * (local_e_dxi2 * __half2float(SXxx[offset]) + x1));
	pmlySxy[offset] = __float2half(__half2float(pmlySxy[offset]) * local_e_dyj + (-DT * local_dyj * 0.5f) * (local_e_dyj * __half2float(SXxy[offset]) + x2));
	pmlzSxz[offset] = __float2half(__half2float(pmlzSxz[offset]) * local_e_dzk + (-DT * local_dzk * 0.5f) * (local_e_dzk * __half2float(SXxz[offset]) + x3));
	pmlxss[offset] = __float2half(__half2float(pmlxss[offset]) * local_e_dxi2 + (-DT * local_dxi2 * 0.5f) * (local_e_dxi2 * __half2float(SXss[offset]) + s1));
	SXxx[offset] = __float2half(x1); SXxy[offset] = __float2half(x2); SXxz[offset] = __float2half(x3); SXss[offset] = __float2half(s1);
	x1 = x1 + __half2float(pmlxSxx[offset]);
	x2 = x2 + __half2float(pmlySxy[offset]);
	x3 = x3 + __half2float(pmlzSxz[offset]);
	s1 = s1 + __half2float(pmlxss[offset]);

	pmlxSxy[offset] = __float2half(__half2float(pmlxSxy[offset]) * local_e_dxi + (-DT * local_dxi * 0.5f) * (local_e_dxi * __half2float(SYxy[offset]) + y2));
	pmlySyy[offset] = __float2half(__half2float(pmlySyy[offset]) * local_e_dyj2 + (-DT * local_dyj2 * 0.5f) * (local_e_dyj2 * __half2float(SYyy[offset]) + y1));
	pmlzSyz[offset] = __float2half(__half2float(pmlzSyz[offset]) * local_e_dzk + (-DT * local_dzk * 0.5f) * (local_e_dzk * __half2float(SYyz[offset]) + y3));
	pmlyss[offset] = __float2half(__half2float(pmlyss[offset]) * local_e_dyj2 + (-DT * local_dyj2 * 0.5f) * (local_e_dyj2 * __half2float(SYss[offset]) + s2));
	SYxy[offset] = __float2half(y2); SYyy[offset] = __float2half(y1); SYyz[offset] = __float2half(y3); SYss[offset] = __float2half(s2);
	y2 = y2 + __half2float(pmlxSxy[offset]);
	y1 = y1 + __half2float(pmlySyy[offset]);
	y3 = y3 + __half2float(pmlzSyz[offset]);
	s2 = s2 + __half2float(pmlyss[offset]);

	pmlxSxz[offset] = __float2half(__half2float(pmlxSxz[offset]) * local_e_dxi + (-DT * local_dxi * 0.5f) * (local_e_dxi * __half2float(SZxz[offset]) + z2));
	pmlySyz[offset] = __float2half(__half2float(pmlySyz[offset]) * local_e_dyj + (-DT * local_dyj * 0.5f) * (local_e_dyj * __half2float(SZyz[offset]) + z3));
	pmlzSzz[offset] = __float2half(__half2float(pmlzSzz[offset]) * local_e_dzk2 + (-DT * local_dzk2 * 0.5f) * (local_e_dzk2 * __half2float(SZzz[offset]) + z1));
	pmlzss[offset] = __float2half(__half2float(pmlzss[offset]) * local_e_dzk2 + (-DT * local_dzk2 * 0.5f) * (local_e_dzk2 * __half2float(SZss[offset]) + s3));
	SZxz[offset] = __float2half(z2); SZyz[offset] = __float2half(z3); SZzz[offset] = __float2half(z1); SZss[offset] = __float2half(s3);
	z2 = z2 + __half2float(pmlxSxz[offset]);
	z3 = z3 + __half2float(pmlySyz[offset]);
	z1 = z1 + __half2float(pmlzSzz[offset]);
	s3 = s3 + __half2float(pmlzss[offset]);

	//速度更新 - 使用显式转换
	vwx[offset] = __float2half(Cvwp1 * __half2float(VelocityWParameter1x[offset]) * __half2float(vwx[offset]) - Cvwp2 * __half2float(VelocityWParameter2x[offset]) * (x1 + x2 + x3) - Cvwp3 * __half2float(VelocityWParameter3x[offset]) * s1);
	vwy[offset] = __float2half(Cvwp1 * __half2float(VelocityWParameter1y[offset]) * __half2float(vwy[offset]) - Cvwp2 * __half2float(VelocityWParameter2y[offset]) * (y1 + y2 + y3) - Cvwp3 * __half2float(VelocityWParameter3y[offset]) * s2);
	vwz[offset] = __float2half(Cvwp1 * __half2float(VelocityWParameter1z[offset]) * __half2float(vwz[offset]) - Cvwp2 * __half2float(VelocityWParameter2z[offset]) * (z1 + z2 + z3) - Cvwp3 * __half2float(VelocityWParameter3z[offset]) * s3);

	vux[offset] = __float2half(__half2float(vux[offset]) + Cvup1 * __half2float(VelocityUParameter1x[offset]) * (x1 + x2 + x3) - Cvup2 * __half2float(VelocityUParameter2x[offset]) * (__half2float(vwx[offset]) - __half2float(vwx2[offset])));
	vuy[offset] = __float2half(__half2float(vuy[offset]) + Cvup1 * __half2float(VelocityUParameter1y[offset]) * (y1 + y2 + y3) - Cvup2 * __half2float(VelocityUParameter2y[offset]) * (__half2float(vwy[offset]) - __half2float(vwy2[offset])));
	vuz[offset] = __float2half(__half2float(vuz[offset]) + Cvup1 * __half2float(VelocityUParameter1z[offset]) * (z1 + z2 + z3) - Cvup2 * __half2float(VelocityUParameter2z[offset]) * (__half2float(vwz[offset]) - __half2float(vwz2[offset])));
	vwx2[offset] = vwx[offset]; vwy2[offset] = vwy[offset]; vwz2[offset] = vwz[offset];
	}
}



//FD_T kernel - 使用半精度混合精度
//使用__half2float和__float2half进行显式转换，保持与main函数一致的缩放因子
__global__ void FD_T(half* vux, half* vuy, half* vuz, half* txx, half* tzz, half* tyy, half* txz, half* txy, half* tyz,
	half* pmlxVux, half* pmlyVuy, half* pmlzVuz, half* pmlxVuy, half* pmlyVux, half* pmlyVuz, half* pmlzVuy, half* pmlzVux, half* pmlxVuz,
	half* Vuxx, half* Vuxy, half* Vuxz, half* Vuyx, half* Vuyy, half* Vuyz, half* Vuzx, half* Vuzy, half* Vuzz,
	half* e_dxi, half* dxi, half* e_dxi2, half* dxi2, half* e_dyj2, half* dyj2, half* e_dyj, half* dyj, half* dzk2, half* e_dzk2, half* e_dzk, half* dzk,
	half* Vwxx, half* Vwyy, half* Vwzz, half* vwx, half* vwy, half* vwz, half* ss, half* pmlxVwx, half* pmlyVwy, half* pmlzVwz, float DT,
	half* PressParameter1, half* PressParameter2, half* StressParameter1, half* StressParameter2, half* StressParameter3, half* StressParameterxy, half* StressParameterxz, half* StressParameteryz)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	float H = 100.0f;  // 差分系数
	//缩放因子 - 与main函数保持一致
	float Cpp1 = 0.01f;   // 对应Cpp1=100
	float Cpp2 = 0.01f;   // 对应Cpp2=100
	float Csp1 = 0.01f;   // 对应Csp1=100
	float Csp2 = 0.01f;   // 对应Csp2=100
	float Csp3 = 0.01f;   // 对应Csp3=100
	float Csp4 = 0.01f;   // 对应Csp4=100
	float Cpmll = 100.0f; // 仅用于PML阻尼d参数，对应main中的Cpml=0.01

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;

	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
	//使用__half2float进行高精度转换
	uxx = (__half2float(vux[offset]) - __half2float(vux[offset_l])) * H;
	uyy = (__half2float(vuy[offset]) - __half2float(vuy[offset_h])) * H;
	uzz = (__half2float(vuz[offset]) - __half2float(vuz[offset_b])) * H;

	wx = (__half2float(vwx[offset]) - __half2float(vwx[offset_l])) * H;
	wy = (__half2float(vwy[offset]) - __half2float(vwy[offset_h])) * H;
	wz = (__half2float(vwz[offset]) - __half2float(vwz[offset_b])) * H;

	uxy = (__half2float(vux[offset_q]) - __half2float(vux[offset])) * H;
	uyx = (__half2float(vuy[offset_r]) - __half2float(vuy[offset])) * H;

	uxz = (__half2float(vux[offset_u]) - __half2float(vux[offset])) * H;
	uzx = (__half2float(vuz[offset_r]) - __half2float(vuz[offset])) * H;

	uyz = (__half2float(vuy[offset_u]) - __half2float(vuy[offset])) * H;
	uzy = (__half2float(vuz[offset_q]) - __half2float(vuz[offset])) * H;

	//PML参数
	float local_dxi, local_dxi2, local_dyj, local_dyj2, local_dzk, local_dzk2;
	float local_e_dxi, local_e_dxi2, local_e_dyj, local_e_dyj2, local_e_dzk, local_e_dzk2;
	local_dxi = __half2float(dxi[offset]) * Cpmll;
	local_dxi2 = __half2float(dxi2[offset]) * Cpmll;
	local_dyj = __half2float(dyj[offset]) * Cpmll;
	local_dyj2 = __half2float(dyj2[offset]) * Cpmll;
	local_dzk = __half2float(dzk[offset]) * Cpmll;
	local_dzk2 = __half2float(dzk2[offset]) * Cpmll;
	// e_* = exp(-d*dt) 是无量纲衰减因子，物理区应为1，不能像d_*一样缩放/反缩放。
	local_e_dxi = __half2float(e_dxi[offset]);
	local_e_dxi2 = __half2float(e_dxi2[offset]);
	local_e_dyj = __half2float(e_dyj[offset]);
	local_e_dyj2 = __half2float(e_dyj2[offset]);
	local_e_dzk = __half2float(e_dzk[offset]);
	local_e_dzk2 = __half2float(e_dzk2[offset]);

	//PML计算 - 使用显式转换确保精度
	pmlxVux[offset] = __float2half(__half2float(pmlxVux[offset]) * local_e_dxi + (-DT * local_dxi * 0.5f) * (local_e_dxi * __half2float(Vuxx[offset]) + uxx));
	pmlyVuy[offset] = __float2half(__half2float(pmlyVuy[offset]) * local_e_dyj + (-DT * local_dyj * 0.5f) * (local_e_dyj * __half2float(Vuyy[offset]) + uyy));
	pmlzVuz[offset] = __float2half(__half2float(pmlzVuz[offset]) * local_e_dzk + (-DT * local_dzk * 0.5f) * (local_e_dzk * __half2float(Vuzz[offset]) + uzz));
	Vuxx[offset] = __float2half(uxx); Vuyy[offset] = __float2half(uyy); Vuzz[offset] = __float2half(uzz);
	uxx = uxx + __half2float(pmlxVux[offset]);
	uyy = uyy + __half2float(pmlyVuy[offset]);
	uzz = uzz + __half2float(pmlzVuz[offset]);

	pmlxVwx[offset] = __float2half(__half2float(pmlxVwx[offset]) * local_e_dxi + (-DT * local_dxi * 0.5f) * (local_e_dxi * __half2float(Vwxx[offset]) + wx));
	pmlyVwy[offset] = __float2half(__half2float(pmlyVwy[offset]) * local_e_dyj + (-DT * local_dyj * 0.5f) * (local_e_dyj * __half2float(Vwyy[offset]) + wy));
	pmlzVwz[offset] = __float2half(__half2float(pmlzVwz[offset]) * local_e_dzk + (-DT * local_dzk * 0.5f) * (local_e_dzk * __half2float(Vwzz[offset]) + wz));
	Vwxx[offset] = __float2half(wx); Vwyy[offset] = __float2half(wy); Vwzz[offset] = __float2half(wz);
	wx = wx + __half2float(pmlxVwx[offset]);
	wy = wy + __half2float(pmlyVwy[offset]);
	wz = wz + __half2float(pmlzVwz[offset]);

	pmlxVuy[offset] = __float2half(__half2float(pmlxVuy[offset]) * local_e_dyj2 + (-DT * local_dyj2 * 0.5f) * (local_e_dyj2 * __half2float(Vuxy[offset]) + uxy));
	pmlyVux[offset] = __float2half(__half2float(pmlyVux[offset]) * local_e_dxi2 + (-DT * local_dxi2 * 0.5f) * (local_e_dxi2 * __half2float(Vuyx[offset]) + uyx));
	Vuxy[offset] = __float2half(uxy); Vuyx[offset] = __float2half(uyx);
	uxy = uxy + __half2float(pmlxVuy[offset]);
	uyx = uyx + __half2float(pmlyVux[offset]);

	pmlxVuz[offset] = __float2half(__half2float(pmlxVuz[offset]) * local_e_dzk2 + (-DT * local_dzk2 * 0.5f) * (local_e_dzk2 * __half2float(Vuxz[offset]) + uxz));
	pmlzVux[offset] = __float2half(__half2float(pmlzVux[offset]) * local_e_dxi2 + (-DT * local_dxi2 * 0.5f) * (local_e_dxi2 * __half2float(Vuzx[offset]) + uzx));
	Vuxz[offset] = __float2half(uxz); Vuzx[offset] = __float2half(uzx);
	uxz = uxz + __half2float(pmlxVuz[offset]);
	uzx = uzx + __half2float(pmlzVux[offset]);

	pmlyVuz[offset] = __float2half(__half2float(pmlyVuz[offset]) * local_e_dzk2 + (-DT * local_dzk2 * 0.5f) * (local_e_dzk2 * __half2float(Vuyz[offset]) + uyz));
	pmlzVuy[offset] = __float2half(__half2float(pmlzVuy[offset]) * local_e_dyj2 + (-DT * local_dyj2 * 0.5f) * (local_e_dyj2 * __half2float(Vuzy[offset]) + uzy));
	Vuzy[offset] = __float2half(uzy); Vuyz[offset] = __float2half(uyz);
	uyz = uyz + __half2float(pmlyVuz[offset]);
	uzy = uzy + __half2float(pmlzVuy[offset]);

	//应力和压力更新 - 使用显式转换确保精度
	ss[offset] = __float2half(__half2float(ss[offset]) - Cpp1 * __half2float(PressParameter1[offset]) * (uxx + uyy + uzz) - Cpp2 * __half2float(PressParameter2[offset]) * (wx + wy + wz));
	txx[offset] = __float2half(__half2float(txx[offset]) + Csp1 * __half2float(StressParameter1[offset]) * (uyy + uzz) + Csp2 * __half2float(StressParameter2[offset]) * uxx + Csp3 * __half2float(StressParameter3[offset]) * (wx + wy + wz));
	tyy[offset] = __float2half(__half2float(tyy[offset]) + Csp1 * __half2float(StressParameter1[offset]) * (uxx + uzz) + Csp2 * __half2float(StressParameter2[offset]) * uyy + Csp3 * __half2float(StressParameter3[offset]) * (wx + wy + wz));
	tzz[offset] = __float2half(__half2float(tzz[offset]) + Csp1 * __half2float(StressParameter1[offset]) * (uxx + uyy) + Csp2 * __half2float(StressParameter2[offset]) * uzz + Csp3 * __half2float(StressParameter3[offset]) * (wx + wy + wz));
	txy[offset] = __float2half(__half2float(txy[offset]) + Csp4 * __half2float(StressParameterxy[offset]) * (uxy + uyx));
	tyz[offset] = __float2half(__half2float(tyz[offset]) + Csp4 * __half2float(StressParameteryz[offset]) * (uyz + uzy));
	txz[offset] = __float2half(__half2float(txz[offset]) + Csp4 * __half2float(StressParameterxz[offset]) * (uxz + uzx));
	}

}


//---------------------------------------------------------------------------------------------------------------
//主函数
int main()
{
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
	size_t mem_sizeHalf = NZ_ext * NY_ext * NX_ext * sizeof(half);     //half内存大小
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
	//应力和速度分量内存开辟 - 使用float存储以减少精度损失
	//关键修改：波场变量使用float存储，只将参数转换为half
	//这样可以显著减少half<->float转换带来的累积误差
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
	float*** VelocityWParameter1x;
	float*** VelocityWParameter1y;
	float*** VelocityWParameter1z;
	float*** VelocityWParameter2x;
	float*** VelocityWParameter2y;
	float*** VelocityWParameter2z;
	float*** VelocityWParameter3x;
	float*** VelocityWParameter3y;
	float*** VelocityWParameter3z;
	float*** VelocityUParameter1x;
	float*** VelocityUParameter1y;
	float*** VelocityUParameter1z;
	float*** VelocityUParameter2x;
	float*** VelocityUParameter2y;
	float*** VelocityUParameter2z;
	float*** PressParameter1;
	float*** PressParameter2;
	float*** StressParameter1;
	float*** StressParameter2;
	float*** StressParameter3;
	float*** StressParameterxy;
	float*** StressParameterxz;
	float*** StressParameteryz;
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
	VelocityWParameter1x = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter1y = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter1z = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter2x = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter2y = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter2z = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter3x = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter3y = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityWParameter3z = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter1x = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter1y = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter1z = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter2x = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter2y = space3d(NZ_ext, NY_ext, NX_ext);
	VelocityUParameter2z = space3d(NZ_ext, NY_ext, NX_ext);
	PressParameter1 = space3d(NZ_ext, NY_ext, NX_ext);
	PressParameter2 = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameter1 = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameter2 = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameter3 = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameterxy = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameterxz = space3d(NZ_ext, NY_ext, NX_ext);
	StressParameteryz = space3d(NZ_ext, NY_ext, NX_ext);
	C1x = space3d(NZ_ext, NY_ext, NX_ext);
	C1y = space3d(NZ_ext, NY_ext, NX_ext);
	C1z = space3d(NZ_ext, NY_ext, NX_ext);
	C2x = space3d(NZ_ext, NY_ext, NX_ext);
	C2y = space3d(NZ_ext, NY_ext, NX_ext);
	C2z = space3d(NZ_ext, NY_ext, NX_ext);
	//缩放因子 - 恢复原始设置，根据论文方法设计
	//根据原始代码分析，这些值用于将参数缩放到FP16的合理范围(0.01-100)
	float Cv = 1.0 * pow(10.0, 8);  // 速度项的整体缩放因子
	float Cp = 1.0;                  // 应力项的整体缩放因子
	float Cvwp1 = 1.0;
	float Cvwp2 = 1.0;
	float Cvwp3 = 1.0;
	float Cvup1 = 1.0;
	float Cvup2 = 1.0;
	float Cpp1 = 100.0;   // 压力项缩放，保持100使参数在合理范围
	float Cpp2 = 100.0;
	float Csp1 = 100.0;   // 应力项缩放，保持100
	float Csp2 = 100.0;
	float Csp3 = 100.0;
	float Csp4 = 100.0;
	float zero = 0.0;
	float Cpml = 0.01;    // 仅缩放PML阻尼d参数；e=exp(-d*dt)不缩放，保持物理区为1

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

				// 交错网格速度点使用密度倒数的界面平均，保持与Float版本一致。
				rho_tempx[iz][iy][ix] = 2.0f / (rho_ext[iz][iy][ix] + rho_ext[iz][iy][ix + 1]);
				rho_tempy[iz][iy][ix] = 2.0f / (rho_ext[iz][iy][ix] + rho_ext[iz][iy + 1][ix]);
				rho_tempz[iz][iy][ix] = 2.0f / (rho_ext[iz][iy][ix] + rho_ext[iz + 1][iy][ix]);

				C1x[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz][iy][ix + 1]);
				C1y[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz][iy + 1][ix]);
				C1z[iz][iy][ix] = 0.5 * (C1_ext[iz][iy][ix] + C1_ext[iz + 1][iy][ix]);

				C2x[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz][iy][ix + 1]);
				C2y[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz][iy + 1][ix]);
				C2z[iz][iy][ix] = 0.5 * (C2_ext[iz][iy][ix] + C2_ext[iz + 1][iy][ix]);

			}
		}
	}

	for (iz = 1; iz < NZ_ext - 1; iz++)
	{
		for (iy = 1; iy < NY_ext - 1; iy++)
		{
			for (ix = 1; ix < NX_ext - 1; ix++)
			{
				//计算流体速度方程中的参数（与 kernel2.cu FD_V：各向仅当 C1*==0 时该向系数置零）
				if (C1x[iz][iy][ix] == 0)
				{
					VelocityWParameter1x[iz][iy][ix] = 0;
					VelocityWParameter2x[iz][iy][ix] = 0;
					VelocityWParameter3x[iz][iy][ix] = 0;
				}
				else
				{
					VelocityWParameter1x[iz][iy][ix] = Cvwp1 * (C2x[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempx[iz][iy][ix]) - 0.5 * DT * C1x[iz][iy][ix]) / (C2x[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempx[iz][iy][ix]) + 0.5 * DT * C1x[iz][iy][ix]);
					VelocityWParameter2x[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_ext[iz][iy][ix] / rho_tempx[iz][iy][ix]) / (Cp * (C2x[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempx[iz][iy][ix]) + 0.5 * DT * C1x[iz][iy][ix]));
					VelocityWParameter3x[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2x[iz][iy][ix] - rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempx[iz][iy][ix] + 0.5 * DT * C1x[iz][iy][ix]));
				}

				if (C1y[iz][iy][ix] == 0)
				{
					VelocityWParameter1y[iz][iy][ix] = 0;
					VelocityWParameter2y[iz][iy][ix] = 0;
					VelocityWParameter3y[iz][iy][ix] = 0;
				}
				else
				{
					VelocityWParameter1y[iz][iy][ix] = Cvwp1 * (C2y[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempy[iz][iy][ix]) - 0.5 * DT * C1y[iz][iy][ix]) / (C2y[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempy[iz][iy][ix]) + 0.5 * DT * C1y[iz][iy][ix]);
					VelocityWParameter2y[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_ext[iz][iy][ix] / rho_tempy[iz][iy][ix]) / (Cp * (C2y[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempy[iz][iy][ix]) + 0.5 * DT * C1y[iz][iy][ix]));
					VelocityWParameter3y[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2y[iz][iy][ix] - rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempy[iz][iy][ix] + 0.5 * DT * C1y[iz][iy][ix]));
				}

				if (C1z[iz][iy][ix] == 0)
				{
					VelocityWParameter1z[iz][iy][ix] = 0;
					VelocityWParameter2z[iz][iy][ix] = 0;
					VelocityWParameter3z[iz][iy][ix] = 0;
				}
				else
				{
					VelocityWParameter1z[iz][iy][ix] = Cvwp1 * (C2z[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempz[iz][iy][ix]) - 0.5 * DT * C1z[iz][iy][ix]) / (C2z[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempz[iz][iy][ix]) + 0.5 * DT * C1z[iz][iy][ix]);
					VelocityWParameter2z[iz][iy][ix] = Cv * DT * Cvwp2 * (rhof_ext[iz][iy][ix] / rho_tempz[iz][iy][ix]) / (Cp * (C2z[iz][iy][ix] - (rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempz[iz][iy][ix]) + 0.5 * DT * C1z[iz][iy][ix]));
					VelocityWParameter3z[iz][iy][ix] = Cv * DT * Cvwp3 / (Cp * (C2z[iz][iy][ix] - rhof_ext[iz][iy][ix] * rhof_ext[iz][iy][ix] / rho_tempz[iz][iy][ix] + 0.5 * DT * C1z[iz][iy][ix]));
				}

				//计算固体速度方程中的参数
				VelocityUParameter1x[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempx[iz][iy][ix] * Cp);
				VelocityUParameter1y[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempy[iz][iy][ix] * Cp);
				VelocityUParameter1z[iz][iy][ix] = Cv * DT * Cvup1 / (rho_tempz[iz][iy][ix] * Cp);

				VelocityUParameter2x[iz][iy][ix] = rhof_ext[iz][iy][ix] * Cvup2 / (rho_tempx[iz][iy][ix]);
				VelocityUParameter2y[iz][iy][ix] = rhof_ext[iz][iy][ix] * Cvup2 / (rho_tempy[iz][iy][ix]);
				VelocityUParameter2z[iz][iy][ix] = rhof_ext[iz][iy][ix] * Cvup2 / (rho_tempz[iz][iy][ix]);

				//计算压力方程中的参数
				PressParameter1[iz][iy][ix] = C_ext[iz][iy][ix] * DT * Cp * Cpp1 / Cv;
				PressParameter2[iz][iy][ix] = M_ext[iz][iy][ix] * DT * Cp * Cpp2 / Cv;

				//计算正应力方程中的参数
				StressParameter1[iz][iy][ix] = Cp * DT * H2u_ext[iz][iy][ix] * Csp1 / Cv;
				StressParameter2[iz][iy][ix] = Cp * DT * HH_ext[iz][iy][ix] * Csp2 / Cv;
				StressParameter3[iz][iy][ix] = Cp * DT * C_ext[iz][iy][ix] * Csp3 / Cv;

				//计算偏应力方程中的参数
				StressParameterxy[iz][iy][ix] = Cp * DT * muxy[iz][iy][ix] * Csp4 / Cv;
				StressParameterxz[iz][iy][ix] = Cp * DT * muxz[iz][iy][ix] * Csp4 / Cv;
				StressParameteryz[iz][iy][ix] = Cp * DT * muyz[iz][iy][ix] * Csp4 / Cv;
			}
		}
	}



	char rhof_extxname[] = "rhof_extx.dat";
	wfile3d(rhof_extxname, rhof_extx, NZ_ext, NY_ext, NX_ext);
	char rhof_extyname[] = "rhof_exty.dat";
	wfile3d(rhof_extyname, rhof_exty, NZ_ext, NY_ext, NX_ext);
	char rhof_extzname[] = "rhof_extz.dat";
	wfile3d(rhof_extzname, rhof_extz, NZ_ext, NY_ext, NX_ext);

	char rho_tempxname[] = "rho_tempx.dat";
	wfile3d(rho_tempxname, rho_tempx, NZ_ext, NY_ext, NX_ext);
	char rho_tempyname[] = "rho_tempy.dat";
	wfile3d(rho_tempyname, rho_tempy, NZ_ext, NY_ext, NX_ext);
	char rho_tempzname[] = "rho_tempz.dat";
	wfile3d(rho_tempzname, rho_tempz, NZ_ext, NY_ext, NX_ext);
	char muxzname[] = "muxz.dat";
	wfile3d(muxzname, muxz, NZ_ext, NY_ext, NX_ext);
	char muxyname[] = "muxy.dat";
	wfile3d(muxyname, muxy, NZ_ext, NY_ext, NX_ext);
	char muyzname[] = "muyz.dat";
	wfile3d(muyzname, muyz, NZ_ext, NY_ext, NX_ext);
	char C1xname[] = "C1x.dat";
	wfile3d(C1xname, C1x, NZ_ext, NY_ext, NX_ext);
	char C1yname[] = "C1y.dat";
	wfile3d(C1yname, C1y, NZ_ext, NY_ext, NX_ext);
	char C1zname[] = "C1z.dat";
	wfile3d(C1zname, C1z, NZ_ext, NY_ext, NX_ext);
	char C2xname[] = "C2x.dat";
	wfile3d(C2xname, C2x, NZ_ext, NY_ext, NX_ext);
	char C2yname[] = "C2y.dat";
	wfile3d(C2yname, C2y, NZ_ext, NY_ext, NX_ext);
	char C2zname[] = "C2z.dat";
	wfile3d(C2zname, C2z, NZ_ext, NY_ext, NX_ext);
	char cname[] = "C.dat";
	wfile3d(cname, C_ext, NZ_ext, NY_ext, NX_ext);
	char Mname[] = "M.dat";
	wfile3d(Mname, M_ext, NZ_ext, NY_ext, NX_ext);
	char C1name[] = "C1.dat";
	wfile3d(C1name, C1_ext, NZ_ext, NY_ext, NX_ext);
	char C2name[] = "C2.dat";
	wfile3d(C2name, C2_ext, NZ_ext, NY_ext, NX_ext);
	char HHname[] = "HH.dat";
	wfile3d(HHname, HH_ext, NZ_ext, NY_ext, NX_ext);
	char H2Uname[] = "H2U.dat";
	wfile3d(H2Uname, H2u_ext, NZ_ext, NY_ext, NX_ext);

	char VelocityWParameter1xname[] = "VelocityWParameter1x.dat";
	wfile3d(VelocityWParameter1xname, VelocityWParameter1x, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter1yname[] = "VelocityWParameter1y.dat";
	wfile3d(VelocityWParameter1yname, VelocityWParameter1y, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter1zname[] = "VelocityWParameter1z.dat";
	wfile3d(VelocityWParameter1zname, VelocityWParameter1z, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter2xname[] = "VelocityWParameter2x.dat";
	wfile3d(VelocityWParameter2xname, VelocityWParameter2x, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter2yname[] = "VelocityWParameter2y.dat";
	wfile3d(VelocityWParameter2yname, VelocityWParameter2y, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter2zname[] = "VelocityWParameter2z.dat";
	wfile3d(VelocityWParameter2zname, VelocityWParameter2z, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter3xname[] = "VelocityWParameter3x.dat";
	wfile3d(VelocityWParameter3xname, VelocityWParameter3x, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter3yname[] = "VelocityWParameter3y.dat";
	wfile3d(VelocityWParameter3yname, VelocityWParameter3y, NZ_ext, NY_ext, NX_ext);
	char VelocityWParameter3zname[] = "VelocityWParameter3z.dat";
	wfile3d(VelocityWParameter3zname, VelocityWParameter3z, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter1xname[] = "VelocityUParameter1x.dat";
	wfile3d(VelocityUParameter1xname, VelocityUParameter1x, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter1yname[] = "VelocityUParameter1y.dat";
	wfile3d(VelocityUParameter1yname, VelocityUParameter1y, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter1zname[] = "VelocityUParameter1z.dat";
	wfile3d(VelocityUParameter1zname, VelocityUParameter1z, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter2xname[] = "VelocityUParameter2x.dat";
	wfile3d(VelocityUParameter2xname, VelocityUParameter2x, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter2yname[] = "VelocityUParameter2y.dat";
	wfile3d(VelocityUParameter2yname, VelocityUParameter2y, NZ_ext, NY_ext, NX_ext);
	char VelocityUParameter2zname[] = "VelocityUParameter2z.dat";
	wfile3d(VelocityUParameter2zname, VelocityUParameter2z, NZ_ext, NY_ext, NX_ext);
	char PressParameter1name[] = "PressParameter1.dat";
	wfile3d(PressParameter1name, PressParameter1, NZ_ext, NY_ext, NX_ext);
	char PressParameter2name[] = "PressParameter2.dat";
	wfile3d(PressParameter2name, PressParameter2, NZ_ext, NY_ext, NX_ext);
	char StressParameter1name[] = "StressParameter1.dat";
	wfile3d(StressParameter1name, StressParameter1, NZ_ext, NY_ext, NX_ext);
	char StressParameter2name[] = "StressParameter2.dat";
	wfile3d(StressParameter2name, StressParameter2, NZ_ext, NY_ext, NX_ext);
	char StressParameter3name[] = "StressParameter3.dat";
	wfile3d(StressParameter3name, StressParameter3, NZ_ext, NY_ext, NX_ext);
	char StressParameterxyname[] = "StressParameterxy.dat";
	wfile3d(StressParameterxyname, StressParameterxy, NZ_ext, NY_ext, NX_ext);
	char StressParameterxzname[] = "StressParameterxz.dat";
	wfile3d(StressParameterxzname, StressParameterxz, NZ_ext, NY_ext, NX_ext);
	char StressParameteryzname[] = "StressParameteryz.dat";
	wfile3d(StressParameteryzname, StressParameteryz, NZ_ext, NY_ext, NX_ext);
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
	char dxiname[] = "dxi.dat";
	wfile3d(dxiname, dxi, NZ_ext, NY_ext, NX_ext);
	char dxi2name[] = "dxi2.dat";
	wfile3d(dxi2name, dxi2, NZ_ext, NY_ext, NX_ext);
	char dyjname[] = "dyj.dat";
	wfile3d(dyjname, dyj, NZ_ext, NY_ext, NX_ext);
	char dyj2name[] = "dyj2.dat";
	wfile3d(dyj2name, dyj2, NZ_ext, NY_ext, NX_ext);
	char dzkname[] = "dzk.dat";
	wfile3d(dzkname, dzk, NZ_ext, NY_ext, NX_ext);
	char dzk2name[] = "dzk2.dat";
	wfile3d(dzk2name, dzk2, NZ_ext, NY_ext, NX_ext);
	//-----------------------------------------------并行计算-------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------
	//在主机端CPU定义参数，分配内存
	//地层参数
	half* h_VelocityWParameter1x = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter1y = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter1z = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter2x = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter2y = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter2z = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter3x = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter3y = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityWParameter3z = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter1x = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter1y = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter1z = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter2x = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter2y = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_VelocityUParameter2z = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_PressParameter1 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_PressParameter2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameter1 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameter2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameter3 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameterxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameterxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_StressParameteryz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//速度应力
	half* h_vwx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vwy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vwz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_ss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vux = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vuy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vuz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_txx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_tyy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_tzz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_txz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_txy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_tyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//前一时刻的速度
	half* h_vwx2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vwy2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_vwz2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//pml内的差分值
	half* h_pmlxSxx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlySxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzSxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxSxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlySyy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzSyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxSxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlySyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzSzz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxVux = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlyVuy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzVuz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxVuy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlyVux = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxVuz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzVux = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlyVuz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzVuy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxVwx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlyVwy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzVwz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlxss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlyss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_pmlzss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//前一时刻差分值
	half* h_SXxx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SXxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SXxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SYxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SYyy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SYyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SZxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SZyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SZzz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SXss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SYss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_SZss = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuxx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuyy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuzz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuxy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuyx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuyz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuzy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuxz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vuzx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vwxx = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vwyy = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_Vwzz = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//pml参数
	half* h_dxi = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_dyj = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_dzk = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_dxi2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_dyj2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_dzk2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dxi = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dyj = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dzk = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dxi2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dyj2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	half* h_e_dzk2 = (half*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(half));
	//---------------------------------------------------------------------------------
	//在主机端将三维参数转化为一维参数
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				h_VelocityWParameter1x[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter1x[k][j][i];
				h_VelocityWParameter1y[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter1y[k][j][i];
				h_VelocityWParameter1z[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter1z[k][j][i];
				h_VelocityWParameter2x[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter2x[k][j][i];
				h_VelocityWParameter2y[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter2y[k][j][i];
				h_VelocityWParameter2z[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter2z[k][j][i];
				h_VelocityWParameter3x[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter3x[k][j][i];
				h_VelocityWParameter3y[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter3y[k][j][i];
				h_VelocityWParameter3z[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityWParameter3z[k][j][i];
				h_VelocityUParameter1x[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter1x[k][j][i];
				h_VelocityUParameter1y[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter1y[k][j][i];
				h_VelocityUParameter1z[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter1z[k][j][i];
				h_VelocityUParameter2x[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter2x[k][j][i];
				h_VelocityUParameter2y[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter2y[k][j][i];
				h_VelocityUParameter2z[i + j * NX_ext + k * NX_ext * NY_ext] = VelocityUParameter2z[k][j][i];
				h_PressParameter1[i + j * NX_ext + k * NX_ext * NY_ext] = PressParameter1[k][j][i];
				h_PressParameter2[i + j * NX_ext + k * NX_ext * NY_ext] = PressParameter2[k][j][i];
				h_StressParameter1[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameter1[k][j][i];
				h_StressParameter2[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameter2[k][j][i];
				h_StressParameter3[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameter3[k][j][i];
				h_StressParameterxy[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameterxy[k][j][i];
				h_StressParameterxz[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameterxz[k][j][i];
				h_StressParameteryz[i + j * NX_ext + k * NX_ext * NY_ext] = StressParameteryz[k][j][i];
				h_dxi[i + j * NX_ext + k * NX_ext * NY_ext] = dxi[k][j][i] * Cpml;
				h_dyj[i + j * NX_ext + k * NX_ext * NY_ext] = dyj[k][j][i] * Cpml;
				h_dzk[i + j * NX_ext + k * NX_ext * NY_ext] = dzk[k][j][i] * Cpml;
				h_dxi2[i + j * NX_ext + k * NX_ext * NY_ext] = dxi2[k][j][i] * Cpml;
				h_dyj2[i + j * NX_ext + k * NX_ext * NY_ext] = dyj2[k][j][i] * Cpml;
				h_dzk2[i + j * NX_ext + k * NX_ext * NY_ext] = dzk2[k][j][i] * Cpml;
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
	half* d_dxi;
	half* d_dyj;
	half* d_dzk;
	half* d_dxi2;
	half* d_dyj2;
	half* d_dzk2;
	half* d_e_dxi;
	half* d_e_dyj;
	half* d_e_dzk;
	half* d_e_dxi2;
	half* d_e_dyj2;
	half* d_e_dzk2;
	half* d_VelocityWParameter1x;
	half* d_VelocityWParameter1y;
	half* d_VelocityWParameter1z;
	half* d_VelocityWParameter2x;
	half* d_VelocityWParameter2y;
	half* d_VelocityWParameter2z;
	half* d_VelocityWParameter3x;
	half* d_VelocityWParameter3y;
	half* d_VelocityWParameter3z;
	half* d_VelocityUParameter1x;
	half* d_VelocityUParameter1y;
	half* d_VelocityUParameter1z;
	half* d_VelocityUParameter2x;
	half* d_VelocityUParameter2y;
	half* d_VelocityUParameter2z;
	half* d_PressParameter1;
	half* d_PressParameter2;
	half* d_StressParameter1;
	half* d_StressParameter2;
	half* d_StressParameter3;
	half* d_StressParameterxy;
	half* d_StressParameterxz;
	half* d_StressParameteryz;
	cudaMalloc(&d_VelocityWParameter1x, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter1y, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter1z, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter2x, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter2y, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter2z, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter3x, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter3y, mem_sizeHalf);
	cudaMalloc(&d_VelocityWParameter3z, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter1x, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter1y, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter1z, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter2x, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter2y, mem_sizeHalf);
	cudaMalloc(&d_VelocityUParameter2z, mem_sizeHalf);
	cudaMalloc(&d_PressParameter1, mem_sizeHalf);
	cudaMalloc(&d_PressParameter2, mem_sizeHalf);
	cudaMalloc(&d_StressParameter1, mem_sizeHalf);
	cudaMalloc(&d_StressParameter2, mem_sizeHalf);
	cudaMalloc(&d_StressParameter3, mem_sizeHalf);
	cudaMalloc(&d_StressParameterxy, mem_sizeHalf);
	cudaMalloc(&d_StressParameterxz, mem_sizeHalf);
	cudaMalloc(&d_StressParameteryz, mem_sizeHalf);
	cudaMalloc(&d_vux, mem_sizeHalf);
	cudaMalloc(&d_vuy, mem_sizeHalf);
	cudaMalloc(&d_vuz, mem_sizeHalf);
	cudaMalloc(&d_txx, mem_sizeHalf);
	cudaMalloc(&d_tyy, mem_sizeHalf);
	cudaMalloc(&d_tzz, mem_sizeHalf);
	cudaMalloc(&d_txz, mem_sizeHalf);
	cudaMalloc(&d_txy, mem_sizeHalf);
	cudaMalloc(&d_tyz, mem_sizeHalf);
	cudaMalloc(&d_ss, mem_sizeHalf);
	cudaMalloc(&d_vwx, mem_sizeHalf);
	cudaMalloc(&d_vwy, mem_sizeHalf);
	cudaMalloc(&d_vwz, mem_sizeHalf);
	cudaMalloc(&d_vwx2, mem_sizeHalf);
	cudaMalloc(&d_vwy2, mem_sizeHalf);
	cudaMalloc(&d_vwz2, mem_sizeHalf);
	cudaMalloc(&d_pmlxSxx, mem_sizeHalf);
	cudaMalloc(&d_pmlySxy, mem_sizeHalf);
	cudaMalloc(&d_pmlzSxz, mem_sizeHalf);
	cudaMalloc(&d_pmlxSxy, mem_sizeHalf);
	cudaMalloc(&d_pmlySyy, mem_sizeHalf);
	cudaMalloc(&d_pmlzSyz, mem_sizeHalf);
	cudaMalloc(&d_pmlxSxz, mem_sizeHalf);
	cudaMalloc(&d_pmlySyz, mem_sizeHalf);
	cudaMalloc(&d_pmlzSzz, mem_sizeHalf);
	cudaMalloc(&d_pmlxVux, mem_sizeHalf);
	cudaMalloc(&d_pmlyVuy, mem_sizeHalf);
	cudaMalloc(&d_pmlzVuz, mem_sizeHalf);
	cudaMalloc(&d_pmlxVuy, mem_sizeHalf);
	cudaMalloc(&d_pmlyVux, mem_sizeHalf);
	cudaMalloc(&d_pmlxVuz, mem_sizeHalf);
	cudaMalloc(&d_pmlzVux, mem_sizeHalf);
	cudaMalloc(&d_pmlyVuz, mem_sizeHalf);
	cudaMalloc(&d_pmlzVuy, mem_sizeHalf);
	cudaMalloc(&d_SXxx, mem_sizeHalf);
	cudaMalloc(&d_SXxy, mem_sizeHalf);
	cudaMalloc(&d_SXxz, mem_sizeHalf);
	cudaMalloc(&d_SYxy, mem_sizeHalf);
	cudaMalloc(&d_SYyy, mem_sizeHalf);
	cudaMalloc(&d_SYyz, mem_sizeHalf);
	cudaMalloc(&d_SZxz, mem_sizeHalf);
	cudaMalloc(&d_SZyz, mem_sizeHalf);
	cudaMalloc(&d_SZzz, mem_sizeHalf);
	cudaMalloc(&d_Vuxx, mem_sizeHalf);
	cudaMalloc(&d_Vuyy, mem_sizeHalf);
	cudaMalloc(&d_Vuzz, mem_sizeHalf);
	cudaMalloc(&d_Vuxy, mem_sizeHalf);
	cudaMalloc(&d_Vuyx, mem_sizeHalf);
	cudaMalloc(&d_Vuxz, mem_sizeHalf);
	cudaMalloc(&d_Vuzx, mem_sizeHalf);
	cudaMalloc(&d_Vuyz, mem_sizeHalf);
	cudaMalloc(&d_Vuzy, mem_sizeHalf);
	cudaMalloc(&d_pmlxVwx, mem_sizeHalf);
	cudaMalloc(&d_pmlyVwy, mem_sizeHalf);
	cudaMalloc(&d_pmlzVwz, mem_sizeHalf);
	cudaMalloc(&d_pmlxss, mem_sizeHalf);
	cudaMalloc(&d_pmlyss, mem_sizeHalf);
	cudaMalloc(&d_pmlzss, mem_sizeHalf);
	cudaMalloc(&d_SXss, mem_sizeHalf);
	cudaMalloc(&d_SYss, mem_sizeHalf);
	cudaMalloc(&d_SZss, mem_sizeHalf);
	cudaMalloc(&d_Vwxx, mem_sizeHalf);
	cudaMalloc(&d_Vwyy, mem_sizeHalf);
	cudaMalloc(&d_Vwzz, mem_sizeHalf);
	cudaMalloc(&d_dxi, mem_sizeHalf);
	cudaMalloc(&d_dyj, mem_sizeHalf);
	cudaMalloc(&d_dzk, mem_sizeHalf);
	cudaMalloc(&d_dxi2, mem_sizeHalf);
	cudaMalloc(&d_dyj2, mem_sizeHalf);
	cudaMalloc(&d_dzk2, mem_sizeHalf);
	cudaMalloc(&d_e_dxi, mem_sizeHalf);
	cudaMalloc(&d_e_dyj, mem_sizeHalf);
	cudaMalloc(&d_e_dzk, mem_sizeHalf);
	cudaMalloc(&d_e_dxi2, mem_sizeHalf);
	cudaMalloc(&d_e_dyj2, mem_sizeHalf);
	cudaMalloc(&d_e_dzk2, mem_sizeHalf);
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
	cudaMemcpy(d_VelocityWParameter1x, h_VelocityWParameter1x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter1y, h_VelocityWParameter1y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter1z, h_VelocityWParameter1z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter2x, h_VelocityWParameter2x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter2y, h_VelocityWParameter2y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter2z, h_VelocityWParameter2z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter3x, h_VelocityWParameter3x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter3y, h_VelocityWParameter3y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityWParameter3z, h_VelocityWParameter3z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter1x, h_VelocityUParameter1x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter1y, h_VelocityUParameter1y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter1z, h_VelocityUParameter1z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter2x, h_VelocityUParameter2x, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter2y, h_VelocityUParameter2y, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_VelocityUParameter2z, h_VelocityUParameter2z, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_PressParameter1, h_PressParameter1, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_PressParameter2, h_PressParameter2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameter1, h_StressParameter1, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameter2, h_StressParameter2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameter3, h_StressParameter3, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameterxy, h_StressParameterxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameterxz, h_StressParameterxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_StressParameteryz, h_StressParameteryz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vux, h_vux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuy, h_vuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vuz, h_vuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txx, h_txx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyy, h_tyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tzz, h_tzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txy, h_txy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_txz, h_txz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tyz, h_tyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxx, h_pmlxSxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySxy, h_pmlySxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSxz, h_pmlzSxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxy, h_pmlxSxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyy, h_pmlySyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSyz, h_pmlzSyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxSxz, h_pmlxSxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlySyz, h_pmlySyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzSzz, h_pmlzSzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVux, h_pmlxVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuy, h_pmlyVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuz, h_pmlzVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuy, h_pmlxVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVux, h_pmlyVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVuz, h_pmlxVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVux, h_pmlzVux, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVuz, h_pmlyVuz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVuy, h_pmlzVuy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxx, h_SXxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxy, h_SXxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXxz, h_SXxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYxy, h_SYxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyy, h_SYyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYyz, h_SYyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZxz, h_SZxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZyz, h_SZyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZzz, h_SZzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxx, h_Vuxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyy, h_Vuyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzz, h_Vuzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxy, h_Vuxy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyx, h_Vuyx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuxz, h_Vuxz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzx, h_Vuzx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuyz, h_Vuyz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vuzy, h_Vuzy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxi, h_dxi, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj, h_dyj, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk, h_dzk, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxi2, h_dxi2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj2, h_dyj2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk2, h_dzk2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi, h_e_dxi, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj, h_e_dyj, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk, h_e_dzk, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi2, h_e_dxi2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj2, h_e_dyj2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk2, h_e_dzk2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxVwx, h_pmlxVwx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyVwy, h_pmlyVwy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzVwz, h_pmlzVwz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlxss, h_pmlxss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlyss, h_pmlyss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlzss, h_pmlzss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SXss, h_SXss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SYss, h_SYss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_SZss, h_SZss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwxx, h_Vwxx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwyy, h_Vwyy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vwzz, h_Vwzz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ss, h_ss, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx, h_vwx, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy, h_vwy, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz, h_vwz, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwx2, h_vwx2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwy2, h_vwy2, mem_sizeHalf, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vwz2, h_vwz2, mem_sizeHalf, cudaMemcpyHostToDevice);
	//---------------------------------------------------------------------------------------------------------------------
	//在时间上进行迭代
	clock_t start, finish;
	start = clock();
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
		FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_SXxx, d_SXxy, d_SXxz, d_SYxy, d_SYyy, d_SYyz, d_SZxz, d_SZyz, d_SZzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj, d_dyj, d_e_dyj2, d_dyj2, d_e_dzk, d_dzk, d_dzk2, d_e_dzk2, d_ss, d_vwx, d_vwy, d_vwz, d_vwx2, d_vwy2, d_vwz2, d_SXss, d_SYss, d_SZss, d_pmlxss, d_pmlyss, d_pmlzss, DT, d_VelocityWParameter1x, d_VelocityWParameter1y, d_VelocityWParameter1z, d_VelocityWParameter2x, d_VelocityWParameter2y, d_VelocityWParameter2z, d_VelocityWParameter3x, d_VelocityWParameter3y, d_VelocityWParameter3z, d_VelocityUParameter1x, d_VelocityUParameter1y, d_VelocityUParameter1z, d_VelocityUParameter2x, d_VelocityUParameter2y, d_VelocityUParameter2z);
		cudaDeviceSynchronize();
		////----------------------------------------------------------------------------------------------------------
		////计算应力
		////FD_T  <BLOCK_X, BLOCK_Y> << <Gridsize1, Blocksize1 >> > (d_vux, d_vuy, d_vuz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_lamda2u_ext, d_lamda_ext, d_pmlxVux, d_muxy, d_muxz, d_muyz, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_Vuxx, d_Vuxy, d_Vuxz, d_Vuyx, d_Vuyy, d_Vuyz, d_Vuzx, d_Vuzy, d_Vuzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj2, d_dyj2, d_e_dyj, d_dyj, d_dzk2, d_e_dzk2, d_e_dzk, d_dzk, d_Vwxx, d_Vwyy, d_Vwzz, d_vwx, d_vwy, d_vwz, d_C_ext, d_M_ext, d_HH_ext, d_H2u_ext, d_ss, d_pmlxVwx, d_pmlyVwy, d_pmlzVwz);
		FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_pmlxVux, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_Vuxx, d_Vuxy, d_Vuxz, d_Vuyx, d_Vuyy, d_Vuyz, d_Vuzx, d_Vuzy, d_Vuzz, d_e_dxi, d_dxi, d_e_dxi2, d_dxi2, d_e_dyj2, d_dyj2, d_e_dyj, d_dyj, d_dzk2, d_e_dzk2, d_e_dzk, d_dzk, d_Vwxx, d_Vwyy, d_Vwzz, d_vwx, d_vwy, d_vwz, d_ss, d_pmlxVwx, d_pmlyVwy, d_pmlzVwz, DT, d_PressParameter1, d_PressParameter2, d_StressParameter1, d_StressParameter2, d_StressParameter3, d_StressParameterxy, d_StressParameterxz, d_StressParameteryz);
		cudaDeviceSynchronize();
		//----------------------------------------------------------------------------------------------------------
		//将速度，应力从设备端拷贝到主机
		cudaMemcpy(h_txx, d_txx, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_tzz, d_tzz, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_ss, d_ss, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_vux, d_vux, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_vwx, d_vwx, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_vuz, d_vuz, mem_sizeHalf, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_vwz, d_vwz, mem_sizeHalf, cudaMemcpyDeviceToHost);

		//记录地震记录

		for (iz = 0; iz < NZ_ext; iz++)
		{
			sis_x[iz][it] = __half2float(h_txx[iz * NX_ext * NY_ext + sx + sy * NX_ext]);
			//sis_z[iz][it] = h_tzz[iz * NX_ext * NY_ext + sx + sy * NX_ext];

			//sis_vu[iz][it] = h_vux[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			//sis_vw[iz][it] = h_vwx[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			//sis_p[iz][it] = h_ss[iz * NX_ext * NY_ext + sx + sy * NX_ext];
		}
		if (it == 500)
			for (int k = 0; k < NZ_ext; k++)
			{
				for (int j = 0; j < NY_ext; j++)
				{
					for (int i = 0; i < NX_ext; i++)
					{
						txx50[k][j][i] = __half2float(h_txx[i + j * NX_ext + k * NX_ext * NY_ext]);
						//vux50[k][j][i] = h_vux[i + j * NX_ext + k * NX_ext * NY_ext];
						//vwx50[k][j][i] = h_vwx[i + j * NX_ext + k * NX_ext * NY_ext];
						//vuz50[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						//vwz50[k][j][i] = h_vwz[i + j * NX_ext + k * NX_ext * NY_ext];
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
	free(h_VelocityWParameter1x);
	free(h_VelocityWParameter1y);
	free(h_VelocityWParameter1z);
	free(h_VelocityWParameter2x);
	free(h_VelocityWParameter2y);
	free(h_VelocityWParameter2z);
	free(h_VelocityWParameter3x);
	free(h_VelocityWParameter3y);
	free(h_VelocityWParameter3z);
	free(h_VelocityUParameter1x);
	free(h_VelocityUParameter1y);
	free(h_VelocityUParameter1z);
	free(h_VelocityUParameter2x);
	free(h_VelocityUParameter2y);
	free(h_VelocityUParameter2z);
	free(h_PressParameter1);
	free(h_PressParameter2);
	free(h_StressParameter1);
	free(h_StressParameter2);
	free(h_StressParameter3);
	free(h_StressParameterxy);
	free(h_StressParameterxz);
	free(h_StressParameteryz);
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
	free_space3d(VelocityWParameter1x, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter1y, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter1z, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter2x, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter2y, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter2z, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter3x, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter3y, NZ_ext, NY_ext);
	free_space3d(VelocityWParameter3z, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter1x, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter1y, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter1z, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter2x, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter2y, NZ_ext, NY_ext);
	free_space3d(VelocityUParameter2z, NZ_ext, NY_ext);
	free_space3d(PressParameter1, NZ_ext, NY_ext);
	free_space3d(PressParameter2, NZ_ext, NY_ext);
	free_space3d(StressParameter1, NZ_ext, NY_ext);
	free_space3d(StressParameter2, NZ_ext, NY_ext);
	free_space3d(StressParameter3, NZ_ext, NY_ext);
	free_space3d(StressParameterxy, NZ_ext, NY_ext);
	free_space3d(StressParameterxz, NZ_ext, NY_ext);
	free_space3d(StressParameteryz, NZ_ext, NY_ext);
	//设备端释放内存
	cudaFree(d_VelocityWParameter1x);
	cudaFree(d_VelocityWParameter1y);
	cudaFree(d_VelocityWParameter1z);
	cudaFree(d_VelocityWParameter2x);
	cudaFree(d_VelocityWParameter2y);
	cudaFree(d_VelocityWParameter2z);
	cudaFree(d_VelocityWParameter3x);
	cudaFree(d_VelocityWParameter3y);
	cudaFree(d_VelocityWParameter3z);
	cudaFree(d_VelocityUParameter1x);
	cudaFree(d_VelocityUParameter1y);
	cudaFree(d_VelocityUParameter1z);
	cudaFree(d_VelocityUParameter2x);
	cudaFree(d_VelocityUParameter2y);
	cudaFree(d_VelocityUParameter2z);
	cudaFree(d_PressParameter1);
	cudaFree(d_PressParameter2);
	cudaFree(d_StressParameter1);
	cudaFree(d_StressParameter2);
	cudaFree(d_StressParameter3);
	cudaFree(d_StressParameterxy);
	cudaFree(d_StressParameterxz);
	cudaFree(d_StressParameteryz);
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
					vp[iz][iy][ix] = 6096.0f;
					vs[iz][iy][ix] = 3424.0f;
					rhos[iz][iy][ix] = 2770.0f;
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
