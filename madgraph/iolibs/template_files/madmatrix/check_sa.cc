// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: O. Mattelaer (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2026).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
//
// Standalone script for MadGraph7 standalone_mg7 mode.
// Generates phase-space points with RAMBO and evaluates the matrix element
// through the UMAMI interface (umami.h).
//
//==========================================================================

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "GpuAbstraction.h"
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"
#include "epoch_process_id.h"
#include "timermap.h"
#include "umami.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#define STRINGIFY( s ) #s
#define XSTRINGIFY( s ) STRINGIFY( s )
#define SEP79 79

namespace
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  // Fixed physics inputs
  constexpr fptype kEnergy = 1500.;                  // Ecms = 1.5 TeV
  constexpr unsigned long long kSeed = 20200805ULL;  // reproducible RAMBO seed

  bool is_number( const char* s )
  {
    const char* t = s;
    while( *t != '\0' && isdigit( *t ) ) ++t;
    return (int)strlen( s ) == t - s;
  }

  int usage( const char* argv0, int ret = 1 )
  {
    std::cout << "Usage: " << argv0
              << " [--verbose|-v] [--debug|-d] [--performance|-p] [--json|-j] [--flavor|-f <int>]"
              << " [#blocksPerGrid #threadsPerBlock] #iterations" << std::endl
              << std::endl
              << "Number of events per iteration = #blocksPerGrid * #threadsPerBlock" << std::endl
              << "(in CPU/C++ code only the product matters)." << std::endl;
    return ret;
  }
  // AOSOA -> UMAMI SoA single-event helper. Layout reminder:
  //   AOSOA: aosoa[i_page * npar*4*neppM + ipar*4*neppM + ip4*neppM + i_vector]
  //   UMAMI: soa[ip4 * npar*nevt + ipar*nevt + ievt]
  __host__ __device__ inline void
  aosoa_to_umami_one( const fptype* aosoa,
                      double* soa,
                      std::size_t ievt,
                      std::size_t nevt )
  {
    constexpr int npar = CPPProcess::npar;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      for( int ip4 = 0; ip4 < 4; ++ip4 )
      {
        soa[(std::size_t)ip4 * npar * nevt + (std::size_t)ipar * nevt + ievt] =
          (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, ip4, ipar );
      }
    }
  }

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  aosoa_to_umami_kernel( const fptype* aosoa,
                         double* soa,
                         std::size_t nevt )
  {
    std::size_t ievt = blockDim.x * blockIdx.x + threadIdx.x;
    if( ievt >= nevt ) return;
    aosoa_to_umami_one( aosoa, soa, ievt, nevt );
  }
#endif
}

int main( int argc, char** argv )
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  // CLI defaults
  bool verbose = false;
  bool debug = false;
  bool perf = false;
  bool json = false;
  unsigned int niter = 0;
  unsigned int gpublocks = 1;
  unsigned int gputhreads = 32;
  unsigned int jsondate = 0;
  unsigned int jsonrun = 0;
  unsigned int flavorID = 0; // default flavor index
  unsigned int numvec[5] = { 0, 0, 0, 0, 0 };
  int nnum = 0;
  constexpr unsigned int UmamiInKeyNum = 2;

  for( int argn = 1; argn < argc; ++argn )
  {
    std::string arg = argv[argn];
    if( arg == "--verbose" || arg == "-v" ) verbose = true;
    else if( arg == "--debug" || arg == "-d" ) debug = true;
    else if( arg == "--performance" || arg == "-p" ) perf = true;
    else if( arg == "--json" || arg == "-j" ) json = true;
    else if( ( arg == "--flavor" || arg == "-f" ) && argn + 1 < argc && is_number( argv[argn + 1] ) )
      flavorID = strtoul( argv[++argn], nullptr, 0 );
    else if( is_number( argv[argn] ) && nnum < 5 )
      numvec[nnum++] = strtoul( argv[argn], nullptr, 0 );
    else
      return usage( argv[0] );
  }

  if( nnum == 3 || nnum == 5 )
  {
    gpublocks = numvec[0];
    gputhreads = numvec[1];
    niter = numvec[2];
    if( nnum == 5 )
    {
      jsondate = numvec[3];
      jsonrun = numvec[4];
    }
  }
  else if( nnum == 1 )
  {
    niter = numvec[0];
  }
  else
  {
    return usage( argv[0] );
  }
  if( niter == 0 ) return usage( argv[0] );

  const unsigned int nevt = gpublocks * gputhreads;

  mgOnGpu::TimerMap timermap;

#ifdef MGONGPUCPP_GPUIMPL
  // Initialise the GPU runtime (cudaSetDevice / cudaDeviceReset).
  timermap.start( "00 GpuInit" );
  GpuRuntime gpuRuntime( debug );
#else
  (void)debug;
#endif

  // ---- Buffers ----
#ifdef MGONGPUCPP_GPUIMPL
  PinnedHostBufferRndNumMomenta hstRndmom( nevt );
  PinnedHostBufferMomenta hstMomenta( nevt ); // for verbose printing
  PinnedHostBufferWeights hstWeights( nevt );
  DeviceBufferRndNumMomenta devRndmom( nevt );
  DeviceBufferMomenta devMomenta( nevt );
  DeviceBufferWeights devWeights( nevt );
  // SoA buffers for UMAMI live entirely on the device.
  DeviceBufferBase<double> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
  DeviceBufferBase<double> devUmamiMEs( nevt );
  DeviceBufferBase<unsigned int> devFlv( nevt );
  std::vector<double> hstUmamiMEs( nevt );
#else
  HostBufferRndNumMomenta hstRndmom( nevt );
  HostBufferMomenta hstMomenta( nevt );
  HostBufferWeights hstWeights( nevt );
  std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
  std::vector<double> umamiMEs( nevt );
#endif

  // ---- Kernels ----
  std::unique_ptr<RandomNumberKernelBase> prnk(
    new CommonRandomNumberKernel( hstRndmom ) );

  std::unique_ptr<SamplingKernelBase> prsk;
#ifdef MGONGPUCPP_GPUIMPL
  prsk.reset( new RamboSamplingKernelDevice( kEnergy, devRndmom, devMomenta, devWeights, gpublocks, gputhreads ) );
#else
  prsk.reset( new RamboSamplingKernelHost( kEnergy, hstRndmom, hstMomenta, hstWeights, nevt ) );
#endif

  // ---- UMAMI handle ----
  UmamiHandle umami_handle = nullptr;
  if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
  {
    std::cerr << "ERROR! umami_initialize failed" << std::endl;
    return 2;
  }

  // ---- Per-iteration timings ----
  std::unique_ptr<double[]> genrtimes( new double[niter] );
  std::unique_ptr<double[]> rambtimes( new double[niter] );
  std::unique_ptr<double[]> wavetimes( new double[niter] );

  // ---- Inline event statistics ----
  unsigned int nevtABN = 0;
  unsigned int nevtZERO = 0;
  double sumME = 0.;
  double sumMEsq = 0.;
  double minME = std::numeric_limits<double>::infinity();
  double maxME = -std::numeric_limits<double>::infinity();
  unsigned int nevtALL = 0;

  const int meGeVexponent = -( 2 * CPPProcess::npar - 8 );

  for( unsigned int iiter = 0; iiter < niter; ++iiter )
  {
    // Step 1 - random numbers (always generated on host).
    double genrtime = 0;
    timermap.start( "1a GenSeed " );
    prnk->seedGenerator( kSeed + iiter );
    genrtime += timermap.stop();
    timermap.start( "1b GenRnGen" );
    prnk->generateRnarray();
    genrtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "1c CpHTDrnd" );
    copyDeviceFromHost( devRndmom, hstRndmom );
    genrtime += timermap.stop();
#endif

    // Step 2 - RAMBO momenta.
    double rambtime = 0;
    timermap.start( "2a RamboIni" );
    prsk->getMomentaInitial();
    rambtime += timermap.stop();
    timermap.start( "2b RamboFin" );
    prsk->getMomentaFinal();

// HARDCODED MOMENTA FOR REPRODUCIBILITY UNCOMMMENT AND SET TO DESIRABLE
/*constexpr fptype fixedMomenta[CPPProcess::npar][4] = {
  { 5.00000000000000000e+02,  0.00000000000000000e+00,  0.00000000000000000e+00,  5.00000000000000000e+02 },
  { 5.00000000000000000e+02,  0.00000000000000000e+00,  0.00000000000000000e+00, -5.00000000000000000e+02 },
  { 1.48659511984701965e+02, -1.41759229004224707e+01,  3.50671576461468604e+01, -7.10210403195511617e+01 },
  { 2.61015718768703039e+02, -7.74573847394504043e+01, -2.45254854257569406e+02,  4.44928697294952826e+01 },
  { 1.18826220061214102e+02, -8.42809323311964818e+01, -7.75060213187374387e+01,  3.17680921485658274e+01 },
  { 3.85888758674269582e+02,  2.16314560882611602e+02,  3.10062481231316838e+02, -7.73265966793502173e+01 },
  { 8.56097905111112709e+01, -4.04003209115422379e+01, -2.23687633011569602e+01,  7.20866751208402832e+01 }
};

for( unsigned int ievt = 0; ievt < nevt; ++ievt )
{
  for( int ipar = 0; ipar < CPPProcess::npar; ipar++ )
  {
    MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, 0, ipar ) = fixedMomenta[ipar][0];
    MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, 1, ipar ) = fixedMomenta[ipar][1];
    MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, 2, ipar ) = fixedMomenta[ipar][2];
    MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, 3, ipar ) = fixedMomenta[ipar][3];
  }
  if( verbose )
  {
    std::cout << "Momenta (fixed) for event: " << ievt << std::endl;
    for( int ipar = 0; ipar < CPPProcess::npar; ipar++ )
    {
      const int ndigits = std::numeric_limits<double>::digits10;
      std::cout << std::scientific
                << std::setprecision( ndigits )
                << std::setw( 4 ) << ipar + 1
                << std::setw( ndigits + 8 ) << fixedMomenta[ipar][0]
                << std::setw( ndigits + 8 ) << fixedMomenta[ipar][1]
                << std::setw( ndigits + 8 ) << fixedMomenta[ipar][2]
                << std::setw( ndigits + 8 ) << fixedMomenta[ipar][3]
                << std::endl;
    }
    std::cout << std::string( SEP79, '-' ) << std::endl;
  }
}*/
    // *** STOP THE OLD-STYLE TIMER FOR RAMBO ***
    rambtime += timermap.stop();

    // Step 2c - convert AOSOA -> UMAMI [ip4][ipar][ievt] layout.
    timermap.start( "2c Aosoa2U " );
#ifdef MGONGPUCPP_GPUIMPL
    gpuLaunchKernel( aosoa_to_umami_kernel, gpublocks, gputhreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
    checkGpu( gpuPeekAtLastError() );
#else
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
      aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif
    rambtime += timermap.stop();

    // Step 3 - matrix elements via UMAMI.

    double wavetime = 0;

    std::vector<unsigned int> FlvVec(nevt, flavorID);
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpy( devFlv.data(), FlvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
#endif

    timermap.start( "3a SigmaKin" );
    UmamiInputKey in_keys[UmamiInKeyNum] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX };
    UmamiOutputKey out_keys[1] = { UMAMI_OUT_MATRIX_ELEMENT };
#ifdef MGONGPUCPP_GPUIMPL
    const void* inputs[UmamiInKeyNum] = { devUmamiMomenta.data(), devFlvVec.data() };
    void* outputs[1] = { devUmamiMEs.data() };
#else
    const void* inputs[UmamiInKeyNum] = { umamiMomenta.data(), FlvVec.data() };
    void* outputs[1] = { umamiMEs.data() };
#endif
    UmamiStatus st = umami_matrix_element(
      umami_handle, nevt, nevt, 0, UmamiInKeyNum, in_keys, inputs, 1, out_keys, outputs );
    if( st != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << std::endl;
      umami_free( umami_handle );
      return 3;
    }
    wavetime += timermap.stop();

#ifdef MGONGPUCPP_GPUIMPL
    // Step 3b - copy the matrix elements (and momenta, for verbose) back.
    timermap.start( "3b CpDTHmes" );
    gpuMemcpy( hstUmamiMEs.data(), devUmamiMEs.data(), nevt * sizeof( double ), gpuMemcpyDeviceToHost );
    if( verbose ) copyHostFromDevice( hstMomenta, devMomenta );
    wavetime += timermap.stop();
#endif

#ifdef MGONGPUCPP_GPUIMPL
    const double* mes = hstUmamiMEs.data();
#else
    const double* mes = umamiMEs.data();
#endif

    // Step 4 - update inline statistics + per-event verbose printout.
    timermap.start( "4@ UpdtStat" );
    for( unsigned int ievt = 0; ievt < nevt; ++ievt )
    {
      double me = mes[ievt];
      ++nevtALL;
      if( !std::isfinite( me ) )
        ++nevtABN;
      else if( me == 0. )
        ++nevtZERO;
      sumME += me;
      sumMEsq += me * me;
      if( me < minME ) minME = me;
      if( me > maxME ) maxME = me;
    }

    timermap.start( "4a DumpLoop" );
    genrtimes[iiter] = genrtime;
    rambtimes[iiter] = rambtime;
    wavetimes[iiter] = wavetime;

    if( verbose )
    {
      std::cout << std::string( SEP79, '*' ) << std::endl
                << "Iteration #" << iiter + 1 << " of " << niter << std::endl;
      if( perf ) std::cout << "Wave function time: " << wavetime << std::endl;
      for( unsigned int ievt = 0; ievt < nevt; ++ievt )
      {
        std::cout << "Momenta:" << std::endl;
        for( int ipar = 0; ipar < CPPProcess::npar; ipar++ )
        {
          std::cout << std::scientific
                    << std::setw( 4 ) << ipar + 1
                    << std::setw( 14 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 0, ipar )
                    << std::setw( 14 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 1, ipar )
                    << std::setw( 14 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 2, ipar )
                    << std::setw( 14 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 3, ipar )
                    << std::endl
                    << std::defaultfloat;
        }
        std::cout << std::string( SEP79, '-' ) << std::endl
                  << " Matrix element = " << mes[ievt]
                  << " GeV^" << meGeVexponent << std::endl
                  << std::string( SEP79, '-' ) << std::endl;
      }
    }
    else if( !( debug || perf ) )
    {
      std::cout << "." << std::flush;
    }
  }

  if( !( verbose || debug || perf ) ) std::cout << std::endl;

  // ---- summary ----
  timermap.start( "8a CompStat" );
  double sumgtim = 0, sumrtim = 0, sumwtim = 0;
  double minwtim = wavetimes[0], maxwtim = wavetimes[0];
  for( unsigned int i = 0; i < niter; ++i )
  {
    sumgtim += genrtimes[i];
    sumrtim += rambtimes[i];
    sumwtim += wavetimes[i];
    minwtim = std::min( minwtim, wavetimes[i] );
    maxwtim = std::max( maxwtim, wavetimes[i] );
  }
  double meanwtim = sumwtim / niter;

  unsigned int nevtGood = nevtALL - nevtABN;
  double meanME = ( nevtGood > 0 ) ? sumME / nevtGood : 0.;
  double varME = ( nevtGood > 0 ) ? sumMEsq / nevtGood - meanME * meanME : 0.;
  double stdME = ( varME > 0 ) ? std::sqrt( varME ) : 0.;

  if( perf )
  {
#ifdef __CUDACC__
    const std::string proc_suffix = "_CUDA";
#elif defined( __HIPCC__ )
    const std::string proc_suffix = "_HIP";
#else
    const std::string proc_suffix = "_CPP";
#endif
    std::cout << std::string( SEP79, '*' ) << std::endl
              << "Process                     = " << XSTRINGIFY( MG_EPOCH_PROCESS_ID ) << proc_suffix
#ifdef MGONGPU_HARDCODE_PARAM
              << " [hardcodePARAM=1]" << std::endl
#else
              << " [hardcodePARAM=0]" << std::endl
#endif
              << "NumBlocksPerGrid            = " << gpublocks << std::endl
              << "NumThreadsPerBlock          = " << gputhreads << std::endl
              << "NumIterations               = " << niter << std::endl
              << std::string( SEP79, '-' ) << std::endl
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
              << "FP precision                = MIXED (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << std::endl
#elif defined MGONGPU_FPTYPE_DOUBLE
              << "FP precision                = DOUBLE (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << std::endl
#elif defined MGONGPU_FPTYPE_FLOAT
              << "FP precision                = FLOAT (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << std::endl
#endif
              << "Random number generation    = COMMON RANDOM HOST" << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "NumberOfEntries             = " << niter << std::endl
              << std::scientific
              << "TotalTime[Rnd+Rmb+ME] (123) = ( " << sumgtim + sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[Rambo+ME]    (23) = ( " << sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[RndNumGen]    (1) = ( " << sumgtim << " )  sec" << std::endl
              << "TotalTime[Rambo]        (2) = ( " << sumrtim << " )  sec" << std::endl
              << "TotalTime[MatrixElems]  (3) = ( " << sumwtim << " )  sec" << std::endl
              << "MeanTimeInMatrixElems       = ( " << meanwtim << " )  sec" << std::endl
              << "[Min,Max]TimeInMatrixElems  = [ " << minwtim << " ,  " << maxwtim << " ]  sec" << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "TotalEventsComputed         = " << nevtALL << std::endl
              << "EvtsPerSec[Rnd+Rmb+ME](123) = ( " << nevtALL / ( sumgtim + sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[Rmb+ME]     (23) = ( " << nevtALL / ( sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[MatrixElems] (3) = ( " << nevtALL / sumwtim << " )  sec^-1" << std::endl
              << std::defaultfloat
              << std::string( SEP79, '*' ) << std::endl
              << "MeanMatrixElemValue         = ( " << meanME << " +- " << stdME / std::sqrt( (double)std::max( 1u, nevtGood ) )
              << " )  GeV^" << meGeVexponent << std::endl
              << "[Min,Max]MatrixElemValue    = [ " << minME << " ,  " << maxME << " ]  GeV^" << meGeVexponent << std::endl
              << std::string( SEP79, '*' ) << std::endl;
    timermap.dump();
    std::cout << std::string( SEP79, '*' ) << std::endl;
  }

  // ---- json dump ----
  if( json )
  {
    std::string jsonFileName = "./perf/data/" + std::to_string( jsondate ) + "-perf-test-run" + std::to_string( jsonrun ) + ".json";
    std::ifstream fileCheck( jsonFileName );
    bool fileExists = (bool)fileCheck;
    if( fileCheck ) fileCheck.close();
    std::ofstream jsonFile( jsonFileName, std::ios_base::app );
    if( !fileExists )
    {
      jsonFile << "[" << std::endl;
    }
    else
    {
      std::string temp = "truncate -s-1 " + jsonFileName;
      if( system( temp.c_str() ) != 0 )
        std::cout << "WARNING! Command '" << temp << "' failed" << std::endl;
      jsonFile << ", " << std::endl;
    }
    jsonFile << "{" << std::endl
             << "\"NumIterations\": " << niter << ", " << std::endl
             << "\"NumThreadsPerBlock\": " << gputhreads << ", " << std::endl
             << "\"NumBlocksPerGrid\": " << gpublocks << ", " << std::endl
             << "\"TotalTime[Rnd+Rmb+ME] (123)\": \"" << std::to_string( sumgtim + sumrtim + sumwtim ) << " sec\"," << std::endl
             << "\"TotalTime[Rambo+ME] (23)\": \"" << std::to_string( sumrtim + sumwtim ) << " sec\"," << std::endl
             << "\"TotalTime[RndNumGen] (1)\": \"" << std::to_string( sumgtim ) << " sec\"," << std::endl
             << "\"TotalTime[Rambo] (2)\": \"" << std::to_string( sumrtim ) << " sec\"," << std::endl
             << "\"TotalTime[MatrixElems] (3)\": \"" << std::to_string( sumwtim ) << " sec\"," << std::endl
             << "\"MeanTimeInMatrixElems\": \"" << std::to_string( meanwtim ) << " sec\"," << std::endl
             << "\"MinTimeInMatrixElems\": \"" << std::to_string( minwtim ) << " sec\"," << std::endl
             << "\"MaxTimeInMatrixElems\": \"" << std::to_string( maxwtim ) << " sec\"," << std::endl
             << "\"TotalEventsComputed\": " << nevtALL << "," << std::endl
             << "\"EvtsPerSec[Rnd+Rmb+ME](123)\": \"" << std::to_string( nevtALL / ( sumgtim + sumrtim + sumwtim ) ) << " sec^-1\"," << std::endl
             << "\"EvtsPerSec[Rmb+ME] (23)\": \"" << std::to_string( nevtALL / ( sumrtim + sumwtim ) ) << " sec^-1\"," << std::endl
             << "\"EvtsPerSec[MatrixElems] (3)\": \"" << std::to_string( nevtALL / sumwtim ) << " sec^-1\"," << std::endl
             << "\"NumMatrixElems(notAbnormal)\": " << nevtGood << "," << std::endl
             << "\"MeanMatrixElemValue\": \"" << std::to_string( meanME ) << " GeV^" << std::to_string( meGeVexponent ) << "\"," << std::endl
             << "\"StdDevMatrixElemValue\": \"" << std::to_string( stdME ) << " GeV^" << std::to_string( meGeVexponent ) << "\"," << std::endl
             << "\"MinMatrixElemValue\": \"" << std::to_string( minME ) << " GeV^" << std::to_string( meGeVexponent ) << "\"," << std::endl
             << "\"MaxMatrixElemValue\": \"" << std::to_string( maxME ) << " GeV^" << std::to_string( meGeVexponent ) << "\"" << std::endl
             << "}" << std::endl
             << "]";
    jsonFile.close();
  }

  umami_free( umami_handle );
  return 0;
}
