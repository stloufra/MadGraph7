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

#ifdef __CADNA_ANALYSIS__
#include <cadna.h>
#endif

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

#ifndef __MGCONSTEXPR__
#if __CADNA_ANALYSIS__
#define __MGCONSTEXPR__ const
#else
#define __MGCONSTEXPR__ constexpr
#endif
#endif

  // Fixed physics inputs
  __MGCONSTEXPR__  fptype kEnergy = 1500.;                  // Ecms = 1.5 TeV
  __MGCONSTEXPR__  unsigned long long kSeed = 20200805ULL;  // reproducible RAMBO seed

  template<typename T> auto numeric_infinity()           { return std::numeric_limits<T>::infinity(); }
  template<>           auto numeric_infinity<double_st>() { return std::numeric_limits<double>::infinity(); }
  template<>           auto numeric_infinity<float_st>()  { return std::numeric_limits<float>::infinity(); }

  bool is_number( const char* s )
  {
    const char* t = s;
    while( *t != '\0' && isdigit( *t ) ) ++t;
    return (int)strlen( s ) == t - s;
  }

  int usage( const char* argv0, int ret = 1 )
  {
    std::cout << "Usage: " << argv0
              << " [--verbose|-v] [--debug|-d] [--performance|-p] [--json|-j]"
              << " [#blocksPerGrid #threadsPerBlock] #iterations" << '\n'
              << '\n'
              << "Number of events per iteration = #blocksPerGrid * #threadsPerBlock" << '\n'
              << "(in CPU/C++ code only the product matters)." << '\n';
    return ret;
  }

  // AOSOA -> UMAMI SoA single-event helper. Layout reminder:
  //   AOSOA: aosoa[i_page * npar*4*neppM + ipar*4*neppM + ip4*neppM + i_vector]
  //   UMAMI: soa[ip4 * npar*nevt + ipar*nevt + ievt]
  __host__ __device__ inline void
  aosoa_to_umami_one( const fptype* aosoa,
                      fptype* soa,
                      std::size_t ievt,
                      std::size_t nevt )
  {
    constexpr int npar = CPPProcess::npar;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      for( int ip4 = 0; ip4 < 4; ++ip4 )
      {

  // Casting for cadna to make incoming momenta exact
#ifdef __CADNA_ANALYSIS__
          double holder = static_cast<double>(MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, ip4, ipar ));
#else
          fptype holder = MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, ip4, ipar );
#endif
        soa[(std::size_t)ip4 * npar * nevt + (std::size_t)ipar * nevt + ievt] = static_cast<fptype>(holder);

      }
    }
  }


  inline std::string to_string( fptype A ){
#if defined(MGONGPU_FPTYPE_FLOAT) && defined(__CADNA_ANALYSIS__)
    std::string buf( 64, '\0' );
    str( buf.data(), float_st( A ) );
    buf.resize( std::strlen( buf.data() ) );
    return buf;
#elif defined(__CADNA_ANALYSIS__)
    std::string buf( 64, '\0' );
    str( buf.data(), double_st( A ) );
    buf.resize( std::strlen( buf.data() ) );
    return buf;
#else
    return std::to_string( A );
#endif
  }

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  aosoa_to_umami_kernel( const fptype* aosoa,
                         fptype* soa,
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
    using std::isfinite;
    using std::max;
    using std::min;
    using std::sqrt;

#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

#ifdef __CADNA_ANALYSIS__
  cadna_init(-1);
  double avgMEAccuracy= 0.f;
  int avgMEAccuracy_n = 0;
#endif

#if defined(__CADNA_ANALYSIS__) && defined(MGONGPUCPP_GPUIMPL)
  throw("Cadna GPU analysis not implemented yet.");
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
  unsigned int numvec[5] = { 0, 0, 0, 0, 0 };
  int nnum = 0;
  const auto ndigits = std::numeric_limits<fptype>::digits10;

  for( int argn = 1; argn < argc; ++argn )
  {
    std::string arg = argv[argn];
    if( arg == "--verbose" || arg == "-v" ) verbose = true;
    else if( arg == "--debug" || arg == "-d" ) debug = true;
    else if( arg == "--performance" || arg == "-p" ) perf = true;
    else if( arg == "--json" || arg == "-j" ) json = true;
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
  DeviceBufferBase<fptype> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
  DeviceBufferBase<fptype> devUmamiMEs( nevt );
  std::vector<fptype> hstUmamiMEs( nevt );
#else
  HostBufferRndNumMomenta hstRndmom( nevt );
  HostBufferMomenta hstMomenta( nevt );
  HostBufferWeights hstWeights( nevt );
  std::vector<fptype> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
  std::vector<fptype> umamiMEs( nevt );
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
    std::cerr << "ERROR! umami_initialize failed" << '\n';
    return 2;
  }

  // ---- Per-iteration timings ----
  std::unique_ptr<double[]> genrtimes( new double[niter] );
  std::unique_ptr<double[]> rambtimes( new double[niter] );
  std::unique_ptr<double[]> wavetimes( new double[niter] );

  // ---- Inline event statistics ----
  unsigned int nevtABN = 0;
  unsigned int nevtZERO = 0;
  fptype sumME = 0.;
  fptype sumMEsq = 0.;
  fptype minME = numeric_infinity<fptype>(); 
  fptype maxME = -numeric_infinity<fptype>(); 
  unsigned int nevtALL = 0;

  const int meGeVexponent = -( 2 * CPPProcess::npar - 8 );

  for( unsigned int iiter = 0; iiter < niter; ++iiter )
  {
    // Step 1 - random numbers (always generated on host).
    fptype genrtime = 0;
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
    fptype rambtime = 0;
    timermap.start( "2a RamboIni" );
    prsk->getMomentaInitial();
    rambtime += timermap.stop();
    timermap.start( "2b RamboFin" );
    prsk->getMomentaFinal();
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
    fptype wavetime = 0;
    timermap.start( "3a SigmaKin" );
    UmamiInputKey in_keys[1] = { UMAMI_IN_MOMENTA };
    UmamiOutputKey out_keys[1] = { UMAMI_OUT_MATRIX_ELEMENT };
#ifdef MGONGPUCPP_GPUIMPL
    const void* inputs[1] = { devUmamiMomenta.data() };
    void* outputs[1] = { devUmamiMEs.data() };
#else
    const void* inputs[1] = { umamiMomenta.data() };
    void* outputs[1] = { umamiMEs.data() };
#endif
    UmamiStatus st = umami_matrix_element(
      umami_handle, nevt, nevt, 0, 1, in_keys, inputs, 1, out_keys, outputs );
    if( st != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << '\n';
      umami_free( umami_handle );
      return 3;
    }
    wavetime += timermap.stop();

#ifdef MGONGPUCPP_GPUIMPL
    // Step 3b - copy the matrix elements (and momenta, for verbose) back.
    timermap.start( "3b CpDTHmes" );
    gpuMemcpy( hstUmamiMEs.data(), devUmamiMEs.data(), nevt * sizeof( fptype ), gpuMemcpyDeviceToHost );
    if( verbose ) copyHostFromDevice( hstMomenta, devMomenta );
    wavetime += timermap.stop();
#endif

#ifdef MGONGPUCPP_GPUIMPL
    const fptype* mes = hstUmamiMEs.data();
#else
    const fptype* mes = umamiMEs.data();
#endif

    // Step 4 - update inline statistics + per-event verbose printout.
    timermap.start( "4@ UpdtStat" );
    for( unsigned int ievt = 0; ievt < nevt; ++ievt )
    {
      fptype me = mes[ievt];
      ++nevtALL;
#ifdef __CADNA_ANALYSIS__
      if( !finite( me ) )
#else
      if( !isfinite( me ) )
#endif
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

      std::cout << std::string( SEP79, '*' ) << '\n'
                << "Iteration #" << iiter + 1 << " of " << niter << '\n';
      if( perf ) std::cout << "Wave function time: " << wavetime << '\n';
      for( unsigned int ievt = 0; ievt < nevt; ++ievt )
      {
        std::cout << "Momenta:" << '\n';
        for( int ipar = 0; ipar < CPPProcess::npar; ipar++ )
        {

          std::cout << std::scientific << std::setprecision(ndigits) 
                    << std::setw( 4 ) << ipar + 1
                    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 0, ipar )
                    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 1, ipar )
                    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 2, ipar )
                    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 3, ipar )
                    << '\n'
#ifdef __CADNA_ANALYSIS__
		    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 0, ipar ).nb_significant_digit()
		    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 1, ipar ).nb_significant_digit()
		    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 2, ipar ).nb_significant_digit()
		    << std::setw( ndigits + 8 ) << MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, 3, ipar ).nb_significant_digit()
                    << '\n'
#endif
                    << std::defaultfloat;
        }
        std::cout << std::string( SEP79, '-' ) << '\n'
                  << " Matrix element = " << mes[ievt] 
                  << " GeV^" << meGeVexponent << '\n'
#ifdef __CADNA_ANALYSIS__
                  << " Matrix element number of sig dig = " << mes[ievt].nb_significant_digit()<< " "<< '\n'
     
#endif
                  << std::string( SEP79, '-' ) << '\n';
#ifdef __CADNA_ANALYSIS__
                  avgMEAccuracy += mes[ievt].nb_significant_digit();
                  avgMEAccuracy_n++;
#endif
 
      }
    }
    else if( !( debug || perf ) )
    {
      std::cout << "." << std::flush;
    }
  }


#ifdef __CADNA_ANALYSIS__
  if(verbose)
  {
	std::cout << std::setw( ndigits + 8 ) <<  
	"Average element accuracy = " << avgMEAccuracy/avgMEAccuracy_n << '\n'; 
  }
#endif
  if( !( verbose || debug || perf ) ) std::cout << '\n';

  // ---- summary ----
  timermap.start( "8a CompStat" );
  double sumgtim = 0, sumrtim = 0, sumwtim = 0;
  double minwtim = wavetimes[0], maxwtim = wavetimes[0];
  for( unsigned int i = 0; i < niter; ++i )
  {
    sumgtim += genrtimes[i];
    sumrtim += rambtimes[i];
    sumwtim += wavetimes[i];
    minwtim = min( minwtim, wavetimes[i] );
    maxwtim = max( maxwtim, wavetimes[i] );
  }
  double  meanwtim = sumwtim / niter;

  unsigned int  nevtGood = nevtALL - nevtABN;
  fptype meanME = ( nevtGood > 0 ) ? sumME / static_cast<fptype>(nevtGood) : static_cast<fptype>(0.f);
  fptype varME = ( nevtGood > 0 ) ? sumMEsq / static_cast<fptype>(nevtGood) - meanME * meanME : static_cast<fptype>(0.f);
  fptype stdME = ( varME > 0 ) ? sqrt( varME ) : static_cast<fptype>(0.);

  if( perf )
  {
#ifdef __CUDACC__
    const std::string proc_suffix = "_CUDA";
#elif defined( __HIPCC__ )
    const std::string proc_suffix = "_HIP";
#else
    const std::string proc_suffix = "_CPP";
#endif
    std::cout << std::string( SEP79, '*' ) << '\n'
              << "Process                     = " << XSTRINGIFY( MG_EPOCH_PROCESS_ID ) << proc_suffix
#ifdef MGONGPU_HARDCODE_PARAM
              << " [hardcodePARAM=1]" << '\n'
#else
              << " [hardcodePARAM=0]" << '\n'
#endif
              << "NumBlocksPerGrid            = " << gpublocks << '\n'
              << "NumThreadsPerBlock          = " << gputhreads << '\n'
              << "NumIterations               = " << niter << '\n'
              << std::string( SEP79, '-' ) << '\n'
#if defined MGONGPU_FPTYPE_fptype and defined MGONGPU_FPTYPE2_FLOAT
              << "FP precision                = MIXED (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << '\n'
#elif defined MGONGPU_FPTYPE_fptype
              << "FP precision                = fptype (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << '\n'
#elif defined MGONGPU_FPTYPE_FLOAT
              << "FP precision                = FLOAT (NaN/abnormal=" << nevtABN << ", zero=" << nevtZERO << ")" << '\n'
#endif
              << "Random number generation    = COMMON RANDOM HOST" << '\n'
              << std::string( SEP79, '-' ) << '\n'
              << "NumberOfEntries             = " << niter << '\n'
              << std::scientific
              << "TotalTime[Rnd+Rmb+ME] (123) = ( " << sumgtim + sumrtim + sumwtim << " )  sec" << '\n'
              << "TotalTime[Rambo+ME]    (23) = ( " << sumrtim + sumwtim << " )  sec" << '\n'
              << "TotalTime[RndNumGen]    (1) = ( " << sumgtim << " )  sec" << '\n'
              << "TotalTime[Rambo]        (2) = ( " << sumrtim << " )  sec" << '\n'
              << "TotalTime[MatrixElems]  (3) = ( " << sumwtim << " )  sec" << '\n'
              << "MeanTimeInMatrixElems       = ( " << meanwtim << " )  sec" << '\n'
              << "[Min,Max]TimeInMatrixElems  = [ " << minwtim << " ,  " << maxwtim << " ]  sec" << '\n'
              << std::string( SEP79, '-' ) << '\n'
              << "TotalEventsComputed         = " << nevtALL << '\n'
              << "EvtsPerSec[Rnd+Rmb+ME](123) = ( " << nevtALL / ( sumgtim + sumrtim + sumwtim ) << " )  sec^-1" << '\n'
              << "EvtsPerSec[Rmb+ME]     (23) = ( " << nevtALL / ( sumrtim + sumwtim ) << " )  sec^-1" << '\n'
              << "EvtsPerSec[MatrixElems] (3) = ( " << nevtALL / sumwtim << " )  sec^-1" << '\n'
              << std::defaultfloat
              << std::string( SEP79, '*' ) << '\n'
              << "MeanMatrixElemValue         = ( " << meanME << " +- " << stdME / sqrt( (fptype)std::max( 1u, nevtGood ) )
              << " )  GeV^" << meGeVexponent << '\n'
              << "[Min,Max]MatrixElemValue    = [ " << minME << " ,  " << maxME << " ]  GeV^" << meGeVexponent << '\n'
              << std::string( SEP79, '*' ) << '\n';
    timermap.dump();
    std::cout << std::string( SEP79, '*' ) << '\n';
  }

  // ---- json dump ----
  if( json )
  {
    std::string jsonFileName = "./perf/data/" + to_string( jsondate ) + "-perf-test-run" + to_string( jsonrun ) + ".json";
    std::ifstream fileCheck( jsonFileName );
    bool fileExists = (bool)fileCheck;
    if( fileCheck ) fileCheck.close();
    std::ofstream jsonFile( jsonFileName, std::ios_base::app );
    if( !fileExists )
    {
      jsonFile << "[" << '\n';
    }
    else
    {
      std::string temp = "truncate -s-1 " + jsonFileName;
      if( system( temp.c_str() ) != 0 )
        std::cout << "WARNING! Command '" << temp << "' failed" << '\n';
      jsonFile << ", " << '\n';
    }
    jsonFile << "{" << '\n'
             << "\"NumIterations\": " << niter << ", " << '\n'
             << "\"NumThreadsPerBlock\": " << gputhreads << ", " << '\n'
             << "\"NumBlocksPerGrid\": " << gpublocks << ", " << '\n'
             << "\"TotalTime[Rnd+Rmb+ME] (123)\": \"" << to_string( sumgtim + sumrtim + sumwtim ) << " sec\"," << '\n'
             << "\"TotalTime[Rambo+ME] (23)\": \"" << to_string( sumrtim + sumwtim ) << " sec\"," << '\n'
             << "\"TotalTime[RndNumGen] (1)\": \"" << to_string( sumgtim ) << " sec\"," << '\n'
             << "\"TotalTime[Rambo] (2)\": \"" << to_string( sumrtim ) << " sec\"," << '\n'
             << "\"TotalTime[MatrixElems] (3)\": \"" << to_string( sumwtim ) << " sec\"," << '\n'
             << "\"MeanTimeInMatrixElems\": \"" << to_string( meanwtim ) << " sec\"," << '\n'
             << "\"MinTimeInMatrixElems\": \"" << to_string( minwtim ) << " sec\"," << '\n'
             << "\"MaxTimeInMatrixElems\": \"" << to_string( maxwtim ) << " sec\"," << '\n'
             << "\"TotalEventsComputed\": " << nevtALL << "," << '\n'
             << "\"EvtsPerSec[Rnd+Rmb+ME](123)\": \"" << to_string( nevtALL / ( sumgtim + sumrtim + sumwtim ) ) << " sec^-1\"," << '\n'
             << "\"EvtsPerSec[Rmb+ME] (23)\": \"" << to_string( nevtALL / ( sumrtim + sumwtim ) ) << " sec^-1\"," << '\n'
             << "\"EvtsPerSec[MatrixElems] (3)\": \"" << to_string( nevtALL / sumwtim ) << " sec^-1\"," << '\n'
             << "\"NumMatrixElems(notAbnormal)\": " << nevtGood << "," << '\n'
             << "\"MeanMatrixElemValue\": \"" << to_string( meanME ) << " GeV^" << to_string( meGeVexponent ) << "\"," << '\n'
             << "\"StdDevMatrixElemValue\": \"" << to_string( stdME ) << " GeV^" << to_string( meGeVexponent ) << "\"," << '\n'
             << "\"MinMatrixElemValue\": \"" << to_string( minME ) << " GeV^" << to_string( meGeVexponent ) << "\"," << '\n'
             << "\"MaxMatrixElemValue\": \"" << to_string( maxME ) << " GeV^" << to_string( meGeVexponent ) << "\"" << '\n'
             << "}" << '\n'
             << "]";
    jsonFile.close();
  }

  umami_free( umami_handle );

#ifdef __CADNA_ANALYSIS__
  cadna_end();
#endif

  return 0;
}
