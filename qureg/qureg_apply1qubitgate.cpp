//------------------------------------------------------------------------------
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

#include "qureg.hpp"
#include "highperfkernels.hpp"

/// \addtogroup qureg
/// @{

/// @file qureg_apply1qubitgate.cpp
/// @brief Define the @c QubitRegister methods corresponding to the application of single-qubit gates.

//Ryan: Primary gate computation function
/////////////////////////////////////////////////////////////////////////////////////////
template <class Type>
double QubitRegister<Type>::SZ_HP_Distrpair(unsigned position, TM2x2<Type> const&m)
{
#ifdef INTELQS_HAS_MPI
  MPI_Status status;
  MPI_Comm comm = openqu::mpi::Environment::comm();
  std::size_t myrank = openqu::mpi::Environment::rank();

  assert(position < num_qubits);
  int strideexp = position;
  int memexp = num_qubits - openqu::ilog2(openqu::mpi::Environment::size());
  int pstrideexp = strideexp - memexp;

  //  Steps:     1.         2.           3.              4.
  //          i    j   | i    j   |  i      j     |  i       j
  //          s1   d1  | s1   d1  |  s1     d1&s1 |  s1&d1   d1&s1
  //          s2   d2  | s2   d2  |  s2&d2  d2    |  s2&d2   d2&s2
  //          T    T   | d2   s1  |  d2&s2  s1&d1 |  T       T

  int tag1 = 1, tag2 = 2;
  int tag3 = 3, tag4 = 4;
  int tag5 = 5, tag6 = 6;
  int tag7 = 7, tag8 = 8;
  
  std::size_t glb_start = UL(myrank) * LocalSize();

  // std::string s;
  unsigned int itask, jtask;
  if (check_bit(glb_start, UL(position)) == 0)
  {
      itask = myrank;
      jtask = itask + (1 << pstrideexp);
      // s = openqu::toString(itask) + "==>" + openqu::toString(jtask);
  }
  else
  {
      jtask = myrank;
      itask = jtask - (1 << pstrideexp);
      // s = openqu::toString(jtask) + "==>" + openqu::toString(itask);
  }


  // 1. allocate temp buffer
  //tmp buffer is now allocated in BlkDecompress function 

  std::size_t lcl_size_half = num_block / 2L;

  double t, tnet = 0;
  int len_count = 1;
  int cache_status;
  for(size_t c = 0; c < lcl_size_half; c++)
  {
    // if(!myrank) printf("c=%lu lcl_size_half=%lu lcl_chujnk=%lu\n", c, lcl_size_half, lcl_chunk);
    if (itask == myrank)  // this is itask
    {
        // 2. src sends s1 to dst into dT
        //    dst sends d2 to src into dT
        t = sec();
        tmp_len = 0;
        MPI_Sendrecv_x(&(list_len[c]), len_count, jtask, tag1, &tmp_len,
                       len_count, jtask, tag2, comm, &status);
        if(tmp_compressed_blk != NULL) free(tmp_compressed_blk);
        tmp_compressed_blk = (unsigned char*) malloc(tmp_len);
        MPI_Sendrecv_x(list_compressed_blk[c], list_len[c], jtask, tag3, tmp_compressed_blk,
                       tmp_len, jtask, tag4, comm, &status);
        tnet += sec() - t;
        network_time += sec() - t;

        cache_status = CheckCache(lcl_size_half + c, list_compressed_blk[lcl_size_half + c], list_len[lcl_size_half + c], -1, tmp_compressed_blk, tmp_len);
        if (cache_status != 0){

          t = sec();
          DecompressTmpState();
          DecompressLclState(lcl_size_half + c);
          decompression_time += sec() - t;
  
          // 3. src and dst compute
          t = sec();
          Loop_SN(0L, block_size, state, tmp_state, 0L, 0L, m, specialize, timer);
          compute_time += sec() - t;
  
          t = sec();
          CompressLclState(lcl_size_half + c);
          CompressTmpState();
          compression_time += sec() - t;

          if (cache_status == 1){
            SetCache(lcl_size_half + c, list_compressed_blk[lcl_size_half + c], list_len[lcl_size_half + c], -1, tmp_compressed_blk, tmp_len);
          }
        }

        t = sec();
        MPI_Sendrecv_x(&tmp_len, len_count, jtask, tag5, &(list_len[c]),
                       len_count, jtask, tag6, comm, &status);
        if (list_compressed_blk[c] != NULL) free(list_compressed_blk[c]);
        list_compressed_blk[c] = (unsigned char*) malloc(list_len[c]);
        MPI_Sendrecv_x(tmp_compressed_blk, tmp_len, jtask, tag7, list_compressed_blk[c],
                       list_len[c], jtask, tag8, comm, &status);
        tnet += sec() - t;
        network_time += sec() - t;
    }
    else  // this is jtask
    {
        // 2. src sends s1 to dst into dT
        //    dst sends d2 to src into dT
        t = sec();
        tmp_len = 0;
        MPI_Sendrecv_x(&(list_len[lcl_size_half + c]), len_count, itask, tag2, &tmp_len,
                       len_count, itask, tag1, comm, &status);
        if(tmp_compressed_blk != NULL) free(tmp_compressed_blk);
        tmp_compressed_blk = (unsigned char*) malloc(tmp_len);
        MPI_Sendrecv_x(list_compressed_blk[lcl_size_half + c], list_len[lcl_size_half + c], itask, tag4, tmp_compressed_blk,
                       tmp_len, itask, tag3, comm, &status);
        network_time += sec() - t;
        tnet += sec() - t;

        cache_status = CheckCache(-1, tmp_compressed_blk, tmp_len, c, list_compressed_blk[c], list_len[c]);
        if (cache_status != 0){
          t = sec();
          DecompressTmpState();
          DecompressLclState(c);
          decompression_time += sec() - t;
  
          t = sec();
          Loop_SN(0L, block_size, tmp_state, state, 0L, 0L, m, specialize, timer);
          compute_time += sec() - t;
  
          t = sec();
          CompressLclState(c);
          CompressTmpState();
          compression_time += sec() - t;
          if (cache_status == 1){
            SetCache(-1, tmp_compressed_blk, tmp_len, c, list_compressed_blk[c], list_len[c]);
          }
        }

        t = sec();
        MPI_Sendrecv_x(&tmp_len, len_count, itask, tag6, &(list_len[lcl_size_half + c]),
                       len_count, itask, tag5, comm, &status);
        if (list_compressed_blk[lcl_size_half + c] != NULL) free(list_compressed_blk[lcl_size_half + c]);
        list_compressed_blk[lcl_size_half + c] = (unsigned char*) malloc(list_len[lcl_size_half + c]);
        MPI_Sendrecv_x(tmp_compressed_blk, tmp_len, itask, tag8, list_compressed_blk[lcl_size_half + c],
                       list_len[lcl_size_half + c], itask, tag7, comm, &status);
        tnet += sec() - t;
        network_time += sec() - t;
    }
  }

  double netsize = 2.0 * sizeof(Type) * 2.0 * D(lcl_size_half), netbw = netsize / tnet;
  // printf("[%3d] size=%10lld tnet = %.3lf s netsize = %10.0lf bytes netbw = %6.2lf GB/s\n",
  //      it, sizeof(Type)*lcl_size_half, tnet, netsize, netbw / 1e9);

  if (timer) {timer->record_cm(tnet, netbw); };
#else
  assert(0);
#endif
  return 0.0;
}

template <class Type>
double QubitRegister<Type>::HP_Distrpair(unsigned position, TM2x2<Type> const&m)
{
#ifdef INTELQS_HAS_MPI
  MPI_Status status;
  MPI_Comm comm = openqu::mpi::Environment::comm();
  std::size_t myrank = openqu::mpi::Environment::rank();

  assert(position < num_qubits);
  int strideexp = position;
  int memexp = num_qubits - openqu::ilog2(openqu::mpi::Environment::size());
  int pstrideexp = strideexp - memexp;

  //  Steps:     1.         2.           3.              4.
  //          i    j   | i    j   |  i      j     |  i       j
  //          s1   d1  | s1   d1  |  s1     d1&s1 |  s1&d1   d1&s1
  //          s2   d2  | s2   d2  |  s2&d2  d2    |  s2&d2   d2&s2
  //          T    T   | d2   s1  |  d2&s2  s1&d1 |  T       T

  int tag1 = 1, tag2 = 2;
  int tag3 = 3, tag4 = 4;
  std::size_t glb_start = UL(myrank) * LocalSize();

  // std::string s;
  unsigned int itask, jtask;
  if (check_bit(glb_start, UL(position)) == 0)
  {
      itask = myrank;
      jtask = itask + (1 << pstrideexp);
      // s = openqu::toString(itask) + "==>" + openqu::toString(jtask);
  }
  else
  {
      jtask = myrank;
      itask = jtask - (1 << pstrideexp);
      // s = openqu::toString(jtask) + "==>" + openqu::toString(itask);
  }

  // openqu::mpi::print(s, true);

  if (specialize == true)
  { 
      // check for special case of diagonal
      bool Xgate = (m[0][0] == Type(0., 0.) && m[0][1] == Type(1., 0.) &&
                    m[1][0] == Type(1., 0.) && m[1][1] == Type(0., 0.));
      if (Xgate == true)
      {
          // printf("Xgate: remaping MPI rank %d <==> %d\n", jtask, itask);
          if (check_bit(glb_start, UL(position)) == 0)
              openqu::mpi::Environment::remaprank(jtask);
          else
              openqu::mpi::Environment::remaprank(itask);
          TODO(Fix problem when coming here from controlled gate)
          openqu::mpi::barrier();
          if (timer)
              timer->record_cm(0., 0.);
          return 0.0;
      }
    bool Ygate = (m[0][0] == Type(0., 0.) && m[0][1] == Type(0., -1.) &&
                  m[1][0] == Type(0., 1.) && m[1][1] == Type(0., 0.));
    if (Ygate == true)
    {
      // printf("Ygate: remaping MPI rank\n");
      if (check_bit(glb_start, UL(position)) == 0)
      {
          openqu::mpi::Environment::remaprank(jtask);
          ScaleState(0UL, LocalSize(), state, Type(0, 1.0), timer);
      }
      else
      {
          openqu::mpi::Environment::remaprank(itask);
          ScaleState(0UL, LocalSize(), state, Type(0, -1.0), timer);
      }
      openqu::mpi::barrier();
      if (timer)
          timer->record_cm(0., 0.);
      return 0.0;
    }

  }


  // 1. allocate temp buffer
  size_t lcl_chunk = TmpSize();
  Type *tmp_state = TmpSpace();

  std::size_t lcl_size_half = LocalSize() / 2L;
  assert(lcl_size_half <= std::numeric_limits<size_t>::max());

  if (lcl_chunk > lcl_size_half) 
      lcl_chunk = lcl_size_half;
  else
      assert((lcl_size_half % lcl_chunk) == 0);
  
  double t, tnet = 0;
  for(size_t c = 0; c < lcl_size_half; c += lcl_chunk)
  {
    // if(!myrank) printf("c=%lu lcl_size_half=%lu lcl_chujnk=%lu\n", c, lcl_size_half, lcl_chunk);
    if (itask == myrank)  // this is itask
    {
        // 2. src sends s1 to dst into dT
        //    dst sends d2 to src into dT
        t = sec();
        MPI_Sendrecv_x(&(state[c]), lcl_chunk, jtask, tag1, &(tmp_state[0]),
                       lcl_chunk, jtask, tag2, comm, &status);
        tnet += sec() - t;

        // 3. src and dst compute
        Loop_SN(0L, lcl_chunk, &(state[c]), tmp_state, lcl_size_half, 0L, m, specialize, timer);

        t = sec();
        MPI_Sendrecv_x(&(tmp_state[0]), lcl_chunk, jtask, tag3, &(state[c]),
                       lcl_chunk, jtask, tag4, comm, &status);
        tnet += sec() - t;
    }
    else  // this is jtask
    {
        // 2. src sends s1 to dst into dT
        //    dst sends d2 to src into dT
        t = sec();
        MPI_Sendrecv_x(&(state[lcl_size_half + c]), lcl_chunk, itask, tag2,
                       &(tmp_state[0]), lcl_chunk, itask, tag1, comm,
                       &status);
        tnet += sec() - t;

        Loop_SN(0L, lcl_chunk, tmp_state, &(state[c]), 0L, 0L, m, specialize, timer);

        t = sec();
        MPI_Sendrecv_x(&(tmp_state[0]), lcl_chunk, itask, tag4,
                       &(state[lcl_size_half + c]), lcl_chunk, itask, tag3,
                       comm, &status);
        tnet += sec() - t;
    }
  }

  double netsize = 2.0 * sizeof(Type) * 2.0 * D(lcl_size_half), netbw = netsize / tnet;
  // printf("[%3d] size=%10lld tnet = %.3lf s netsize = %10.0lf bytes netbw = %6.2lf GB/s\n",
  //      it, sizeof(Type)*lcl_size_half, tnet, netsize, netbw / 1e9);

  if (timer) {timer->record_cm(tnet, netbw); };

#else
  assert(0);
#endif
  return 0.0;
}


/////////////////////////////////////////////////////////////////////////////////////////
template <class Type>
bool QubitRegister<Type>::Apply1QubitGate_helper(unsigned qubit_,  TM2x2<Type> const&m, 
                                                 std::size_t sind, std::size_t eind)
{
  assert(qubit_ < num_qubits);
  unsigned qubit = (*permutation)[qubit_]; 
  assert(qubit < num_qubits);
  double t;

  TODO(Add diagonal special case)

  unsigned myrank=0, nprocs=1, log2_nprocs=0;
#ifdef INTELQS_HAS_MPI
  myrank = openqu::mpi::Environment::rank();
  nprocs = openqu::mpi::Environment::size();
  log2_nprocs = openqu::ilog2(openqu::mpi::Environment::size());
#endif
  unsigned M = num_qubits - log2_nprocs;
  unsigned B = (unsigned)std::log2(block_size);
  std::size_t P = qubit;

  std::size_t src_glb_start = UL(myrank) * LocalSize();
  // check for special case of diagonal
  bool diagonal = (m[0][1].real() == 0. && m[0][1].imag() == 0. &&
                   m[1][0].real() == 0. && m[1][0].imag() == 0.);

  std::string gate_name = "SQG("+openqu::toString(P)+")::"+m.name;

  if (timer)
      timer->Start(gate_name, P);

  int cache_status;
  if (P < M)
  {
      assert(eind - sind <= LocalSize());
      //Loop_DN(sind, eind, UL(P), state, state, 0UL, (1UL << P), m, specialize, timer);
      if (P < B){
        for (int i = 0; i < num_block; i++){
          cache_status = CheckCache(i, list_compressed_blk[i], list_len[i]);
          if (cache_status != 0){
            t = sec();
            DecompressLclState(i);
            decompression_time += sec() - t;

            t = sec();
            Loop_DN(0, block_size, UL(P), state, state, 0UL, (1UL << P), m, specialize, timer);
            compute_time += sec() - t;

            t = sec();
            CompressLclState(i);
            compression_time += sec() - t;
            if (cache_status == 1){
              SetCache(i, list_compressed_blk[i], list_len[i]);
            }
          }
        }
      } else {
        unsigned int p_dist = P - B;
        int idx_i, idx_j;
        for (idx_i = 0; idx_i < num_block; idx_i = idx_i + (1 << (p_dist + 1))){
          for (idx_j = 0; idx_j < (1 << p_dist); idx_j++){
            int a = idx_i + idx_j;
            int b = idx_i + idx_j + (1 << p_dist);
            cache_status = CheckCache(a, list_compressed_blk[a], list_len[a], b, list_compressed_blk[b], list_len[b]);
            if (cache_status != 0){
              if (tmp_compressed_blk != NULL) free(tmp_compressed_blk);
              tmp_compressed_blk = list_compressed_blk[b];
              list_compressed_blk[b] = NULL;
              tmp_len = list_len[b];
              t = sec();
              DecompressTmpState();
              DecompressLclState(a);
              decompression_time += sec() - t;
              t = sec();
              Loop_SN(0L, block_size, state, tmp_state, 0L, 0L, m, specialize, timer);
              compute_time += sec() - t;
              t = sec();
              CompressLclState(a);
              CompressTmpState();
              compression_time += sec() - t;
              if (list_compressed_blk[b] != NULL) free(list_compressed_blk[b]);
              list_compressed_blk[b] = tmp_compressed_blk;
              tmp_compressed_blk = NULL;
              list_len[b] = tmp_len;
              if (cache_status == 1){
                SetCache(a, list_compressed_blk[a], list_len[a], b, list_compressed_blk[b], list_len[b]);
              }
            }
          }
        }
      }
  }
  else
  {
      assert(eind - sind == LocalSize());
      if (specialize && diagonal)
      {
          if (check_bit(src_glb_start, P) == 0 )
              ScaleState(sind, eind, state, m[0][0], timer);
          else
              ScaleState(sind, eind, state, m[1][1], timer);
      }
      else
      {
          SZ_HP_Distrpair(P, m);
      }
  }

  if (timer)
      timer->Stop();
  
  return true;
}


/////////////////////////////////////////////////////////////////////////////////////////
template <class Type>
void QubitRegister<Type>::Apply1QubitGate(unsigned qubit, TM2x2<Type> const&m)
{
  if (fusion == true)
  {
      assert((*permutation)[qubit] < num_qubits);
      if ((*permutation)[qubit] < log2llc)
      {
          std::string name = "sqg";
          fwindow.push_back(std::make_tuple(name, m, qubit, 0U));
          return;
      }
      else
      {
          ApplyFusedGates();
          goto L;
      }
  }

  L:
  Apply1QubitGate_helper(qubit, m, 0UL, LocalSize());
  if (enable_blk_cache == 1){
    ClearCache();
  }
  CompressionInfo();
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Rotation around the X axis by an angle theta
/// @param qubit index of the involved qubit
/// @param theta rotation angle
///
/// Explicitely, the gate corresponds to:\n
///     exp( -i X theta/2 )\n
/// This convention is based on the fact that the generators
/// of rotations for spin-1/2 spins are {X/2, Y/2, Z/2}.
template <class Type>
void QubitRegister<Type>::ApplyRotationX(unsigned const qubit, BaseType theta)
{
  openqu::TinyMatrix<Type, 2, 2, 32> rx;
  rx(0, 1) = rx(1, 0) = Type(0, -std::sin(theta / 2.));
  rx(0, 0) = rx(1, 1) = std::cos(theta / 2.);
  Apply1QubitGate(qubit, rx);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Rotation around the Y axis by an angle theta
/// @param qubit index of the involved qubit
/// @param theta rotation angle
///
/// Explicitely, the gate corresponds to:\n
///     exp( -i Y theta/2 )\n
/// This convention is based on the fact that the generators
/// of rotations for spin-1/2 spins are {X/2, Y/2, Z/2}.
template <class Type>
void QubitRegister<Type>::ApplyRotationY(unsigned const qubit, BaseType theta)
{
  openqu::TinyMatrix<Type, 2, 2, 32> ry;
  ry(0, 1) = Type(-std::sin(theta / 2.), 0.);
  ry(1, 0) = Type( std::sin(theta / 2.), 0.);
  ry(0, 0) = ry(1, 1) = std::cos(theta / 2.);
  Apply1QubitGate(qubit, ry);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Rotation around the Z axis by an angle theta
/// @param qubit index of the involved qubit
/// @param theta rotation angle
///
/// Explicitely, the gate corresponds to:\n
///     exp( -i Z theta/2 )\n
/// This convention is based on the fact that the generators
/// of rotations for spin-1/2 spins are {X/2, Y/2, Z/2}.
template <class Type>
void QubitRegister<Type>::ApplyRotationZ(unsigned const qubit, BaseType theta)
{
  openqu::TinyMatrix<Type, 2, 2, 32> rz;
  rz(0, 0) = Type(std::cos(theta / 2.), -std::sin(theta / 2.));
  rz(1, 1) = Type(std::cos(theta / 2.), std::sin(theta / 2.));
  rz(0, 1) = rz(1, 0) = Type(0., 0.);
  Apply1QubitGate(qubit, rz);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply X Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     i * exp( -i X pi/2 ) = X 
template <class Type>
void QubitRegister<Type>::ApplyPauliX(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> px;
  px(0, 0) = Type(0., 0.);
  px(0, 1) = Type(1., 0.);
  px(1, 0) = Type(1., 0.);
  px(1, 1) = Type(0., 0.);
  Apply1QubitGate(qubit, px);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply square root of the X Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     sqrt(X) 
template <class Type>
void QubitRegister<Type>::ApplyPauliSqrtX(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> px;
  px(0, 0) = Type(0.5,  0.5);
  px(0, 1) = Type(0.5, -0.5);
  px(1, 0) = Type(0.5, -0.5);
  px(1, 1) = Type(0.5,  0.5);
  Apply1QubitGate(qubit, px);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply Y Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     i * exp( -i Y pi/2 ) = Y 
template <class Type>
void QubitRegister<Type>::ApplyPauliY(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> py;
  py(0, 0) = Type(0., 0.);
  py(0, 1) = Type(0., -1.);
  py(1, 0) = Type(0., 1.);
  py(1, 1) = Type(0., 0.);
  Apply1QubitGate(qubit, py);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply square root of the Y Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     sqrt(Y) 
template <class Type>
void QubitRegister<Type>::ApplyPauliSqrtY(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> py;
  py(0, 0) = Type(0.5,   0.5);
  py(0, 1) = Type(-0.5, -0.5);
  py(1, 0) = Type(0.5,   0.5);
  py(1, 1) = Type(0.5,  0.5);
  Apply1QubitGate(qubit, py);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply Z Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     i * exp( -i Z pi/2 ) = Z 
template <class Type>
void QubitRegister<Type>::ApplyPauliZ(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> pz;
  pz(0, 0) = Type(1., 0.);
  pz(0, 1) = Type(0., 0.);
  pz(1, 0) = Type(0., 0.);
  pz(1, 1) = Type(-1., 0.);
  Apply1QubitGate(qubit, pz);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply square root of the Z Pauli operator
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to:\n
///     sqrt(Z) 
template <class Type>
void QubitRegister<Type>::ApplyPauliSqrtZ(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> pz;
  pz(0, 0) = Type(1., 0.);
  pz(0, 1) = Type(0., 0.);
  pz(1, 0) = Type(0., 0.);
  pz(1, 1) = Type(0., 1.);
  Apply1QubitGate(qubit, pz);
}



/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply Hadamard gate
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to the 2x2 matrix:\n
///     | 1/sqrt(2)   1/sqrt(2) |\n
///     | 1/sqrt(2)  -1/sqrt(2) |
template <class Type>
void QubitRegister<Type>::ApplyHadamard(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> h;
  BaseType f = 1. / std::sqrt(2.);
  h(0, 0) = h(0, 1) = h(1, 0) = Type(f, 0.);
  h(1, 1) = Type(-f, 0.);
  Apply1QubitGate(qubit, h);
}


/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply T gate
/// @param qubit index of the involved qubit
///
/// Explicitely, the gate corresponds to the 2x2 matrix:\n
///     | 1              0           |\n
///     | 0    cos(pi/4)+i*sin(pi/4) |
template <class Type>
void QubitRegister<Type>::ApplyT(unsigned const qubit)
{
  openqu::TinyMatrix<Type, 2, 2, 32> t;
  t(0, 0) = Type(1.0, 0.0);
  t(0, 1) = Type(0.0, 0.0);
  t(1, 0) = Type(0.0, 0.0);
  t(1, 1) = Type(cos(M_PI/4.0), sin(M_PI/4.0));
  Apply1QubitGate(qubit, t);

}

template class QubitRegister<ComplexSP>;
template class QubitRegister<ComplexDP>;

/// @}
