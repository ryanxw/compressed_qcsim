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

#include <vector>

using namespace std;

template <class Type = ComplexDP>
class QubitRegisterMetric: public QubitRegister<Type> {
  int iTotalQubitGateCount=0;
  int iOneQubitGateCount=0;
  int iTwoQubitGateCount=0;
  std::vector<int> vParallelDepth;
  void OneQubitIncrements(int);
  void TwoQubitIncrements(int,int);

public:
  //Constructor
  QubitRegisterMetric<Type>(int iNQubits):QubitRegister<Type>(iNQubits){
    vParallelDepth.resize(iNQubits);
  }

  //Get stats
  int GetTotalQubitGateCount();
  int GetOneQubitGateCount();
  int GetTwoQubitGateCount();
  int GetParallelDepth();

  //Perform gates
  void ApplyHadamard(int);
  void ApplyRotationX(int, double);
  void ApplyRotationY(int, double);
  void ApplyRotationZ(int, double);
  void ApplyCPauliX(int, int);
  void ApplyControlled1QubitGate(int, int, openqu::TinyMatrix<Type, 2, 2, 32>);
};

template <class Type>
int QubitRegisterMetric<Type>::GetOneQubitGateCount(){
  return iOneQubitGateCount;
}

template <class Type>
int QubitRegisterMetric<Type>::GetTwoQubitGateCount(){
  return iTwoQubitGateCount;
}

template <class Type>
int QubitRegisterMetric<Type>::GetTotalQubitGateCount(){
  return iTotalQubitGateCount;
}

template <class Type>

int QubitRegisterMetric<Type>::GetParallelDepth(){
  return *std::max_element(std::begin(vParallelDepth), std::end(vParallelDepth));
}

template <class Type>
void QubitRegisterMetric<Type>::OneQubitIncrements(int q){
  iTotalQubitGateCount++;
  iOneQubitGateCount++;
  vParallelDepth[q]++;
}

template <class Type>
void QubitRegisterMetric<Type>::TwoQubitIncrements(int q1, int q2){
  iTotalQubitGateCount++;
  iTwoQubitGateCount++;
  int iNewDepth = max(vParallelDepth[q1],vParallelDepth[q2])+1;
  vParallelDepth[q1]=iNewDepth;
  vParallelDepth[q2]=iNewDepth;
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyHadamard(int q){
  QubitRegister<Type>::ApplyHadamard(q);
  OneQubitIncrements(q); 
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyRotationX(int q, double theta){
  QubitRegister<Type>::ApplyRotationX(q,theta);
  OneQubitIncrements(q); 
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyRotationY(int q, double theta){
  QubitRegister<Type>::ApplyRotationY(q,theta);
  OneQubitIncrements(q); 
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyRotationZ(int q, double theta){
  QubitRegister<Type>::ApplyRotationZ(q,theta);
  OneQubitIncrements(q); 
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyCPauliX(int q1, int q2){
  QubitRegister<Type>::ApplyCPauliX(q1,q2);
  TwoQubitIncrements(q1,q2); 
}

template <class Type>
void QubitRegisterMetric<Type>::ApplyControlled1QubitGate(int q1, int q2, openqu::TinyMatrix<Type, 2, 2, 32> V){
  QubitRegister<Type>::ApplyControlled1QubitGate(q1,q2,V);
  TwoQubitIncrements(q1,q2); 
}
