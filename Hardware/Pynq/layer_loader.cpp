/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

#include <map>
#include <vector>
#include <string.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" { 
#include <libxlnk_cma.h>
}
using namespace std;

typedef uint32_t AccelReg;
typedef uint64_t AccelDblReg;
typedef unsigned long long ExtMemWord;

class DonutDriver
{
public:
  DonutDriver() { m_numSysRegs = 32; }
  virtual ~DonutDriver() {}
  // (optional) functions for host-accelerator buffer management
  virtual void copyBufferHostToAccel(void * hostBuffer, void * accelBuffer, unsigned int numBytes) {}
  virtual void copyBufferAccelToHost(void * accelBuffer, void * hostBuffer, unsigned int numBytes) {}
  virtual void * allocAccelBuffer(unsigned int numBytes) {return 0;}
  virtual void deallocAccelBuffer(void * buffer) {}

  // (optional) functions for accelerator attach-detach handling
  virtual void attach(const char * name) {}
  virtual void detach() {}

  // convenience functions to access platform or jam registers
  // access by register index (0, 1, 2...)
  AccelReg readJamRegInd(unsigned int regInd) {
    return readRegAtAddr((m_numSysRegs + regInd) * sizeof(AccelReg));
  }

  void writeJamRegInd(unsigned int regInd, AccelReg value) {
    writeRegAtAddr((m_numSysRegs + regInd) * sizeof(AccelReg), value);
  }

  AccelReg readSysRegInd(unsigned int regInd) {
    return readRegAtAddr((regInd) * sizeof(AccelReg));
  }

  void writeSysRegInd(unsigned int regInd, AccelReg value) {
    writeRegAtAddr((regInd) * sizeof(AccelReg), value);
  }
  // access by register address (0, 4, 8...)
  AccelReg readJamRegAddr(unsigned int addr) {
    return readRegAtAddr(m_numSysRegs * sizeof(AccelReg) + addr);
  }

  void writeJamRegAddr(unsigned int addr, AccelReg value) {
    writeRegAtAddr(m_numSysRegs * sizeof(AccelReg) + addr, value);
  }

  AccelReg readSysRegAddr(unsigned int addr) {
    return readRegAtAddr(addr);
  }

  void writeSysRegAddr(unsigned int addr, AccelReg value) {
    writeRegAtAddr(addr, value);
  }

  // convenience functions to read/write 64-bit values to/from the jam
  // since each register is 32 bits, this requires two register accesses
  // it is assumed that these two registers' addresses are contiguous,
  // and that bits 31..0 live in the first reg, bits 63..32 in the second reg
  AccelDblReg read64BitJamRegAddr(unsigned int addr) {
    AccelDblReg ret = 0;
    ret = readJamRegAddr(addr+4);
    ret = ret << 32;
    ret = ret | readJamRegAddr(addr);
    return ret;
  }

  void write64BitJamRegAddr(unsigned int addr, AccelDblReg value) {
    writeJamRegAddr(addr, value & 0xffffffff);
    writeJamRegAddr(addr+4, (value >> 32) & 0xffffffff);
  }

protected:
  unsigned int m_numSysRegs;

  // (mandatory) register access methods for the platform wrapper
  virtual void writeRegAtAddr(unsigned int addr, AccelReg regValue) = 0;
  virtual AccelReg readRegAtAddr(unsigned int addr) = 0;

};

class XlnkDriver : public DonutDriver
{
 public:
  XlnkDriver(uint32_t regBase, unsigned int regSize): m_regSize(regSize) {
    m_reg = reinterpret_cast<AccelReg*>(cma_mmap(regBase, regSize));
    if (!m_reg) {
      cout << "Failed to allocate registers";
	}
  }

  virtual ~XlnkDriver() {
    for (PhysMap::iterator iter = m_physmap.begin(); iter != m_physmap.end(); ++iter) {
      cma_free(iter->second);
    }
    cma_munmap(m_reg, m_regSize);
  }

  virtual void copyBufferHostToAccel(void* hostBuffer, void* accelBuffer, unsigned int numBytes) {
    PhysMap::iterator iter = m_physmap.find(accelBuffer);
    if (iter == m_physmap.end()) {
      cout << "Invalid buffer specified";
    }
    void* virt = iter->second;
    std::memcpy(virt, hostBuffer, numBytes);
  }

  virtual void copyBufferAccelToHost(void* accelBuffer, void* hostBuffer, unsigned int numBytes) {
    PhysMap::iterator iter = m_physmap.find(accelBuffer);
    if (iter == m_physmap.end()) {
      cout << "Invalid buffer specified";
    }
    void* virt = iter->second;
    std::memcpy(hostBuffer, virt, numBytes);
  }

  virtual void* allocAccelBuffer(unsigned int numBytes) {
    void* virt = cma_alloc(numBytes, false);
    if (!virt) {
      return 0;
	}
	void* phys = reinterpret_cast<void*>(cma_get_phy_addr(virt));
	m_physmap.insert(std::make_pair(phys, virt));
	return phys;
  }

  virtual void deallocAccelBuffer(void* buffer) {
	PhysMap::iterator iter = m_physmap.find(buffer);
	if (iter == m_physmap.end()) {
	  cout << "Invalid pointer freed";
	}
	cma_free(iter->second);
	m_physmap.erase(iter);
  }

 protected:
  virtual void writeRegAtAddr(unsigned int addr, AccelReg regValue) {
	if (addr & 0x3) {
		cout << "Unaligned register write";
	}
	m_reg[addr >> 2] = regValue;
  }

  virtual AccelReg readRegAtAddr(unsigned int addr) {
	if (addr & 0x3) cout << "Unaligned register read";
	return m_reg[addr >> 2];
  }

 private:
  typedef std::map<void*, void*> PhysMap;
  PhysMap m_physmap;
  AccelReg* m_reg;
  uint32_t m_regSize;	
};

static XlnkDriver* platform = 0;

void platformSIGINTHandler(int signum) {
  std::cout << "Caught SIGINT, forcing exit" << std::endl;
  if(platform) {
    platform->detach();
  }
  delete platform;
  exit(1);
}

DonutDriver* initPlatform(bool cleanSIGINTExit, unsigned int addr) {
  if (!platform) {
    platform = new XlnkDriver(addr, 64 * 1024);
  }
  if (cleanSIGINTExit) {
    struct sigaction action;
    std::memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = &platformSIGINTHandler;
    int res = sigaction(SIGINT, &action, NULL);
  }
  return static_cast<DonutDriver*>(platform);
}

void deinitPlatform(DonutDriver* driver) {
  delete platform;
  platform = 0;
}


DonutDriver *thePlatform = 0;


void FoldedMVInit(const char * attachName, unsigned int addr) {
  thePlatform = initPlatform(true, addr);
}

void FoldedMVDeinit() {
  deinitPlatform(thePlatform);
  thePlatform = 0;
}

void ExecAccel() {
  // invoke accelerator and wait for result
  thePlatform->writeJamRegAddr(0x00, 1);
  while((thePlatform->readJamRegAddr(0x00) & 0x2) == 0);
}

// TODO this variant always assumes an 8 byte val port on the accelerator
void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ExtMemWord val) {
  // enable weight loading mode
  thePlatform->writeJamRegAddr(0x28, 1);
  // set up init data
  thePlatform->writeJamRegAddr(0x30, targetLayer);
  thePlatform->writeJamRegAddr(0x38, targetMem);
  thePlatform->writeJamRegAddr(0x40, targetInd);
  thePlatform->writeJamRegAddr(0x48, targetThresh);
  thePlatform->write64BitJamRegAddr(0x50, (AccelDblReg) val);
  // do write
  ExecAccel();
  // disable weight loading mode
  thePlatform->writeJamRegAddr(0x28, 0);
}

void FoldedMVLoadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem, unsigned int cntThresh) {
  for(unsigned int pe = 0; pe < peCount; pe++) {
    // load weights
    ifstream wf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-weights.bin", ios::binary | ios::in);
    if(!wf.is_open()) {
      cout << "Could not open file";
    }
    for(unsigned int line = 0 ; line < linesWMem; line++) {
      ExtMemWord e = 0;
      wf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo * 2, pe, line, 0, e);
    }
    wf.close();

    // load thresholds
    if(cntThresh > 0) {
      ifstream tf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-thres.bin", ios::binary | ios::in);
      if(!tf.is_open())
        cout << "Could not open file";
      for(unsigned int line = 0 ; line < linesTMem; line++) {
        for(unsigned int i = 0; i < cntThresh; i++){
        ExtMemWord e = 0;
          tf.read((char *)&e, sizeof(ExtMemWord));
          FoldedMVMemSet(layerNo * 2 + 1, pe, line,i, e);
        }
      }
      tf.close();
    }
  }
}

extern "C" void load_layer(const char* path, unsigned int layer, unsigned int PEs, unsigned int Wtiles, 
  unsigned int Ttiles, unsigned int API, unsigned int addr) {
  FoldedMVInit("Wrapper", addr);
  FoldedMVLoadLayerMem(path, layer, PEs, Wtiles, Ttiles, API);
}

extern "C" void deinit() {
  FoldedMVDeinit();
}
