/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2019 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef THREAD_WIN32_OSX_H_INCLUDED
#define THREAD_WIN32_OSX_H_INCLUDED

/// STL thread library used by mingw and gcc when cross compiling for Windows
/// relies on libwinpthread.

#include <condition_variable>
#include <mutex>
#include <thread>


#if defined(_WIN32) && !defined(_MSC_VER)

#ifndef NOMINMAX
#  define NOMINMAX // Disable macros min() and max()
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX

#endif

/// On OSX threads other than the main thread are created with a reduced stack
/// size of 512KB by default, this is dangerously low for deep searches, so
/// adjust it to TH_STACK_SIZE. The implementation calls pthread_create() with
/// proper stack size parameter.

#if defined(__APPLE__)

#include <pthread.h>

static const size_t TH_STACK_SIZE = 2 * 1024 * 1024;

template <class T, class P = std::pair<T*, void(T::*)()>>
void* start_routine(void* ptr)
{
   P* p = reinterpret_cast<P*>(ptr);
   (p->first->*(p->second))(); // Call member function pointer
   delete p;
   return NULL;
}

class NativeThread {

   pthread_t thread;

public:
  template<class T, class P = std::pair<T*, void(T::*)()>>
  explicit NativeThread(void(T::*fun)(), T* obj) {
    pthread_attr_t attr_storage, *attr = &attr_storage;
    pthread_attr_init(attr);
    pthread_attr_setstacksize(attr, TH_STACK_SIZE);
    pthread_create(&thread, attr, start_routine<T>, new P(obj, fun));
  }
  void join() { pthread_join(thread, NULL); }
};

#else // Default case: use STL classes

typedef std::thread NativeThread;

#endif


/// Mutex and ConditionVariable struct are wrappers of the low level locking
/// machinery and are modeled after the corresponding C++11 classes.

/// Mutex class wraps low level atomic operations to provide a yielding spinlock
class Mutex {
  std::atomic_int _lock;
public:
  Mutex() { _lock = 1; } // Init here to workaround a bug with MSVC 2013
  ~Mutex() { unlock();}
  void lock() {
      while (_lock.fetch_sub(1, std::memory_order_acquire) != 1)
          for (int cnt= 0; _lock.load(std::memory_order_relaxed) <= 0; ++cnt)
        	  if (cnt >= 10000) std::this_thread::yield(); // Be nice to hyperthreading
  }
  void unlock() { _lock.store(1, std::memory_order_release); }
};

typedef std::condition_variable_any ConditionVariable;


#endif // #ifndef THREAD_WIN32_OSX_H_INCLUDED
