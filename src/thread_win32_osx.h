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
/// relies on libwinpthread. Currently libwinpthread implements mutexes directly
/// on top of Windows semaphores. Semaphores, being kernel objects, require kernel
/// mode transition in order to lock or unlock, which is very slow compared to
/// interlocked operations (about 30% slower on bench test). To work around this
/// issue, we define our wrappers to the low level Win32 calls. We use critical
/// sections to support Windows XP and older versions. Unfortunately, cond_wait()
/// is racy between unlock() and WaitForSingleObject() but they have the same
/// speed performance as the SRW locks.

#include <condition_variable>
#include <mutex>
#include <thread>

#if defined(_WIN32)// && !defined(_MSC_VER)

#ifndef NOMINMAX
#  define NOMINMAX // Disable macros min() and max()
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX

/// Mutex and ConditionVariable struct are wrappers of the low level locking
/// machinery and are modeled after the corresponding C++11 classes.

struct Mutex {
  Mutex() { InitializeCriticalSection(&cs); }
 ~Mutex() { DeleteCriticalSection(&cs); }
  void lock() { EnterCriticalSection(&cs); }
  void unlock() { LeaveCriticalSection(&cs); }

private:
  CRITICAL_SECTION cs;
};
/*
#include <map>
struct ConditionVariable {
	ConditionVariable() {}
	~ConditionVariable() {}
	template<class _Predicate>
	void wait(std::unique_lock<Mutex>& _Lck, _Predicate _Pred, bool sloppy)
	{
		while (!_Pred())
		{
			// correct way is to take internal lock here instead of below
			//std::unique_lock<Mutex> lk(lock);
			_Lck.unlock();
			if (sloppy)
			{
				Sleep(1000);
			}
			// wrong way is to take internal lock here(my own failed implementation)
			std::unique_lock<Mutex> lk(lock);
			HANDLE myhandle = INVALID_HANDLE_VALUE;
			std::map<DWORD, HANDLE>::iterator iter = waiters.find(GetCurrentThreadId());
			if (iter != waiters.end())
			{
				myhandle = iter->second;
			}
			else
			{
				myhandle = CreateEvent(NULL, FALSE, FALSE, NULL);
				waiters.insert(std::make_pair(GetCurrentThreadId(), myhandle));
			}
			lk.unlock();
			WaitForSingleObject(myhandle, INFINITE);
			CloseHandle(myhandle);
			_Lck.lock();
		}
	}
	void notify_one() {
		std::lock_guard<Mutex> lk(lock);
		std::map<DWORD, HANDLE>::iterator iter = waiters.begin();
		if (iter != waiters.end())
		{
			SetEvent(iter->second);
			waiters.erase(iter);
		}
	}
	void notify_all() {
		std::lock_guard<Mutex> lk(lock);
		std::map<DWORD, HANDLE>::iterator iter = waiters.begin();
		while (iter != waiters.end())
		{
			SetEvent(iter->second);
			iter = waiters.erase(iter);
		}
	}
private:
	std::map<DWORD, HANDLE> waiters;
	Mutex lock;
};
*/

// this is the bugged mingw's way
struct ConditionVariable {
	ConditionVariable() { handle = CreateSemaphore(NULL, 0, 512, NULL); waiters_count = 0; }
	~ConditionVariable() { CloseHandle(handle); }
	template<class _Predicate>
	void wait(std::unique_lock<Mutex>& _Lck, _Predicate _Pred, bool sloppy)
	{
		while (!_Pred())
		{
			std::unique_lock<Mutex> lk(lock);
			waiters_count--;
			_Lck.unlock();
			lk.unlock();
			if (sloppy)
			{
				Sleep(1000);
			}
			WaitForSingleObject(handle, INFINITE);
			_Lck.lock();
		}
	}
	void notify_one() {
		std::lock_guard<Mutex> lk(lock);
		waiters_count++;
		ReleaseSemaphore(handle, 1, NULL);
	}
private:
	HANDLE handle;
	int waiters_count;
	Mutex lock;
};

#else // Default case: use STL classes

typedef std::mutex Mutex;
typedef std::condition_variable ConditionVariable;

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

#endif // #ifndef THREAD_WIN32_OSX_H_INCLUDED
