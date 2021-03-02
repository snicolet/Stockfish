/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

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

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "material.h"
#include "movepick.h"
#include "pawns.h"
#include "position.h"
#include "search.h"
#include "thread_win32_osx.h"


typedef std::vector<Search::RootMove> RootMoves;

/// Thread class keeps together all the thread-related stuff. We use
/// per-thread pawn and material hash tables so that once we get a
/// pointer to an entry its life time is unlimited and we don't have
/// to care about someone changing the entry under our feet.

class Thread {

  std::mutex mutex;
  std::condition_variable cv;
  size_t idx;
  bool exit = false, searching = true; // Set before starting std::thread
  NativeThread stdThread;

public:
  explicit Thread(size_t);
  virtual ~Thread();
  virtual void search();
  void clear();
  void idle_loop();
  void start_searching();
  void wait_for_search_finished();

  Pawns::Table pawnsTable;
  Material::Table materialTable;
  size_t pvIdx, pvLast;
  uint64_t ttHitAverage;
  int selDepth, nmpMinPly;
  Color nmpColor;
  std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;

  Position rootPos;
  StateInfo rootState;
  RootMoves rootMoves;
  Depth rootDepth, completedDepth;
  CounterMoveHistory counterMoves;
  ButterflyHistory mainHistory;
  LowPlyHistory lowPlyHistory;
  CapturePieceToHistory captureHistory;
  ContinuationHistory continuationHistory[2][2];
  Score contempt;
  int failedHighCnt;
};


/// MainThread is a derived class specific for main thread

struct MainThread : public Thread {

  using Thread::Thread;

  void search() override;
  void check_time();

  double previousTimeReduction;
  Value bestPreviousScore;
  Value iterValue[4];
  int callsCnt;
  bool stopOnPonderhit;
  std::atomic_bool ponder;
};


/// ThreadPool struct handles all the threads-related stuff like init, starting,
/// parking and, most importantly, launching a thread. All the access to threads
/// is done through this class.

struct ThreadPool : public std::vector<Thread*> {

  void start_thinking(Position&, StateListPtr&, const Search::LimitsType&, bool = false);
  void clear();
  void set(size_t);

  MainThread* main()        const { return static_cast<MainThread*>(front()); }
  uint64_t nodes_searched() const { return accumulate(&Thread::nodes); }
  uint64_t tb_hits()        const { return accumulate(&Thread::tbHits); }
  Thread* get_best_thread() const;
  void start_searching();
  void wait_for_search_finished() const;

  std::atomic_bool stop, increaseDepth;

private:
  StateListPtr setupStates;

  uint64_t accumulate(std::atomic<uint64_t> Thread::* member) const {

    uint64_t sum = 0;
    for (Thread* th : *this)
        sum += (th->*member).load(std::memory_order_relaxed);
    return sum;
  }
};

extern ThreadPool Threads;

/// Breadcrumbs are used to mark nodes as being searched by a given thread

struct Breadcrumb {
  std::atomic<Thread*> thread;
  std::atomic<Key> key;
};

extern std::array<Breadcrumb, 1024> breadcrumbs;

/// ThreadHolding structure keeps track of which thread left breadcrumbs at the given
/// node for potential reductions. A free node will be marked upon entering the moves
/// loop by the constructor, and unmarked upon leaving that loop by the destructor.

struct ThreadHolding {

  explicit ThreadHolding(Thread* thisThread, Key posKey, int ply) {
  
    location = ply < 8 ? &breadcrumbs[posKey & (breadcrumbs.size() - 1)] : nullptr;
    otherThread = false;
    owning = false;
    if (location)
    {
        // See if another already marked this location, if not, mark it ourselves
        Thread* tmp = (*location).thread.load(std::memory_order_relaxed);
        if (tmp == nullptr)
        {
            (*location).thread.store(thisThread, std::memory_order_relaxed);
            (*location).key.store(posKey, std::memory_order_relaxed);
            owning = true;
        }
        else if (   tmp != thisThread
                 && (*location).key.load(std::memory_order_relaxed) == posKey)
                 otherThread = true;
    }
  }

  ~ThreadHolding() {
     if (owning) // Free the marked location
         (*location).thread.store(nullptr, std::memory_order_relaxed);
  }

  bool marked() { return otherThread; }

private:

  Breadcrumb* location;
  bool otherThread, owning;
};

#endif // #ifndef THREAD_H_INCLUDED
