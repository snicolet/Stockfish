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

#include <cassert>

#include <algorithm> // For std::count
#include "movegen.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "syzygy/tbprobe.h"
#include "tt.h"

ThreadPool Threads; // Global object


/// Thread constructor launches the thread and waits until it goes to sleep
/// in idle_loop(). Note that 'searching' and 'exit' should be already set.

Thread::Thread(size_t n) : idx(n), stdThread(&Thread::idle_loop, this) {

  sync_cout << "[DEBUG_HANG] "
            << "Constructor of thread " << this->thread_index() << " "
            << "is calling wait_for_search_finished()..." << sync_endl; 
                
  wait_for_search_finished();
  
  sync_cout << "[DEBUG_HANG] "
            << "Constructor of thread " << this->thread_index() << " "
            << "is after the call of wait_for_search_finished()..." << sync_endl; 
}


/// Thread destructor wakes up the thread in idle_loop() and waits
/// for its termination. Thread should be already waiting.

Thread::~Thread() {

  assert(!searching);

  exit = true;
  start_searching();
  stdThread.join();
}

/// Thread::bestMoveCount(Move move) return best move counter for the given root move

int Thread::best_move_count(Move move) {

  auto rm = std::find(rootMoves.begin() + pvIdx,
                      rootMoves.begin() + pvLast, move);

  return rm != rootMoves.begin() + pvLast ? rm->bestMoveCount : 0;
}

/// Thread::clear() reset histories, usually before a new game

void Thread::clear() {

  counterMoves.fill(MOVE_NONE);
  mainHistory.fill(0);
  captureHistory.fill(0);

  for (auto& to : continuationHistory)
      for (auto& h : to)
          h->fill(0);

  continuationHistory[NO_PIECE][0]->fill(Search::CounterMovePruneThreshold - 1);
}

/// Thread::start_searching() wakes up the thread that will start the search

void Thread::start_searching() {

  std::lock_guard<Mutex> lk(mutex);
  searching = true;
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is calling notify_one() in start_searching()"
            << ", searching = " << searching << sync_endl;
            
  cv.notify_one(this->thread_index()); // Wake up the thread in idle_loop()
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is after notify_one() in start_searching()"  
            << ", searching = " << searching << sync_endl;
}


/// Thread::wait_for_search_finished() blocks on the condition variable
/// until the thread has finished searching.

void Thread::wait_for_search_finished() {

  std::unique_lock<Mutex> lk(mutex);
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is entering wait_for_search_finished()..." << sync_endl; 
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is calling cv.wait() in wait_for_search_finished()"
            << ", searching = " << searching << sync_endl;
                
  cv.wait(this->thread_index(), lk, [&]{ return !searching; }, 0);
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is after cv.wait() in wait_for_search_finished()"
            << ", searching = " << searching << sync_endl;
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is existing wait_for_search_finished()..." << sync_endl; 
}


/// Thread::idle_loop() is where the thread is parked, blocked on the
/// condition variable, when it has no work to do.

void Thread::idle_loop() {

  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "is entering idle_loop()..." << sync_endl;

  // If OS already scheduled us on a different group than 0 then don't overwrite
  // the choice, eventually we are one of many one-threaded processes running on
  // some Windows NUMA hardware, for instance in fishtest. To make it simple,
  // just check if running threads are below a threshold, in this case all this
  // NUMA machinery is not needed.
  if (Options["Threads"] > 8)
      WinProcGroup::bindThisThread(idx);

  while (true)
  {
      std::unique_lock<Mutex> lk(mutex);
      searching = false;
      
      sync_cout << "[DEBUG_HANG] "
                << "Thread " << this->thread_index() << " "
                << "is calling notify_one() in idle_loop()"
                << ", searching = " << searching << sync_endl;
      
      cv.notify_one(this->thread_index()); // Wake up anyone waiting for search finished
      
      sync_cout << "[DEBUG_HANG] "
                << "Thread " << this->thread_index() << " "
                << "is after notify_one() in idle_loop()"
                << ", searching = " << searching << sync_endl;
      
      sync_cout << "[DEBUG_HANG] "
                << "Thread " << this->thread_index() << " "
                << "is calling cv.wait() in idle_loop()"
                << ", searching = " << searching << sync_endl;
      
      cv.wait(this->thread_index(), lk, [&]{ return searching; }, 0);
      
      sync_cout << "[DEBUG_HANG] "
                << "Thread " << this->thread_index() << " "
                << "is after cv.wait() in idle_loop()"
                << ", searching = " << searching << sync_endl;

      if (exit)
          return;

      lk.unlock();

      search();
  }
  
  sync_cout << "[DEBUG_HANG] "
            << "Thread " << this->thread_index() << " "
            << "exiting idle_loop()..." << sync_endl;
}

/// ThreadPool::set() creates/destroys threads to match the requested number.
/// Created and launched threads will immediately go to sleep in idle_loop.
/// Upon resizing, threads are recreated to allow for binding if necessary.

void ThreadPool::set(size_t requested) {

  if (size() > 0) { // destroy any existing thread(s)
  
      sync_cout << "[DEBUG_HANG] "
                << "ThreadPool::set has size() >0 and "
                << "is calling wait_for_search_finished() for the main thread..." << sync_endl;
                
      main()->wait_for_search_finished();
      
      sync_cout << "[DEBUG_HANG] "
                << "ThreadPool::set has size() > 0 and "
                << "is after the call to wait_for_search_finished() for the main thread..." << sync_endl;

      while (size() > 0)
          delete back(), pop_back();
  }

  if (requested > 0) { // create new thread(s)
      
      sync_cout << "[DEBUG_HANG] "
                << "ThreadPool::set has requested > 0 and "
                << "is constructing the main thread..." << sync_endl;
  
      push_back(new MainThread(0));

      while (size() < requested)
      {
          sync_cout << "[DEBUG_HANG] "
                    << "ThreadPool::set has requested > 0 and "
                    << "is constructing the thread number " << size() << "..." << sync_endl;

          push_back(new Thread(size()));
      }
      clear();

      // Reallocate the hash with the new threadpool size
      TT.resize(Options["Hash"]);
  }
}

/// ThreadPool::clear() sets threadPool data to initial values.

void ThreadPool::clear() {

  for (Thread* th : *this)
      th->clear();

  main()->callsCnt = 0;
  main()->previousScore = VALUE_INFINITE;
  main()->previousTimeReduction = 1.0;
}

/// ThreadPool::start_thinking() wakes up main thread waiting in idle_loop() and
/// returns immediately. Main thread will wake up other threads and start the search.

void ThreadPool::start_thinking(Position& pos, StateListPtr& states,
                                const Search::LimitsType& limits, bool ponderMode) {

  sync_cout << "[DEBUG_HANG] "
            << "ThreadPool::start_thinking "
            << "is calling wait_for_search_finished() for the main thread..." << sync_endl;
                
  main()->wait_for_search_finished();
  
  sync_cout << "[DEBUG_HANG] "
            << "ThreadPool::start_thinking "
            << "is calling wait_for_search_finished() for the main thread..." << sync_endl;

  main()->stopOnPonderhit = stop = false;
  main()->ponder = ponderMode;
  Search::Limits = limits;
  Search::RootMoves rootMoves;

  for (const auto& m : MoveList<LEGAL>(pos))
      if (   limits.searchmoves.empty()
          || std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
          rootMoves.emplace_back(m);

  if (!rootMoves.empty())
      Tablebases::rank_root_moves(pos, rootMoves);

  // After ownership transfer 'states' becomes empty, so if we stop the search
  // and call 'go' again without setting a new position states.get() == NULL.
  assert(states.get() || setupStates.get());

  if (states.get())
      setupStates = std::move(states); // Ownership transfer, states is now empty

  // We use Position::set() to set root position across threads. But there are
  // some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
  // be deduced from a fen string, so set() clears them and to not lose the info
  // we need to backup and later restore setupStates->back(). Note that setupStates
  // is shared by threads but is accessed in read-only mode.
  StateInfo tmp = setupStates->back();

  for (Thread* th : *this)
  {
      th->shuffleExts = th->nodes = th->tbHits = th->nmpMinPly = 0;
      th->rootDepth = th->completedDepth = DEPTH_ZERO;
      th->rootMoves = rootMoves;
      th->rootPos.set(pos.fen(), pos.is_chess960(), &setupStates->back(), th);
  }

  setupStates->back() = tmp;

  main()->start_searching();
}
