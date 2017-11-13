/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include <algorithm>
#include <cassert>
#include <cstddef> // For offsetof()
#include <cstring> // For std::memset, std::memcmp
#include <iomanip>
#include <sstream>

#include "misc.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "types.h"
#include "uci.h"
#include "montecarlo.h"
#include "syzygy/tbprobe.h"
#include "tree-3.1/tree.hh"

using std::string;


// UCT: a class implementing Monte-Carlo Tree Search for Stockfish.
// We are following the survey http://mcts.ai/pubs/mcts-survey-master.pdf
// for the Monte-Carlo algorithm description and the notations used.

// Bibliography:
//     http://mcts.ai/pubs/mcts-survey-master.pdf
//     https://www.ke.tu-darmstadt.de/lehre/arbeiten/bachelor/2012/Arenz_Oleg.pdf
//     https://dke.maastrichtuniversity.nl/m.winands/publications.html

class UCT {
public:

  // Constructors
  UCT(Position& p);

  // The main function of the class
  Move search(Position& pos);

  // The high-level description of the UCT algorithm
  void create_root(Position& p);
  bool computational_budget();
  Node tree_policy();
  Node best_child(Node n, double c);
  Reward playout_policy(Node n);
  void backup(Node n, Reward r);

  // Playing moves
  void do_move(Move m);
  void undo_move();
  void generate_moves();

  // Evaluate current node with a local minimax search
  Value evaluate_with_minimax(Depth d);

private:

  // Data members
  Position&       pos;           // The current position of the tree, changes during search
  Position        rootPosition;  // A full copy of the position used to initialise the class
  Node            root;
  Node            currentNode;

  // Counters
  uint64_t        ply;
  uint64_t        treeSize;
  uint64_t        descentCnt;
  uint64_t        playoutCnt;
  uint64_t        doMoveCnt;
  
  // Stack to do/undo the moves: for compatibility with the alpha-beta search implementation,
  // we want to be able to reference from stack[-4] to stack[MAX_PLY + 2].
  Search::Stack   stackBuffer[MAX_PLY+7] , *stack  = stackBuffer+4;
  StateInfo       statesBuffer[MAX_PLY+7], *states = statesBuffer+4;
};


// UCT::search() is the main function of UCT algorithm.

Move UCT::search(Position& p) {

    create_root(p);

    while (computational_budget()) {
       Node n = tree_policy();
       Reward r = playout_policy(n);
       backup(n, r);
    }

    return move_of(best_child(root, 0));
}


/// UCT::UCT() is the constructor for the UCT class

UCT::UCT(Position& p) : pos(p) {
    create_root(p);
}

/// UCT::create_root() initializes the UCT tree with the given position

void UCT::create_root(Position& p) {

    // Initialize the global counters
    doMoveCnt  = 0;
    treeSize   = 0;
    descentCnt = 0;
    ply        = 0;

    // Prepare the stack to go down and up in the game tree
    std::memset(&stack[-4], 0, 7 * sizeof(Search::Stack));
    for (int i = 4; i > 0; i--)
      stack[-i].contHistory = &(pos.this_thread()->contHistory[NO_PIECE][0]); // Use as sentinel

    // TODO : what to do with killers ???

    // TODO : setupStates should probably come the caller, as a global ???
    StateListPtr setupStates(new std::deque<StateInfo>(1));   

    // Save a hard copy of the root position
    StateInfo tmp = setupStates->back();
    rootPosition.set(pos.fen(), pos.is_chess960(), &setupStates->back(), pos.this_thread());
    setupStates->back() = tmp;
}


/// UCT::computational_budget() stops the search if the computational budget
/// has been reached (time limit, or number of nodes, etc.)

bool UCT::computational_budget() {
    return (treeSize < 5);
}


/// UCT::tree_policy() selects the next node to be expanded

Node UCT::tree_policy() {
    descentCnt++;
    return root;
}


/// UCT::playout_policy() plays a semi random game starting from the last extended node

Reward UCT::playout_policy(Node n) {
    playoutCnt++;
    return 1.0;
}


/// UCT::backup() implements the strategy for accumulating rewards up
/// the tree after a playout.

void UCT::backup(Node n, Reward r) {
}


/// UCT::best_child() selects the best child of a node according to the UCT formula

Node UCT::best_child(Node n, double c) {
    return n;
}


/// UCT::do_move() plays a move in the search tree from the current position

void UCT::do_move(Move m) {

  stack[ply].ply         = ply;
  stack[ply].currentMove = m;
  stack[ply].contHistory = &(pos.this_thread()->contHistory[pos.moved_piece(m)][to_sq(m)]);

  pos.do_move(m, states[ply]);

  ply++;
}


/// UCT::undo_move() undos the current move in the search tree

void UCT::undo_move() {
  ply--;
  pos.undo_move(stack[ply].currentMove);
}


/// UCT::generate_moves() does some Stockfish gimmick to iterate over legal moves
/// in a sensible order.
/// For historical reasons, it is not so easy to get a MovePicker object to
/// generate moves if we want to have a decent order (captures first, then
/// quiet moves, etc.). We have to pass various history tables to the MovePicker
/// constructor, like in the alpha-beta implementation of move ordering.

void UCT::generate_moves() {

  Thread*  thread      = pos.this_thread();
  Square   prevSq      = to_sq(stack[ply-1].currentMove);
  Move     countermove = thread->counterMoves[pos.piece_on(prevSq)][prevSq];
  Move     ttMove      = MOVE_NONE;  // FIXME
  Move*    killers     = stack[ply].killers;
  Depth    depth       = 30 * ONE_PLY;
  
  const CapturePieceToHistory* cph   = &thread->captureHistory;
  const ButterflyHistory* mh         = &thread->mainHistory;
  const PieceToHistory*   contHist[] = { stack[ply-1].contHistory, 
                                         stack[ply-2].contHistory, 
                                         nullptr, 
                                         stack[ply-4].contHistory };

  MovePicker mp(pos, ttMove, depth, mh, cph, contHist, countermove, killers);

  Move move;
  int moveCount = 0;
   
  while ((move = mp.next_move()) != MOVE_NONE)
      if (pos.legal(move))
      {
          stack[ply].moveCount = ++moveCount;
      }
}

/// UCT::evaluate_with_minimax() evaluates the current position in the tree
/// with a small minimax search of the given depth. Use depth==DEPTH_ZERO
/// for a direct quiescence value.

Value UCT::evaluate_with_minimax(Depth depth) {
    return minimax_value(pos, &stack[ply], depth);
}




