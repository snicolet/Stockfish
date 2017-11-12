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

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "montecarlo.h"
#include "syzygy/tbprobe.h"
#include "tree-3.1/tree.hh"

using std::string;


// UCT: a class implementing Monte-Carlo Tree Search for Stockfish.
// We are following the survey http://mcts.ai/pubs/mcts-survey-master.pdf 
// for the Monte-Carlo algorithm description and the notations used.

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

// UCT() : the constructor for the UCT class

UCT::UCT(Position& p) : pos(p) {
    create_root(p);
}

void UCT::create_root(Position& p) {
    doMoveCnt = 0;
    treeSize = 0;
    descentCnt = 0;
    ply = 0;
}

bool UCT::computational_budget() {
    return (treeSize < 5);
}

Node UCT::tree_policy() {
    descentCnt++;
    return root;
}

Reward UCT::playout_policy(Node n) {
    playoutCnt++;
    return 1.0;
}

void UCT::backup(Node n, Reward r) {
}

Node UCT::best_child(Node n, double c) {
    return n;
}

void UCT::do_move(Move m) {

  stack[ply].ply = ply;
  stack[ply].currentMove = m;
  
  pos.do_move(m, states[ply]);

  ply++;
}

void UCT::undo_move() {
  ply--;
  pos.undo_move(stack[ply].currentMove);
}

