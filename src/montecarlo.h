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

#ifndef MONTECARLO_H_INCLUDED
#define MONTECARLO_H_INCLUDED

#include <cassert>
#include <deque>
#include <memory> // For std::unique_ptr
#include <string>

#include "bitboard.h"
#include "types.h"
#include "tree-3.1/tree.hh"


typedef double Reward;


/// UCTInfo class stores information in a node

struct MoveAndPrior {
  Move move;
  Reward prior;
};

struct {
  bool operator()(MoveAndPrior a, MoveAndPrior b) const { return a.prior < b.prior; }
} CompareMoveAndPrior;

const int MAX_SONS = 64;

class UCTInfo {
public:

  Move           last_move()   { return lastMove; }
  MoveAndPrior*  priors_list() { return &(priors[0]); }

  // Data members
  uint64_t       visits        = 0;         // number of visits by the UCT algorithm
  Reward         reward        = 0.0;       // reward from the point of view of the side to move
  int            expandedSons  = 0;         // number of sons expanded by the UCT algorithm
  int            sons          = 0;         // total number of legal moves
  Move           lastMove      = MOVE_NONE; // the move between the parent and this node
  MoveAndPrior   priors[MAX_SONS];
};


//typedef tree<UCTInfo> Node;
typedef UCTInfo* Node;


Node create_node(const Position& pos) {
   return nullptr;  // TODO, FIXME : this should create a Node !
}

Node son_after(Node node, Move move) {
   return node;    /// TODO, FIXME : this is the son after move from the given node !
}

UCTInfo* get_infos(Node node) {
  //return &(node.begin().node->data);
  return node;
}

Move move_of(Node node) {
    return get_infos(node)->last_move();
}

MoveAndPrior* get_list_of_priors(Node node) {
    return get_infos(node)->priors_list();
}

int number_of_sons(Node node) {
    return get_infos(node)->sons;
}



#endif // #ifndef MONTECARLO_H_INCLUDED
