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


#include "position.h"
#include "thread.h"
#include "types.h"


struct NodeInfo;
struct Edge;

typedef double Reward;
typedef NodeInfo* Node;


class UCT {
public:

  // Constructors
  UCT(Position& p);

  // The main function of the class
  Move search();

  // The high-level description of the UCT algorithm
  void create_root();
  bool computational_budget();
  Node tree_policy();
  Edge* best_child(Node node, double C);
  Reward playout_policy(Node node);
  void backup(Node node, Reward r);

  // The UCB formula
  double UCB(Node node, Edge& edge, double C);

  // Playing moves
  Node current_node();
  void do_move(Move move);
  void undo_move();
  void generate_moves();

  // Evaluations of nodes in the tree
  Value evaluate_with_minimax(Depth d);
  Reward calculate_prior(Move m, int moveCount);
  void add_prior_to_node(Node node, Move m, Reward prior, int moveCount);
  Reward value_to_reward(Value v);
  Value reward_to_value(Reward r);

  // Other helpers
  double get_exploration_constant();
  void set_exploration_constant(double C);

  // Testing and debugging
  void print_stats();
  void print_node(Node node);
  void print_edge(Edge edge);
  void test();

private:

  // Data members
  Position&       pos;                  // The current position of the tree, changes during search
  Position        rootPosition;         // A full copy of the position used to initialise the class
  Node            root;                 // A pointer to the root
  double          exploration = 10.0;   // Default value for the UCB formula

  // Counters
  int             ply;
  int             descentCnt;
  int             playoutCnt;
  int             doMoveCnt;
  int             priorCnt;

  // Stack to do/undo the moves: for compatibility with the alpha-beta search implementation,
  // we want to be able to reference from stack[-4] to stack[MAX_PLY + 2].
  Node            nodesBuffer [MAX_PLY+7],  *nodes   = nodesBuffer  + 4;
  Edge*           edgesBuffer [MAX_PLY+7],  **edges  = edgesBuffer  + 4;
  Search::Stack   stackBuffer [MAX_PLY+7],  *stack   = stackBuffer  + 4;
  StateInfo       statesBuffer[MAX_PLY+7],  *states  = statesBuffer + 4;
};


const int MAX_CHILDREN = 64;


/// Edge struct stores the statistics of one edge between nodes in the UCT tree
struct Edge {
  Move    move;
  double  visits;
  Reward  prior;
  Reward  actionValue;
  Reward  meanActionValue;
};

/// NodeInfo struct stores information in a node of the UCT tree
struct NodeInfo {
public:

  Move     last_move()      { return lastMove; }
  Edge*    children_list()  { return &(children[0]); }

  // Data members
  // At initialization time we fill them with artificial values to make debugging easier
  Key      key1           = -99;
  Key      key2           = -553;
  int      visits         = -373;      // number of visits by the UCT algorithm
  int      number_of_sons = -1003;     // total number of legal moves
  int      expandedSons   = -5977;     // number of sons expanded by the UCT algorithm
  Move     lastMove       = MOVE_NONE; // the move between the parent and this node
  Edge     children[MAX_CHILDREN];
};


// Comparison function for edges
struct { bool operator()(Edge a, Edge b) const { return a.prior > b.prior; }} ComparePrior;

// The UCT tree is stored implicitly in one big hash table
const int UCT_HASH_SIZE = 8192;
typedef HashTable<NodeInfo, UCT_HASH_SIZE> UCTHashTable;


#endif // #ifndef MONTECARLO_H_INCLUDED
