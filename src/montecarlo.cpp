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
#include <cmath>
#include <cstring> // For std::memset, std::memcmp
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "misc.h"
#include "montecarlo.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "types.h"
#include "uci.h"



// MonteCarlo is a class implementing Monte-Carlo Tree Search for Stockfish.
// We are following the survey http://mcts.ai/pubs/mcts-survey-master.pdf
// for the notations and the description of the Monte-Carlo algorithm.

// Bibliography:
//     http://mcts.ai/pubs/mcts-survey-master.pdf
//     https://www.ke.tu-darmstadt.de/lehre/arbeiten/bachelor/2012/Arenz_Oleg.pdf
//     https://dke.maastrichtuniversity.nl/m.winands/publications.html
//     https://www.ru.is/faculty/yngvi/pdf/WinandsB11a.pdf
//     http://cassio.free.fr/pdf/alphago-zero-nature.pdf


using namespace std;
using std::string;


MCTSHashTable MCTS;

const Reward REWARD_NONE  = Reward(0.0);
const Reward REWARD_MATED = Reward(0.0);
const Reward REWARD_DRAW  = Reward(0.5);
const Reward REWARD_MATE  = Reward(1.0);

Edge EDGE_NONE = {MOVE_NONE, 0, REWARD_NONE, REWARD_NONE, REWARD_NONE};


/// get_node() probes the MonteCarlo hash table to know if we can find the node
/// for the given position.
Node get_node(const Position& pos) {

   Key key1 = pos.key();
   Key key2 = pos.pawn_key();

   auto range = MCTS.equal_range(key1);
   auto it1 = range.first;
   auto it2 = range.second;

   // If the node already already exists (in the range of all the
   // hash table entries with key "key1"), return it.
   while (it1 != it2)
   {
       Node node = &(it1->second);

       if (node->key1 == key1 && node->key2 == key2)
           return node;

       it1++;
   }

   // Node was not found, so we have to create a new one
   NodeInfo infos;

   infos.key1             = key1;      // Zobrist hash of all pieces, including pawns
   infos.key2             = key2;      // Zobrist hash of pawns
   infos.visits           = 0;         // number of visits by the MonteCarlo algorithm
   infos.number_of_sons   = 0;         // total number of legal moves
   infos.expandedSons     = 0;         // number of sons expanded by the MonteCarlo algorithm
   infos.lastMove         = MOVE_NONE; // the move between the parent and this node

   debug << "inserting into the hash table: key = " << key1 << endl;

   auto it = MCTS.insert(make_pair(key1, infos));
   return &(it->second);
}

// Helpers functions
Move move_of(Node node) { return node->last_move(); }
Edge* get_list_of_children(Node node) { return node->children_list(); }
int number_of_sons(Node node) { return node->number_of_sons; }


// MonteCarlo::search() is the main function of MonteCarlo algorithm.
Move MonteCarlo::search() {

    create_root();

    while (computational_budget()) {
       print_stats();
       Node node = tree_policy();
       Reward reward = playout_policy(node);
       backup(node, reward);
    }

    return best_child(root, 0.0)->move;
}


/// MonteCarlo::MonteCarlo() is the constructor for the MonteCarlo class
MonteCarlo::MonteCarlo(Position& p) : pos(p) {
    create_root();
}


/// MonteCarlo::create_root() initializes the MonteCarlo tree with the given position
void MonteCarlo::create_root() {

    // Initialize the global counters
    doMoveCnt  = 0;
    descentCnt = 0;
    playoutCnt = 0;
    priorCnt   = 0;
    startTime  = now();

    // Prepare the stack to go down and up in the game tree
    ply = 1;
    std::memset(stackBuffer, 0, sizeof(stackBuffer));
    for (int i = -4; i <= MAX_PLY + 2; i++)
      stack[i].contHistory = &(pos.this_thread()->contHistory[NO_PIECE][0]); // Use as sentinel

    // TODO : what to do with killers ???

    // TODO : setupStates should probably come the caller, as a global ???
    StateListPtr setupStates(new std::deque<StateInfo>(1));

    // Save a hard copy of the root position
    StateInfo tmp = setupStates->back();
    rootPosition.set(pos.fen(), pos.is_chess960(), &setupStates->back(), pos.this_thread());
    setupStates->back() = tmp;

    // Erase the list of nodes, and set the current node to the root node
    std::memset(nodesBuffer, 0, sizeof(nodesBuffer));
    root = nodes[ply] = get_node(pos);

    if (current_node()->visits == 0)
       generate_moves();

    assert(ply == 1);
    assert(root == nodes[ply]);
    assert(root == current_node());
}


/// MonteCarlo::computational_budget() stops the search if the computational budget
/// has been reached (time limit, or number of nodes, etc.)
bool MonteCarlo::computational_budget() {
    assert(current_node() == root);

    return (descentCnt < 100);
}


/// MonteCarlo::tree_policy() selects the next node to be expanded
Node MonteCarlo::tree_policy() {
    debug << "Entering tree_policy()..." << endl;

    assert(current_node() == root);
    descentCnt++;

    if (number_of_sons(current_node()) == 0)
       return current_node();

    while (current_node()->visits > 0)
    {
        // Check for mate or stalemate
        if (number_of_sons(current_node()) == 0)
            return current_node();

        // Check for draw by repetition or draw by 50 moves rule
        if (pos.is_draw(ply - 1))
            return current_node();

        double C = get_exploration_constant();

        edges[ply] = best_child(current_node(), C);
        Move m = edges[ply]->move;

        debug << "edges[" << ply << "].move = "
             << UCI::move(edges[ply]->move, pos.is_chess960())
             << std::endl;

        assert(is_ok(m));
        assert(pos.legal(m));

        do_move(m);

        debug << "stack[" << ply-1 << "].currentMove = "
             << UCI::move(stack[ply-1].currentMove, pos.is_chess960())
             << std::endl;

        nodes[ply] = get_node(pos); // Set current node
    }

    assert(current_node()->visits == 0);

    debug << "... exiting tree_policy()" << endl;

    return current_node();
}


/// MonteCarlo::playout_policy() expands the selected node, plays a semi random game starting
/// from there, and return the reward of this playout from the point of view of the
/// player to move in the expanded move.
Reward MonteCarlo::playout_policy(Node node) {

    playoutCnt++;
    assert(current_node() == node);

    if (node->visits > 0 && number_of_sons(node) == 0)
    	return pos.checkers() ? REWARD_MATED : REWARD_DRAW;

    if (pos.is_draw(ply - 1))
        return REWARD_DRAW;

    assert(current_node()->visits == 0);

    // Expand the current node, generating the legal moves and
    // calculating their prior value.
    Node old = current_node();
    generate_moves();
    assert(current_node() == old);

    if (number_of_sons(node) == 0)
        return pos.checkers() ? REWARD_MATED : REWARD_DRAW;

    print_stats();
    assert(current_node()->visits == 1);
    assert(current_node()->number_of_sons > 0);

    // Now implement a play-out policy from the newly expanded node,
    // and return the reward of the play-out from the point of view
    // of the side to play in that node.

    // In this initial implementation we do not really make any play-out,
    // and just return the prior value of the first legal moves (the legal
    // moves were sorted by reward in the generate_moves() call).

    return get_list_of_children(current_node())[0].prior;
}


/// MonteCarlo::UCB() calculates the upper confidence bound formula for the son
/// which we reach from node "node" by following the edge "edge".
double MonteCarlo::UCB(Node node, Edge& edge, double C) {

    int fatherVisits = node->visits;

    assert(fatherVisits > 0);

    double result = 0.0;

    if (edge.visits)
        result += edge.meanActionValue;

    result += C * edge.prior * sqrt(fatherVisits) / (1 + edge.visits);  // (or 1 + edge.losses)

    return result;
}


/// MonteCarlo::backup() implements the strategy for accumulating rewards up the tree
/// after a playout.
void MonteCarlo::backup(Node node, Reward r) {

   debug << "Entering backup()..." << endl;
   debug << pos << endl;
   debug << "reward r = " << r << endl;
   print_stats();
   print_node(current_node());

   assert(node == current_node());
   assert(ply >= 1);

   while (ply != 1) // root ?
   {
       undo_move();

       r = 1.0 - r;

       // Update the stats of the edge
       Edge* edge = edges[ply];

       debug << "stack[" << ply << "].currentMove = "
             << UCI::move(stack[ply].currentMove, pos.is_chess960())
             << std::endl;
       print_edge(*edge);

       edge->visits          = edge->visits + 1.0;
       edge->actionValue     = edge->actionValue + r;
       edge->meanActionValue = edge->actionValue / edge->visits;

       print_edge(*edge);

       assert(stack[ply].currentMove == edge->move);
   }


   debug << "... exiting backup()" << endl;

   assert(ply == 1);
   assert(current_node() == root);
}


/// MonteCarlo::best_child() selects the best child of a node according to the UCB formula
Edge* MonteCarlo::best_child(Node node, double C) {

    debug << "Entering best_child()..." << endl;
    debug << pos << endl;

    if (number_of_sons(node) <= 0)
    {
       debug << "... exiting best_child()" << endl;
       return &EDGE_NONE;
    }

    Edge* children = get_list_of_children(node);

    for (int k = 0 ; k < number_of_sons(node) ; k++)
    {
        debug << "move #" << k << ": "
              << UCI::move(children[k].move, pos.is_chess960())
              << " with " << children[k].visits
              << (children[k].visits > 0 ? " visits":" visit")
              << " and prior " << children[k].prior
              << endl;
    }

    int best = -1;
    double bestValue = -100000000.0;
    for (int k = 0 ; k < number_of_sons(node) ; k++)
    {
        double r = UCB(node, children[k], C);
        if ( r > bestValue )
        {
            bestValue = r;
            best = k;
        }
    }

    debug << "=> Selecting move " << UCI::move(children[best].move, pos.is_chess960())
          << " with UCB " << bestValue
          << endl;
    debug << "... exiting best_child()" << endl;

    return &children[best];
}

/// MonteCarlo::emit_pv() checks if it should write the pv of the game tree.
void MonteCarlo::emit_pv(bool forced) {

    bool emission;

    if (forced)
        emission = true;
    else
    {
        TimePoint elapsed = now() - startTime + 1;  // in milliseconds
        emission = false;
        if (elapsed < 1000)            emission |= (elapsed % 500) == 0;
        if (elapsed < 10 * 1000)       emission |= (elapsed % 1000) == 0;
        if (elapsed < 60 * 1000)       emission |= (elapsed % 10000) == 0;
        if (elapsed < 5 * 60 * 1000)   emission |= (elapsed % 30000) == 0;
        if (elapsed < 60 * 60 * 1000)  emission |= (elapsed % 60000) == 0;
        if (elapsed >= 60 * 60 * 1000) emission |= (elapsed % 600000) == 0;
    }

   if (emission)
   {
       const Search::RootMoves& rootMoves = pos.this_thread()->rootMoves;
   }

}



/// MonteCarlo::current_node() is the current node of our tree exploration
Node MonteCarlo::current_node() {
    return nodes[ply];
}


/// MonteCarlo::do_move() plays a move in the search tree from the current position
void MonteCarlo::do_move(Move m) {

    assert(ply < MAX_PLY);

    doMoveCnt++;

    stack[ply].ply         = ply;
    stack[ply].currentMove = m;
    stack[ply].contHistory = &(pos.this_thread()->contHistory[pos.moved_piece(m)][to_sq(m)]);

    pos.do_move(m, states[ply]);

    ply++;
}


/// MonteCarlo::undo_move() undo the current move in the search tree
void MonteCarlo::undo_move() {

    assert(ply > 1);

    ply--;
    pos.undo_move(stack[ply].currentMove);
}


/// MonteCarlo::add_prior_to_node() adds the given (move,prior) pair as a new son for a node
void MonteCarlo::add_prior_to_node(Node node, Move m, Reward prior, int moveCount) {

   assert(node->number_of_sons < MAX_CHILDREN);

   int n = node->number_of_sons;
   if (n < MAX_CHILDREN)
   {
       node->children[n].visits          = 0;
       node->children[n].move            = m;
       node->children[n].prior           = prior;
       node->children[n].actionValue     = 0.0;
       node->children[n].meanActionValue = 0.0;
       node->number_of_sons++;

       debug << "Adding move #" << n << ": "
             << UCI::move(m, pos.is_chess960())
             << " with " << 0 << " visit"
             << " and prior " << prior
             << endl;

       assert(node->number_of_sons == moveCount);
   }
   else
   {
        debug << "ERROR : too many sons (" << node->number_of_sons << ") in add_prior_to_node()" << endl;
   }
}


/// MonteCarlo::generate_moves() does some Stockfish gimmick to iterate over legal moves
/// of the current position, in a sensible order.
/// For historical reasons, it is not so easy to get a MovePicker object to
/// generate moves if we want to have a decent order (captures first, then
/// quiet moves, etc.). We have to pass various history tables to the MovePicker
/// constructor, like in the alpha-beta implementation of move ordering.
void MonteCarlo::generate_moves() {

    assert(current_node()->visits == 0);

    debug << "Entering generate_moves()..." << endl;
    debug << pos << endl;

    if (pos.should_debug())
       hit_any_key();

    print_node(current_node());

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
    Reward prior;
    int moveCount = 0;

    // Generate the legal moves and calculate their priors
    while ((move = mp.next_move()) != MOVE_NONE)
        if (pos.legal(move))
        {
            stack[ply].moveCount = ++moveCount;

            prior = calculate_prior(move, moveCount);

            add_prior_to_node(current_node(), move, prior, moveCount);

        }

    // Sort the moves according to their prior value
    int n = number_of_sons(current_node());
    if (n > 0)
    {
        Edge* children = get_list_of_children(current_node());
        std::sort(children, children + n, ComparePrior);
    }

    // Indicate that we have just expanded the current node
    Node s = current_node();
    s->visits       = 1;
    s->expandedSons = 0;

    debug << "... exiting generate_moves()" << endl;
}


/// MonteCarlo::evaluate_with_minimax() evaluates the current position in the tree
/// with a small minimax search of the given depth. Note : you can use
/// depth==DEPTH_ZERO for a direct quiescence value.
Value MonteCarlo::evaluate_with_minimax(Depth depth) {

    stack[ply].ply          = ply;
    stack[ply].currentMove  = MOVE_NONE;
    stack[ply].excludedMove = MOVE_NONE;

    Value v = minimax_value(pos, &stack[ply], depth);

    debug << pos << endl;
    debug << "minimax value = " << v << endl;

    return v;
}


/// MonteCarlo::calculate_prior() returns the a-priori reward of the move leading to
/// the n-th son of the current node. Here we use the evaluation function to
/// estimate this prior, we could use other strategies too (like the rank n of
/// the son, or the type of the move (good capture/quiet/bad capture), etc).
Reward MonteCarlo::calculate_prior(Move move, int n) {

    assert(n >= 0);

    priorCnt++;

    do_move(move);
    Reward prior = value_to_reward(-evaluate_with_minimax(3 * ONE_PLY));
    //Reward prior = value_to_reward(-evaluate_with_minimax(DEPTH_ZERO));
    undo_move();

    return prior;
}


/// MonteCarlo::value_to_reward() transforms a Stockfish value to a reward in [0..1]
/// We scale the logistic function such that a value of 600 (about three pawns)
/// is given a probability of win of 0.75, and a value of -600 is given a probability
/// of win of 0.25
Reward MonteCarlo::value_to_reward(Value v) {
    const double k = -0.00183102048111;
    double r = 1.0 / (1 + exp(k * int(v)));

    assert(REWARD_MATED <= r && r <= REWARD_MATE);
    return Reward(r);
}


/// MonteCarlo::reward_to_value() transforms a reward in [0..1] to a Stockfish value.
/// The scale is such that a reward of 0.75 corresponds to 600 (about three pawns),
/// and a reward of 0.25 corresponds to -600 (about minus three pawns).
Value MonteCarlo::reward_to_value(Reward r) {
    if (r > 0.99) return  VALUE_KNOWN_WIN;
    if (r < 0.01) return -VALUE_KNOWN_WIN;

    const double g = 546.14353597715121;  //  this is 1 / k
    double v = g * log(r / (1.0 - r)) ;
    return Value(int(v));
}


/// MonteCarlo::set_exploration_constant() changes the exploration constant of the UCB formula.
///
/// This constant sets the balance between the exploitation of past results and the
/// exploration of new branches in the MonteCarlo tree. The higher the constant, the more
/// likely is the algorithm to explore new parts of the tree, whereas lower values
/// of the constant makes an algorithm which focuses more on the already explored
/// parts of the tree. Default value is 10.0
void MonteCarlo::set_exploration_constant(double C) {
    exploration = C;
}


/// MonteCarlo::get_exploration_constant() returns the exploration constant of the UCB formula
double MonteCarlo::get_exploration_constant() {
    return exploration;
}


/// MonteCarlo::test()
void MonteCarlo::test() {
   debug << "---------------------------------------------------------------------------------" << endl;
   debug << "Testing MonteCarlo for position..." << endl;
   debug << pos << endl;

   search();

   debug << "... end of MonteCarlo testing!" << endl;
   debug << "---------------------------------------------------------------------------------" << endl;
}


/// MonteCarlo::print_stats()
void MonteCarlo::print_stats() {
   debug << "ply        = " << ply             << endl;
   debug << "descentCnt = " << descentCnt      << endl;
   debug << "playoutCnt = " << playoutCnt      << endl;
   debug << "doMoveCnt  = " << doMoveCnt       << endl;
   debug << "priorCnt   = " << priorCnt        << endl;
   debug << "hash size  = " << MCTS.size()     << endl;
}


/// MonteCarlo::print_node()
void MonteCarlo::print_node(Node node) {
   debug << "isCurrent    = " << (node == current_node()) << endl;
   debug << "isRoot       = " << (node == root)           << endl;
   debug << "key1         = " << node->key1               << endl;
   debug << "key2         = " << node->key2               << endl;
   debug << "visits       = " << node->visits             << endl;
   debug << "sons         = " << node->number_of_sons     << endl;
   debug << "expandedSons = " << node->expandedSons       << endl;
}


/// MonteCarlo::print_edge()
void MonteCarlo::print_edge(Edge e) {
   debug << "edge = { "
         << UCI::move(e.move, pos.is_chess960()) << " , "
         << "N = " << e.visits                     << " , "
         << "P = " << e.prior                      << " , "
         << "W = " << e.actionValue                << " , "
         << "Q = " << e.meanActionValue            << " }"
         << endl;
}

// List of FIXME/TODO for the monte-carlo branch
//
// 1. ttMove = MOVE_NONE    in generate_moves()
// 2. what to do with killers in create_root()
// 3. setupStates should probably come the caller, as a global in create_root()
// 4. debug the priors for the following key : 5DB5F8476356FB19


















