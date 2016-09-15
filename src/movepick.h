/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include <algorithm> // For std::max
#include <cstring>   // For std::memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"


/// The Stats struct stores moves statistics. According to the template parameter
/// the class can store History and Countermoves. History records how often
/// different moves have been successful or unsuccessful during the current search
/// and is used for reduction and move ordering decisions.
/// Countermoves store the move that refute a previous one. Entries are stored
/// using only the moving piece and destination square, hence two moves with
/// different origin but same destination and piece will be considered identical.
template<typename T, bool CM = false>
struct Stats {

  static const Value Max = Value(1 << 28);

  const T* operator[](Piece pc) const { return table[pc]; }
  T* operator[](Piece pc) { return table[pc]; }
  void clear() { std::memset(table, 0, sizeof(table)); }

  void update(Piece pc, Square to, Move m) { table[pc][to] = m; }

  void update(Piece pc, Square to, Value v) {

    if (abs(int(v)) >= 324)
        return;

    table[pc][to] -= table[pc][to] * abs(int(v)) / (CM ? 936 : 324);
    table[pc][to] += int(v) * 32;
  }

private:
  T table[PIECE_NB][SQUARE_NB];
};

typedef Stats<Move> MoveStats;
typedef Stats<Value, false> HistoryStats;
typedef Stats<Value,  true> CounterMoveStats;
typedef Stats<CounterMoveStats> CounterMoveHistoryStats;

struct FromToStats {

    Value get(Color c, Move m) const { return table[c][from_sq(m)][to_sq(m)]; }
    void clear() { std::memset(table, 0, sizeof(table)); }

    void update(Color c, Move m, Value v)
    {
        if (abs(int(v)) >= 324)
            return;

        Square f = from_sq(m);
        Square t = to_sq(m);

        table[c][f][t] -= table[c][f][t] * abs(int(v)) / 324;
        table[c][f][t] += int(v) * 32;
    }

private:
    Value table[COLOR_NB][SQUARE_NB][SQUARE_NB];
};

enum Stages {
    MAIN_SEARCH, GOOD_CAPTURES, KILLERS, QUIET, BAD_CAPTURES,
    EVASION, ALL_EVASIONS,
    QSEARCH_WITH_CHECKS, QCAPTURES_1, CHECKS,
    QSEARCH_WITHOUT_CHECKS, QCAPTURES_2,
    PROBCUT, PROBCUT_CAPTURES,
    RECAPTURE, RECAPTURES,
    STOP,
    STAGE_NB = STOP
  };
  
 typedef void Generator(void);
 typedef Move Picker(void);
 struct GeneratorAndPicker { Generator g; Picker p; };

/// MovePicker class is used to pick one pseudo legal move at a time from the
/// current position. The most important method is next_move(), which returns a
/// new pseudo legal move each time it is called, until there are no moves left,
/// when MOVE_NONE is returned. In order to improve the efficiency of the alpha
/// beta algorithm, MovePicker attempts to return the moves which are most likely
/// to get a cut-off first.

class MovePicker {
public:
  MovePicker(const MovePicker&) = delete;
  MovePicker& operator=(const MovePicker&) = delete;

  MovePicker(const Position&, Move, Value);
  MovePicker(const Position&, Move, Depth, Square);
  MovePicker(const Position&, Move, Depth, Search::Stack*);

  Move next_move();
  int see_sign() const;
  
  template<Stages> void gns();

private:
  template<GenType> void score();
  void generate_next_stage();
  
  ExtMove* begin() { return moves; }
  ExtMove* end() { return endMoves; }

  const Position& pos;
  const Search::Stack* ss;
  Move countermove;
  Depth depth;
  Move ttMove;
  ExtMove killers[3];
  Square recaptureSquare;
  Value threshold;
  int stage;
  ExtMove* endBadCaptures = moves + MAX_MOVES - 1;
  ExtMove moves[MAX_MOVES], *cur = moves, *endMoves = moves;
  
  void gns_GOOD_CAPTURES();
  void gns_QCAPTURES_1();
  void gns_QCAPTURES_2();
  void gns_PROBCUT_CAPTURES();
  void gns_RECAPTURES();
  void gns_KILLERS();
  void gns_QUIET();
  void gns_BAD_CAPTURES();
  void gns_ALL_EVASIONS();
  void gns_CHECKS();
  void gns_QSEARCH_WITH_CHECKS();
  void gns_QSEARCH_WITHOUT_CHECKS();
  void gns_PROBCUT();
  void gns_RECAPTURE();
  void gns_STOP();
  
  Generator x = &MovePicker::gns_GOOD_CAPTURES;
  
  //GeneratorAndPicker x = { &MovePicker::gns_GOOD_CAPTURES , 0};
  
 // GeneratorAndPicker func[STAGE_NB] =
    //  {gns_GOOD_CAPTURES , null_pointer}

};

inline void MovePicker::gns_GOOD_CAPTURES()          { gns<GOOD_CAPTURES>(); }
inline void MovePicker::gns_QCAPTURES_1()            { gns<QCAPTURES_1>(); }
inline void MovePicker::gns_QCAPTURES_2()            { gns<QCAPTURES_2>(); }
inline void MovePicker::gns_PROBCUT_CAPTURES()       { gns<PROBCUT_CAPTURES>(); }
inline void MovePicker::gns_RECAPTURES()             { gns<RECAPTURES>(); }
inline void MovePicker::gns_KILLERS()                { gns<KILLERS>(); }
inline void MovePicker::gns_QUIET()                  { gns<QUIET>(); }
inline void MovePicker::gns_BAD_CAPTURES()           { gns<BAD_CAPTURES>(); }
inline void MovePicker::gns_ALL_EVASIONS()           { gns<ALL_EVASIONS>(); }
inline void MovePicker::gns_CHECKS()                 { gns<CHECKS>(); }
inline void MovePicker::gns_QSEARCH_WITH_CHECKS()    { gns<QSEARCH_WITH_CHECKS>(); }
inline void MovePicker::gns_QSEARCH_WITHOUT_CHECKS() { gns<QSEARCH_WITHOUT_CHECKS>(); }
inline void MovePicker::gns_PROBCUT()                { gns<PROBCUT>(); }
inline void MovePicker::gns_RECAPTURE()              { gns<RECAPTURE>(); }
inline void MovePicker::gns_STOP()                   { gns<STOP>(); }







#endif // #ifndef MOVEPICK_H_INCLUDED
