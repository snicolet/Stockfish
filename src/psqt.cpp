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

#include <algorithm>

#include "types.h"

Value PieceValue[PHASE_NB][PIECE_NB] = {
  { VALUE_ZERO, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg },
  { VALUE_ZERO, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg }
};

namespace PSQT {

#define S(mg, eg) make_score(mg, eg)

// Bonus[PieceType][Square / 2] contains Piece-Square scores. For each piece
// type on a given square a (middlegame, endgame) score pair is assigned. Table
// is defined for files A..D and white side: it is symmetric for black side and
// second half of the files.
constexpr Score Bonus[][RANK_NB][int(FILE_NB) / 2] = {
  { },
  { },
  { // Knight
   { S(-169,-105), S(-96,-74), S(-80,-46), S(-79,-18) },
   { S( -79, -70), S(-39,-56), S(-24,-15), S( -9,  6) },
   { S( -64, -38), S(-20,-33), S(  4, -5), S( 19, 27) },
   { S( -28, -36), S(  5,  0), S( 41, 13), S( 47, 34) },
   { S( -29, -41), S( 13,-20), S( 42,  4), S( 52, 35) },
   { S( -11, -51), S( 28,-38), S( 63,-17), S( 55, 19) },
   { S( -67, -64), S(-21,-45), S(  6,-37), S( 37, 16) },
   { S(-200, -98), S(-80,-89), S(-53,-53), S(-32,-16) }
  },
  { // Bishop
   { S(-44,-63), S( -4,-30), S(-11,-35), S(-28, -8) },
   { S(-18,-38), S(  7,-13), S( 14,-14), S(  3,  0) },
   { S( -8,-18), S( 24,  0), S( -3, -7), S( 15, 13) },
   { S(  1,-26), S(  8, -3), S( 26,  1), S( 37, 16) },
   { S( -7,-24), S( 30, -6), S( 23,-10), S( 28, 17) },
   { S(-17,-26), S(  4,  2), S( -1,  1), S(  8, 16) },
   { S(-21,-34), S(-19,-18), S( 10, -7), S( -6,  9) },
   { S(-48,-51), S( -3,-40), S(-12,-39), S(-25,-20) }
  },
  { // Rook
   { S(-24, -2), S(-13,-6), S(-7, -3), S( 2,-2) },
   { S(-18,-10), S(-10,-7), S(-5,  1), S( 9, 0) },
   { S(-21, 10), S( -7,-4), S( 3,  2), S(-1,-2) },
   { S(-13, -5), S( -5, 2), S(-4, -8), S(-6, 8) },
   { S(-24, -8), S(-12, 5), S(-1,  4), S( 6,-9) },
   { S(-24,  3), S( -4,-2), S( 4,-10), S(10, 7) },
   { S( -8,  1), S(  6, 2), S(10, 17), S(12,-8) },
   { S(-22, 12), S(-24,-6), S(-6, 13), S( 4, 7) }
  },
  { // Queen
   { S( 3,-69), S(-5,-57), S(-5,-47), S( 4,-26) },
   { S(-3,-55), S( 5,-31), S( 8,-22), S(12, -4) },
   { S(-3,-39), S( 6,-18), S(13, -9), S( 7,  3) },
   { S( 4,-23), S( 5, -3), S( 9, 13), S( 8, 24) },
   { S( 0,-29), S(14, -6), S(12,  9), S( 5, 21) },
   { S(-4,-38), S(10,-18), S( 6,-12), S( 8,  1) },
   { S(-5,-50), S( 6,-27), S(10,-24), S( 8, -8) },
   { S(-2,-75), S(-2,-52), S( 1,-43), S(-2,-36) }
  },
  { // King
   { S(272,  0), S(325, 41), S(273, 80), S(190, 93) },
   { S(277, 57), S(305, 98), S(241,138), S(183,131) },
   { S(198, 86), S(253,138), S(168,165), S(120,173) },
   { S(169,103), S(191,152), S(136,168), S(108,169) },
   { S(145, 98), S(176,166), S(112,197), S( 69,194) },
   { S(122, 87), S(159,164), S( 85,174), S( 36,189) },
   { S( 87, 40), S(120, 99), S( 64,128), S( 25,141) },
   { S( 64,  5), S( 87, 60), S( 49, 75), S(  0, 75) }
  }
};

#define R2(a,b)  S(  0 + (a), (b) )
#define R3(a,b)  S(  2 + (a), (b) )
#define R4(a,b)  S(  4 + (a), (b) )
#define R5(a,b)  S(  8 + (a), (b) )
#define R6(a,b)  S( 12 + (a), (b) )
#define R7(a,b)  S( 18 + (a), (b) )

constexpr Score PBonus[RANK_NB][FILE_NB] =
  { // Pawn (asymmetric distribution)
   { },
   { R2(  0,-10), R2( -5,-3), R2( 10, 7), R2( 13,-1), R2( 21,  7), R2( 17,  6), R2(  6, 1), R2( -3,-20) },
   { R3(-11, -6), R3(-10,-6), R3( 15,-1), R3( 22,-1), R3( 26, -1), R3( 28,  2), R3(  4,-2), R3(-24, -5) },
   { R4( -9,  4), R4(-18,-5), R4(  8,-4), R4( 22,-5), R4( 33, -6), R4( 25,-13), R4( -4,-3), R4(-16, -7) },
   { R5(  6, 18), R5( -3, 2), R5(-10, 2), R5(  1,-9), R5( 12,-13), R5(  6, -8), R5(-12,11), R5(  1,  9) },
   { R6( -6, 25), R6( -8,17), R6(  5,19), R6( 11,29), R6(-14, 29), R6(  0,  8), R6(-12, 4), R6(-14, 12) },
   { R7(-10, -1), R7(  6,-6), R7( -5,18), R7(-11,22), R7( -2, 22), R7(-14, 17), R7( 12, 2), R7( -1,  9) }
  };
  
#undef R2
#undef R3
#undef R4
#undef R5
#undef R6
#undef R7
#undef S

Score psq[PIECE_NB][SQUARE_NB];

// init() initializes piece-square tables: the white halves of the tables are
// copied from Bonus[] adding the piece value, then the black halves of the
// tables are initialized by flipping and changing the sign of the white scores.
void init() {

  for (Piece pc = W_PAWN; pc <= W_KING; ++pc)
  {
      PieceValue[MG][~pc] = PieceValue[MG][pc];
      PieceValue[EG][~pc] = PieceValue[EG][pc];

      Score score = make_score(PieceValue[MG][pc], PieceValue[EG][pc]);

      for (Square s = SQ_A1; s <= SQ_H8; ++s)
      {
          File f = std::min(file_of(s), ~file_of(s));
          psq[ pc][ s] = score + (type_of(pc) == PAWN ? PBonus[rank_of(s)][file_of(s)]
                                                      : Bonus[pc][rank_of(s)][f]);
          psq[~pc][~s] = -psq[pc][s];
      }
  }
}

} // namespace PSQT
