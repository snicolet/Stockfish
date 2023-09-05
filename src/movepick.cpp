/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

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

#include "movepick.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

  enum Stages {
    MAIN_TT, CAPTURE_INIT, GOOD_CAPTURE, REFUTATION, QUIET_INIT, QUIET, BAD_CAPTURE,
    EVASION_TT, EVASION_INIT, EVASION,
    PROBCUT_TT, PROBCUT_INIT, PROBCUT,
    QSEARCH_TT, QCAPTURE_INIT, QCAPTURE, QCHECK_INIT, QCHECK
  };

  // partial_insertion_sort() sorts moves in descending order up to and including
  // a given limit. The order of moves smaller than the limit is left unspecified.
  void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
  }

} // namespace


/// Constructors of the MovePicker class. As arguments we pass information
/// to help it to return the (presumably) good moves first, to decide which
/// moves to return (in the quiescence search, for instance, we only want to
/// search captures, promotions, and some checks) and how important good move
/// ordering is at the current node.

/// MovePicker constructor for the main search
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
                                                             const CapturePieceToHistory* cph,
                                                             const PieceToHistory** ch,
                                                             Move cm,
                                                             const Move* killers)
           : pos(p), mainHistory(mh), captureHistory(cph), continuationHistory(ch),
             ttMove(ttm), refutations{{killers[0], 0}, {killers[1], 0}, {cm, 0}}, depth(d)
{
  assert(d > 0);

  stage = (pos.checkers() ? EVASION_TT : MAIN_TT) +
          !(ttm && pos.pseudo_legal(ttm));
}

/// MovePicker constructor for quiescence search
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
                                                             const CapturePieceToHistory* cph,
                                                             const PieceToHistory** ch,
                                                             Square rs)
           : pos(p), mainHistory(mh), captureHistory(cph), continuationHistory(ch), ttMove(ttm), recaptureSquare(rs), depth(d)
{
  assert(d <= 0);

  stage = (pos.checkers() ? EVASION_TT : QSEARCH_TT) +
          !(   ttm
            && pos.pseudo_legal(ttm));
}

/// MovePicker constructor for ProbCut
/// We generate captures with SEE greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, Value th, const ButterflyHistory* mh,
                                                              const CapturePieceToHistory* cph,
                                                              const PieceToHistory** ch)
           : pos(p), mainHistory(mh), captureHistory(cph), continuationHistory(ch), ttMove(ttm), threshold(th)
{
  assert(!pos.checkers());

  stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm)
                             && pos.pseudo_legal(ttm)
                             && pos.see_ge(ttm, threshold));
}

/// MovePicker::score() assigns a numerical value to each move in a list, used
/// for sorting. Captures are ordered by Most Valuable Victim (MVV), preferring
/// captures with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
void MovePicker::score() {

  static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

  [[maybe_unused]] Bitboard threatenedByPawn, threatenedByMinor, threatenedByRook, threatenedPieces;
  if constexpr (Type == QUIETS || Type == CAPTURES)
  {
      Color us = pos.side_to_move();

      threatenedByPawn  = pos.attacks_by<PAWN>(~us);
      threatenedByMinor = pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatenedByPawn;
      threatenedByRook  = pos.attacks_by<ROOK>(~us) | threatenedByMinor;

      // Pieces threatened by pieces of lesser material value
      threatenedPieces = (pos.pieces(us, QUEEN) & threatenedByRook)
                       | (pos.pieces(us, ROOK)  & threatenedByMinor)
                       | (pos.pieces(us, KNIGHT, BISHOP) & threatenedByPawn);
  }

  for (auto& m : *this)

      if constexpr (Type == QUIETS || Type == CAPTURES)
      {
          Piece     pc       = pos.moved_piece(m);
          PieceType pt       = type_of(pos.moved_piece(m));
          Square    from     = from_sq(m);
          Square    to       = to_sq(m);
          Piece     captured = pos.piece_on(to);

          int checkBonus          = bool(pos.check_squares(pt) & to) * 16384;
          int victimBonus         = 24 * int(PieceValue[captured]);
          int captureHistoryBonus =  4 * (*captureHistory)[pc][to][type_of(captured)];
          int mainHistoryBonus    =  2 * (*mainHistory)[pos.side_to_move()][from_to(m)];
          int continuationBonus   =  2 * (*continuationHistory[0])[pc][to]
                                    +    (*continuationHistory[1])[pc][to]
                                    +    (*continuationHistory[3])[pc][to]
                                    +    (*continuationHistory[5])[pc][to];
          int escapeBonus         =  threatenedPieces & from ?
                                      (pt == QUEEN && !(to & threatenedByRook)  ? 50000
                                     : pt == ROOK  && !(to & threatenedByMinor) ? 25000
                                     :                !(to & threatenedByPawn)  ? 15000
                                     :                                            0 )
                                     :                                            0 ;
          int enPriseBonus        =  !(threatenedPieces & from) ?
                                       (pt == QUEEN ?   bool(to & threatenedByRook)  * 50000
                                                      + bool(to & threatenedByMinor) * 10000
                                                      + bool(to & threatenedByPawn)  * 20000
                                      : pt == ROOK  ?   bool(to & threatenedByMinor) * 25000
                                                      + bool(to & threatenedByPawn)  * 10000
                                      : pt != PAWN ?    bool(to & threatenedByPawn)  * 15000
                                      :                                                0 )
                                      :                                                0 ;

          if constexpr (Type == QUIETS)
             m.value =   checkBonus
                       + escapeBonus
                       + mainHistoryBonus
                       + continuationBonus
                       - enPriseBonus;

          if constexpr (Type == CAPTURES)
             m.value =   victimBonus
                       + captureHistoryBonus;
                       
           // dbg_mean_of(quietBonus                     , 0);
           // dbg_mean_of(captureBonus                   , 1);
           // dbg_mean_of(mainHistoryBonus               , 2);
           // dbg_mean_of(continuationBonus              , 3);
           // dbg_mean_of(escapeBonus                    , 4);
           // dbg_mean_of(-enPriseBonus                  , 5);

           // dbg_mean_of(std::max(continuationBonus, 0) , 6);
           // dbg_mean_of(std::max(mainHistoryBonus , 0) , 7);
      }

      else // Type == EVASIONS
      {
          if (pos.capture_stage(m))
              m.value =  PieceValue[pos.piece_on(to_sq(m))]
                       - Value(type_of(pos.moved_piece(m)))
                       + (1 << 28);
          else
              m.value =  (*mainHistory)[pos.side_to_move()][from_to(m)]
                       + (*continuationHistory[0])[pos.moved_piece(m)][to_sq(m)];
      }
}

/// MovePicker::select() returns the next move satisfying a predicate function.
/// It never returns the TT move.
template<MovePicker::PickType T, typename Pred>
Move MovePicker::select(Pred filter) {

  while (cur < endMoves)
  {
      if constexpr (T == Best)
          std::swap(*cur, *std::max_element(cur, endMoves));

      if (*cur != ttMove && filter())
          return *cur++;

      cur++;
  }
  return MOVE_NONE;
}

/// MovePicker::next_move() is the most important method of the MovePicker class. It
/// returns a new pseudo-legal move every time it is called until there are no more
/// moves left, picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move(bool skipQuiets) {

top:
  switch (stage) {

  case MAIN_TT:
  case EVASION_TT:
  case QSEARCH_TT:
  case PROBCUT_TT:
      ++stage;
      return ttMove;

  case CAPTURE_INIT:
  case PROBCUT_INIT:
  case QCAPTURE_INIT:
      cur = endBadCaptures = moves;
      endMoves = generate<CAPTURES>(pos, cur);

      score<CAPTURES>();
      partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());
      ++stage;
      goto top;

  case GOOD_CAPTURE:
      if (select<Next>([&](){
                       return pos.see_ge(*cur, Value(-cur->value / 64)) ?
                              // Move losing capture to endBadCaptures to be tried later
                              true : (*endBadCaptures++ = *cur, false); }))
          return *(cur - 1);

      // Prepare the pointers to loop over the refutations array
      cur = std::begin(refutations);
      endMoves = std::end(refutations);

      // If the countermove is the same as a killer, skip it
      if (   refutations[0].move == refutations[2].move
          || refutations[1].move == refutations[2].move)
          --endMoves;

      ++stage;
      [[fallthrough]];

  case REFUTATION:
      if (select<Next>([&](){ return    *cur != MOVE_NONE
                                    && !pos.capture_stage(*cur)
                                    &&  pos.pseudo_legal(*cur); }))
          return *(cur - 1);
      ++stage;
      [[fallthrough]];

  case QUIET_INIT:
      if (!skipQuiets)
      {
          cur = endBadCaptures;
          endMoves = generate<QUIETS>(pos, cur);

          score<QUIETS>();
          partial_insertion_sort(cur, endMoves, -3000 * depth);
      }

      ++stage;
      [[fallthrough]];

  case QUIET:
      if (   !skipQuiets
          && select<Next>([&](){return   *cur != refutations[0].move
                                      && *cur != refutations[1].move
                                      && *cur != refutations[2].move;}))
          return *(cur - 1);

      // Prepare the pointers to loop over the bad captures
      cur = moves;
      endMoves = endBadCaptures;

      ++stage;
      [[fallthrough]];

  case BAD_CAPTURE:
      return select<Next>([](){ return true; });

  case EVASION_INIT:
      cur = moves;
      endMoves = generate<EVASIONS>(pos, cur);

      score<EVASIONS>();
      ++stage;
      [[fallthrough]];

  case EVASION:
      return select<Best>([](){ return true; });

  case PROBCUT:
      return select<Next>([&](){ return pos.see_ge(*cur, threshold); });

  case QCAPTURE:
      if (select<Next>([&](){ return   depth > DEPTH_QS_RECAPTURES
                                    || to_sq(*cur) == recaptureSquare; }))
          return *(cur - 1);

      // If we did not find any move and we do not try checks, we have finished
      if (depth != DEPTH_QS_CHECKS)
          return MOVE_NONE;

      ++stage;
      [[fallthrough]];

  case QCHECK_INIT:
      cur = moves;
      endMoves = generate<QUIET_CHECKS>(pos, cur);

      ++stage;
      [[fallthrough]];

  case QCHECK:
      return select<Next>([](){ return true; });
  }

  assert(false);
  return MOVE_NONE; // Silence warning
}

} // namespace Stockfish
