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

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>   // For std::memset
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <vector>

#include "bitboard.h"
#include "evaluate.h"
#include "material.h"
#include "misc.h"
#include "pawns.h"
#include "thread.h"
#include "timeman.h"
#include "uci.h"
#include "incbin/incbin.h"


// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
  INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#else
  const unsigned char        gEmbeddedNNUEData[1] = {0x0};
  const unsigned char *const gEmbeddedNNUEEnd = &gEmbeddedNNUEData[1];
  const unsigned int         gEmbeddedNNUESize = 1;
#endif


using namespace std;

namespace Stockfish {

namespace Eval {

  string eval_file_loaded = "None";

  /// NNUE::init() tries to load a NNUE network at startup time, or when the engine
  /// receives a UCI command "setoption name EvalFile value nn-[a-z0-9]{12}.nnue"
  /// The name of the NNUE network is always retrieved from the EvalFile option.
  /// We search the given network in three locations: internally (the default
  /// network may be embedded in the binary), in the active working directory and
  /// in the engine directory. Distro packagers may define the DEFAULT_NNUE_DIRECTORY
  /// variable to have the engine search in a special directory in their distro.

  void NNUE::init() {

    string eval_file = string(Options["EvalFile"]);

    #if defined(DEFAULT_NNUE_DIRECTORY)
    #define stringify2(x) #x
    #define stringify(x) stringify2(x)
    vector<string> dirs = { "<internal>" , "" , CommandLine::binaryDirectory , stringify(DEFAULT_NNUE_DIRECTORY) };
    #else
    vector<string> dirs = { "<internal>" , "" , CommandLine::binaryDirectory };
    #endif

    for (string directory : dirs)
        if (eval_file_loaded != eval_file)
        {
            if (directory != "<internal>")
            {
                ifstream stream(directory + eval_file, ios::binary);
                if (load_eval(eval_file, stream))
                    eval_file_loaded = eval_file;
            }

            if (directory == "<internal>" && eval_file == EvalFileDefaultName)
            {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) { setg(p, p, p + n); setp(p, p + n); }
                };

                MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(gEmbeddedNNUEData)),
                                    size_t(gEmbeddedNNUESize));

                istream stream(&buffer);
                if (load_eval(eval_file, stream))
                    eval_file_loaded = eval_file;
            }
        }
  }

  /// NNUE::verify() verifies that the last net used was loaded successfully
  void NNUE::verify() {

    string eval_file = string(Options["EvalFile"]);

    if (eval_file_loaded != eval_file)
    {
        UCI::OptionsMap defaults;
        UCI::init(defaults);

        string msg2 = "The network file " + eval_file + " was not loaded successfully.";
        string msg3 = "The UCI option EvalFile might need to specify the full path, including the directory name, to the network file.";
        string msg4 = "The default net can be downloaded from: https://tests.stockfishchess.org/api/nn/" + string(defaults["EvalFile"]);
        string msg5 = "The engine will be terminated now.";

        sync_cout << "info string ERROR: " << msg2 << sync_endl;
        sync_cout << "info string ERROR: " << msg3 << sync_endl;
        sync_cout << "info string ERROR: " << msg4 << sync_endl;
        sync_cout << "info string ERROR: " << msg5 << sync_endl;

        exit(EXIT_FAILURE);
    }

    sync_cout << "info string NNUE evaluation using " << eval_file << " enabled" << sync_endl;
  }
}


namespace {

  constexpr Value CorneredBishop = Value(50);

  /// Fisher Random Chess: correction for cornered bishops, to fix chess960 play with NNUE

  Value fix_FRC(const Position& pos) {

    constexpr Bitboard Corners =  1ULL << SQ_A1 | 1ULL << SQ_H1 | 1ULL << SQ_A8 | 1ULL << SQ_H8;

    if (!(pos.pieces(BISHOP) & Corners))
        return VALUE_ZERO;

    int correction = 0;

    if (   pos.piece_on(SQ_A1) == W_BISHOP
        && pos.piece_on(SQ_B2) == W_PAWN)
        correction += !pos.empty(SQ_B3) ? -CorneredBishop * 4
                                        : -CorneredBishop * 3;

    if (   pos.piece_on(SQ_H1) == W_BISHOP
        && pos.piece_on(SQ_G2) == W_PAWN)
        correction += !pos.empty(SQ_G3) ? -CorneredBishop * 4
                                        : -CorneredBishop * 3;

    if (   pos.piece_on(SQ_A8) == B_BISHOP
        && pos.piece_on(SQ_B7) == B_PAWN)
        correction += !pos.empty(SQ_B6) ? CorneredBishop * 4
                                        : CorneredBishop * 3;

    if (   pos.piece_on(SQ_H8) == B_BISHOP
        && pos.piece_on(SQ_G7) == B_PAWN)
        correction += !pos.empty(SQ_G6) ? CorneredBishop * 4
                                        : CorneredBishop * 3;

    return pos.side_to_move() == WHITE ?  Value(correction)
                                       : -Value(correction);
  }

} // namespace Eval


/// evaluate() is the evaluator for the outer world. It returns a static
/// evaluation of the position from the point of view of the side to move.

Value Eval::evaluate(const Position& pos) {

  // If there is PSQ imbalance we use a faster eval as a proxy, but we switch
  // to full NNUE eval when shuffling or if the material on the board is high.
  int r50 = pos.rule50_count();
  Value psq = Value(abs(eg_value(pos.psq_score())));
  bool faster = psq * 5 > (750 + pos.non_pawn_material() / 64) * (5 + r50);

  int scale =   903
              + 32 * pos.count<PAWN>()
              + 32 * pos.non_pawn_material() / 1024;

  Value v = faster ? NNUE::materialist(pos)
                   : NNUE::evaluate(pos, true);

  v = v * scale / 1024;

  if (pos.is_chess960())
     v += fix_FRC(pos);

  // Damp down the evaluation linearly when shuffling
  v = v * (100 - pos.rule50_count()) / 100;

  // Guarantee evaluation does not hit the tablebase range
  v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

  return v;
}

/// trace() is like evaluate(), but instead of returning a value, it returns
/// a string (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.
/// Trace scores are from white's point of view

std::string Eval::trace(Position& pos) {

  auto to_cp = [&](Value v) { return double(v) / PawnValueEg; };

  if (pos.checkers())
      return "Final evaluation: none (in check)";

  std::stringstream ss;
  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);

  ss << '\n' << NNUE::trace(pos) << '\n';

  ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

  Value v = NNUE::evaluate(pos, false);
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "NNUE evaluation        " << to_cp(v) << " (white side)\n";

  v = evaluate(pos);
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "Final evaluation       " << to_cp(v) << " (white side)";
  ss << " [with scaled NNUE, hybrid, ...]";
  ss << "\n";

  return ss.str();
}

} // namespace Stockfish
