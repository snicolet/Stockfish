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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring> // std::memset()

namespace Eval::NNUE {

  // If vector instructions are enabled, we update and refresh the
  // accumulator tile by tile such that each tile fits in the CPU's
  // vector registers.
  #define VECTOR

  #if defined(USE_AVX512) || defined(USE_AVX2)
  typedef __m256i vec_t;
  #define vec_load(a) _mm256_load_si256(a)
  #define vec_store(a,b) _mm256_store_si256(a,b)
  #define vec_add_16(a,b) _mm256_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
  static constexpr IndexType kNumRegs = 16;

  #else
  #undef VECTOR

  #endif

  // Input feature converter
  class FeatureTransformer {

   private:
    // Number of output dimensions for one side
    static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

    #ifdef VECTOR
    static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
    static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
    #endif

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions = RawFeatures::kDimensions;
    static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

    // Size of forward propagation buffer
    static constexpr std::size_t kBufferSize =
        kOutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {

      return RawFeatures::kHashValue ^ kOutputDimensions;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {
      // TODO - hacking these down to 16 bits currently - should fix quantizer.
      scale_ = read_little_endian<std::int32_t>(stream) >> 16;
      scale_bits_ = read_little_endian<std::int32_t>(stream) - 16;
      weight_zero_point_ = read_little_endian<std::int32_t>(stream);

      // Note weights are stored as int8, but we expand to int16 for speed.
      for (std::size_t i = 0; i < kHalfDimensions; ++i)
        biases_[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
        weights_[i] = read_little_endian<std::int8_t>(stream) - weight_zero_point_;
      return !stream.fail();
    }

    // Convert input features
    void Transform(const Position& pos, OutputType* output) const {

      UpdateAccumulator(pos, WHITE);
      UpdateAccumulator(pos, BLACK);

      const auto& accumulation = pos.state()->accumulator.accumulation;

  #if defined(USE_AVX512) || defined(USE_AVX2)
      constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
      constexpr int kControl = 0b11011000;
      const __m256i kScale = _mm256_set1_epi32(scale_);
  #endif

      const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
      for (IndexType p = 0; p < 2; ++p) {
        const IndexType offset = kHalfDimensions * p;

  #if defined(USE_AVX512) || defined(USE_AVX2)
        auto out = reinterpret_cast<__m256i*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m256i sum0 = _mm256_load_si256(
              &reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
          __m256i sum1 = _mm256_load_si256(
              &reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
          sum0 = mul16x16(sum0, kScale, 15);
          sum1 = mul16x16(sum1, kScale, 15);
          _mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
              _mm256_packus_epi16(sum0, sum1), kControl));
        }

  #else
        for (IndexType j = 0; j < kHalfDimensions; ++j) {
          std::int32_t sum = accumulation[static_cast<int>(perspectives[p])][0][j];
          sum = rounding_shift(static_cast<std::int64_t>(sum) * scale_, scale_bits_);
          sum = std::max(std::min(sum, 255), 0);
          output[offset + j] = static_cast<OutputType>(sum);
        }
  #endif

      }
    }

   private:
    void UpdateAccumulator(const Position& pos, const Color c) const {

  #ifdef VECTOR
      // Gcc-10.2 unnecessarily spills AVX2 registers if this array
      // is defined in the VECTOR code below, once in each branch
      vec_t acc[kNumRegs];
  #endif

      // Look for a usable accumulator of an earlier position. We keep track
      // of the estimated gain in terms of features to be added/subtracted.
      StateInfo *st = pos.state(), *next = nullptr;
      int gain = pos.count<ALL_PIECES>() - 2;
      while (st->accumulator.state[c] == EMPTY)
      {
        auto& dp = st->dirtyPiece;
        // The first condition tests whether an incremental update is
        // possible at all: if this side's king has moved, it is not possible.
        static_assert(std::is_same_v<RawFeatures::SortedTriggerSet,
              Features::CompileTimeList<Features::TriggerEvent, Features::TriggerEvent::kFriendKingMoved>>,
              "Current code assumes that only kFriendlyKingMoved refresh trigger is being used.");
        if (   dp.piece[0] == make_piece(c, KING)
            || (gain -= dp.dirty_num + 1) < 0)
          break;
        next = st;
        st = st->previous;
      }

      if (st->accumulator.state[c] == COMPUTED)
      {
        if (next == nullptr)
          return;

        // Update incrementally in two steps. First, we update the "next"
        // accumulator. Then, we update the current accumulator (pos.state()).

        // Gather all features to be updated. This code assumes HalfKP features
        // only and doesn't support refresh triggers.
        static_assert(std::is_same_v<Features::FeatureSet<Features::HalfKP<Features::Side::kFriend>>,
                                     RawFeatures>);
        Features::IndexList removed[2], added[2];
        Features::HalfKP<Features::Side::kFriend>::AppendChangedIndices(pos,
            next->dirtyPiece, c, &removed[0], &added[0]);
        for (StateInfo *st2 = pos.state(); st2 != next; st2 = st2->previous)
          Features::HalfKP<Features::Side::kFriend>::AppendChangedIndices(pos,
              st2->dirtyPiece, c, &removed[1], &added[1]);

        // Mark the accumulators as computed.
        next->accumulator.state[c] = COMPUTED;
        pos.state()->accumulator.state[c] = COMPUTED;

        // Now update the accumulators listed in info[], where the last element is a sentinel.
        StateInfo *info[3] =
          { next, next == pos.state() ? nullptr : pos.state(), nullptr };
  #ifdef VECTOR
        for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j)
        {
          // Load accumulator
          auto accTile = reinterpret_cast<vec_t*>(
            &st->accumulator.accumulation[c][0][j * kTileHeight]);
          for (IndexType k = 0; k < kNumRegs; ++k)
            acc[k] = vec_load(&accTile[k]);

          for (IndexType i = 0; info[i]; ++i)
          {
            // Difference calculation for the deactivated features
            for (const auto index : removed[i])
            {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], column[k]);
            }

            // Difference calculation for the activated features
            for (const auto index : added[i])
            {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_add_16(acc[k], column[k]);
            }

            // Store accumulator
            accTile = reinterpret_cast<vec_t*>(
              &info[i]->accumulator.accumulation[c][0][j * kTileHeight]);
            for (IndexType k = 0; k < kNumRegs; ++k)
              vec_store(&accTile[k], acc[k]);
          }
        }

  #else
        for (IndexType i = 0; info[i]; ++i)
        {
          std::memcpy(info[i]->accumulator.accumulation[c][0],
              st->accumulator.accumulation[c][0],
              kHalfDimensions * sizeof(BiasType));
          st = info[i];

          // Difference calculation for the deactivated features
          for (const auto index : removed[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] -= weights_[offset + j];
          }

          // Difference calculation for the activated features
          for (const auto index : added[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] += weights_[offset + j];
          }
        }
  #endif
      }
      else
      {
        // Refresh the accumulator
        auto& accumulator = pos.state()->accumulator;
        accumulator.state[c] = COMPUTED;
        Features::IndexList active;
        Features::HalfKP<Features::Side::kFriend>::AppendActiveIndices(pos, c, &active);

  #ifdef VECTOR
        for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j)
        {
          auto biasesTile = reinterpret_cast<const vec_t*>(
              &biases_[j * kTileHeight]);
          for (IndexType k = 0; k < kNumRegs; ++k)
            acc[k] = biasesTile[k];

          for (const auto index : active)
          {
            const IndexType offset = kHalfDimensions * index + j * kTileHeight;
            auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);

            for (unsigned k = 0; k < kNumRegs; ++k)
              acc[k] = vec_add_16(acc[k], column[k]);
          }

          auto accTile = reinterpret_cast<vec_t*>(
              &accumulator.accumulation[c][0][j * kTileHeight]);
          for (unsigned k = 0; k < kNumRegs; k++)
            vec_store(&accTile[k], acc[k]);
        }

  #else
        std::memcpy(accumulator.accumulation[c][0], biases_,
            kHalfDimensions * sizeof(BiasType));

        for (const auto index : active)
        {
          const IndexType offset = kHalfDimensions * index;

          for (IndexType j = 0; j < kHalfDimensions; ++j)
            accumulator.accumulation[c][0][j] += weights_[offset + j];
        }
  #endif
      }
    }

    using BiasType = std::int16_t;
    using WeightType = std::int16_t;

    // Quantization parameters
    std::int32_t scale_;
    std::int32_t scale_bits_;
    std::int32_t weight_zero_point_;

    alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
    alignas(kCacheLineSize)
        WeightType weights_[kHalfDimensions * kInputDimensions];
  };

}  // namespace Eval::NNUE

#endif // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
