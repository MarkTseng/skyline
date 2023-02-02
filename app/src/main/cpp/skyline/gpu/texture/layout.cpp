// SPDX-License-Identifier: MPL-2.0
// Copyright © 2022 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "layout.h"

namespace skyline::gpu::texture {
    #pragma pack(push, 0)
    struct u96 {
        u64 high;
        u32 low;
    };
    #pragma pack(pop)

    // Reference on Block-linear tiling: https://gist.github.com/PixelyIon/d9c35050af0ef5690566ca9f0965bc32
    constexpr size_t SectorWidth{16}; //!< The width of a sector in bytes
    constexpr size_t SectorHeight{2}; //!< The height of a sector in lines
    constexpr size_t GobWidth{64}; //!< The width of a GOB in bytes
    constexpr size_t GobHeight{8}; //!< The height of a GOB in lines
    constexpr size_t SectorLinesInGob{(GobWidth / SectorWidth) * GobHeight}; //!< The number of lines of sectors inside a GOB

    size_t GetBlockLinearLayerSize(Dimensions dimensions, size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, size_t gobBlockHeight, size_t gobBlockDepth) {
        size_t robLineWidth{util::DivideCeil<size_t>(dimensions.width, formatBlockWidth)}; //!< The width of the ROB in terms of format blocks
        size_t robLineBytes{util::AlignUp(robLineWidth * formatBpb, GobWidth)}; //!< The amount of bytes in a single block

        size_t robHeight{GobHeight * gobBlockHeight}; //!< The height of a single ROB (Row of Blocks) in lines
        size_t surfaceHeightLines{util::DivideCeil<size_t>(dimensions.height, formatBlockHeight)}; //!< The height of the surface in lines
        size_t surfaceHeightRobs{util::DivideCeil(surfaceHeightLines, robHeight)}; //!< The height of the surface in ROBs (Row Of Blocks, incl. padding ROB)

        size_t robDepth{util::AlignUp(dimensions.depth, gobBlockDepth)}; //!< The depth of the surface in slices, aligned to include padding Z-axis GOBs

        return robLineBytes * robHeight * surfaceHeightRobs * robDepth;
    }

    template<typename Type>
    constexpr Type CalculateBlockGobs(Type blockGobs, Type surfaceGobs) {
        if (surfaceGobs > blockGobs)
            return blockGobs;
        return std::bit_ceil<Type>(surfaceGobs);
    }

    size_t GetBlockLinearLayerSize(Dimensions dimensions, size_t formatBlockHeight, size_t formatBlockWidth, size_t formatBpb, size_t gobBlockHeight, size_t gobBlockDepth, size_t levelCount, bool isMultiLayer) {
        // Calculate the size of the surface in GOBs on every axis
        size_t gobsWidth{util::DivideCeil<size_t>(util::DivideCeil<size_t>(dimensions.width, formatBlockWidth) * formatBpb, GobWidth)};
        size_t gobsHeight{util::DivideCeil<size_t>(util::DivideCeil<size_t>(dimensions.height, formatBlockHeight), GobHeight)};
        size_t gobsDepth{dimensions.depth};

        size_t totalSize{}, layerAlignment{GobWidth * GobHeight * gobBlockHeight * gobBlockDepth};
        for (size_t i{}; i < levelCount; i++) {
            // Iterate over every level, adding the size of the current level to the total size
            totalSize += (GobWidth * gobsWidth) * (GobHeight * util::AlignUp(gobsHeight, gobBlockHeight)) * util::AlignUp(gobsDepth, gobBlockDepth);

            // Successively divide every dimension by 2 until the final level is reached
            gobsWidth = std::max(gobsWidth / 2, 1UL);
            gobsHeight = std::max(gobsHeight / 2, 1UL);
            gobsDepth = std::max(gobsDepth / 2, 1UL);

            gobBlockHeight = CalculateBlockGobs(gobBlockHeight, gobsHeight);
            gobBlockDepth = CalculateBlockGobs(gobBlockDepth, gobsDepth);
        }

        return isMultiLayer ? util::AlignUp(totalSize, layerAlignment) : totalSize;
    }

    std::vector<MipLevelLayout> GetBlockLinearMipLayout(Dimensions dimensions, size_t formatBlockHeight, size_t formatBlockWidth, size_t formatBpb, size_t targetFormatBlockHeight, size_t targetFormatBlockWidth, size_t targetFormatBpb, size_t gobBlockHeight, size_t gobBlockDepth, size_t levelCount) {
        std::vector<MipLevelLayout> mipLevels;
        mipLevels.reserve(levelCount);

        size_t gobsWidth{util::DivideCeil<size_t>(util::DivideCeil<size_t>(dimensions.width, formatBlockWidth) * formatBpb, GobWidth)};
        size_t gobsHeight{util::DivideCeil<size_t>(util::DivideCeil<size_t>(dimensions.height, formatBlockHeight), GobHeight)};
        // Note: We don't need a separate gobsDepth variable here, since a GOB is always a single slice deep and the value would be the same as the depth dimension

        for (size_t i{}; i < levelCount; i++) {
            size_t linearSize{util::DivideCeil<size_t>(dimensions.width, formatBlockWidth) * formatBpb * util::DivideCeil<size_t>(dimensions.height, formatBlockHeight) * dimensions.depth};
            size_t targetLinearSize{targetFormatBpb == 0 ? linearSize : util::DivideCeil<size_t>(dimensions.width, targetFormatBlockWidth) * targetFormatBpb * util::DivideCeil<size_t>(dimensions.height, targetFormatBlockHeight) * dimensions.depth};

            mipLevels.emplace_back(
                dimensions,
                linearSize,
                targetLinearSize,
                (GobWidth * gobsWidth) * (GobHeight * util::AlignUp(gobsHeight, gobBlockHeight)) * util::AlignUp(dimensions.depth, gobBlockDepth),
                gobBlockHeight, gobBlockDepth
            );

            gobsWidth = std::max(gobsWidth / 2, 1UL);
            gobsHeight = std::max(gobsHeight / 2, 1UL);

            dimensions.width = std::max(dimensions.width / 2, 1U);
            dimensions.height = std::max(dimensions.height / 2, 1U);
            dimensions.depth = std::max(dimensions.depth / 2, 1U);

            gobBlockHeight = CalculateBlockGobs(gobBlockHeight, gobsHeight);
            gobBlockDepth = CalculateBlockGobs(gobBlockDepth, static_cast<size_t>(dimensions.depth));
        }

        return mipLevels;
    }

    /**
     * @brief Copies pixel data between a pitch-linear and blocklinear texture
     * @tparam BlockLinearToPitch Whether to copy from a blocklinear texture to a pitch-linear texture or a pitch-linear texture to a blocklinear texture
     */
    template<bool BlockLinearToPitch>
    void CopyBlockLinearInternal(Dimensions dimensions,
                                 size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                 size_t gobBlockHeight, size_t gobBlockDepth,
                                 u8 *blockLinear, u8 *pitch) {
        size_t textureWidth{util::DivideCeil<size_t>(dimensions.width, formatBlockWidth)};
        size_t textureWidthBytes{textureWidth * formatBpb};
        size_t textureWidthAlignedBytes{util::AlignUp(textureWidthBytes, GobWidth)};

        if (formatBpb != 12) {
            while (formatBpb != 16) {
                if (util::IsAligned(textureWidth, formatBpb << 1)) {
                    textureWidth /= 2;
                    formatBpb <<= 1;
                } else {
                    break;
                }
            }
        }

        size_t textureHeight{util::DivideCeil<size_t>(dimensions.height, formatBlockHeight)};
        size_t robHeight{gobBlockHeight * GobHeight};

        size_t alignedDepth{util::AlignUp(dimensions.depth, gobBlockDepth)};

        size_t pitchBytes{pitchAmount ? pitchAmount : textureWidthBytes};

        size_t blockSize{robHeight * GobWidth * alignedDepth};

        u8 *pitchOffset{pitch};

        auto copyTexture{[&]<typename FORMATBPB>() __attribute((always_inline)) {
            for (size_t slice{}; slice < dimensions.depth; ++slice, blockLinear += (GobHeight * GobWidth * gobBlockHeight)) {
                for (size_t line{}; line < textureHeight; ++line, pitchOffset += pitchBytes) {
                    size_t robOffset{textureWidthAlignedBytes * util::AlignDown(line, robHeight) * alignedDepth};
                    size_t blockHeight{(line & (robHeight - 1)) / GobHeight};
                    // Y Offset in entire GOBs
                    size_t GobYOffset{blockHeight * GobWidth * GobHeight};
                    // Y Offset inside current GOB
                    GobYOffset += (((line & 0x07) >> 1) << 6) + ((line & 0x01) << 4);

                    u8 *deSwizzledOffset{pitchOffset};
                    u8 *swizzledYZOffset{blockLinear + robOffset + GobYOffset};

                    // Copy per every element for the last GOB on X
                    for (size_t pixel{}; pixel < textureWidth; ++pixel, deSwizzledOffset += formatBpb) {
                        size_t xBytes{pixel * formatBpb};
                        size_t blockOffset{(xBytes / GobWidth) * blockSize};

                        // Set offset on X
                        size_t GobXOffset{(((xBytes & 0x3F) >> 5) << 8) + (xBytes & 0xF) + (((xBytes & 0x1F) >> 4) << 5)};
                        u8 *swizzledOffset{swizzledYZOffset + blockOffset + GobXOffset};

                        if constexpr (BlockLinearToPitch)
                            *(FORMATBPB *)deSwizzledOffset = *(FORMATBPB *)swizzledOffset;
                        else
                            *(FORMATBPB *)swizzledOffset = *(FORMATBPB *)deSwizzledOffset;
                    }
                }
            }
        }};

        switch (formatBpb) {
            case 1: {
                copyTexture.template operator()<u8>();
                break;
            }
            case 2: {
                copyTexture.template operator()<u16>();
                break;
            }
            case 4: {
                copyTexture.template operator()<u32>();
                break;
            }
            case 8: {
                copyTexture.template operator()<u64>();
                break;
            }
            case 12: {
                copyTexture.template operator()<u96>();
                break;
            }
            case 16: {
                copyTexture.template operator()<u128>();
                break;
            }
        }
    }

    /**
     * @brief Copies pixel data between a pitch and part of a blocklinear texture
     * @tparam BlockLinearToPitch Whether to copy from a part of a blocklinear texture to a pitch texture or a pitch texture to a part of a blocklinear texture
     * @note The function assumes that the pitch texture is always equal or smaller than the blocklinear texture
     */
    template<bool BlockLinearToPitch>
    void CopyBlockLinearSubrectInternal(Dimensions pitchDimensions, Dimensions blockLinearDimensions,
                                        size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                        size_t gobBlockHeight, size_t gobBlockDepth,
                                        u8 *blockLinear, u8 *pitch,
                                        u32 originX, u32 originY) {

        size_t pitchTextureWidth{util::DivideCeil<size_t>(pitchDimensions.width, formatBlockWidth)};
        size_t pitchTextureWidthBytes{pitchTextureWidth * formatBpb};
        size_t blockLinearTextureWidthAlignedBytes{util::AlignUp(util::DivideCeil<size_t>(blockLinearDimensions.width, formatBlockWidth) * formatBpb, GobWidth)};

        size_t xPerGOB{GobWidth / formatBpb};
        size_t actualOriginX{util::DivideCeil<size_t>(originX, formatBlockWidth)};
        size_t originXBytes{actualOriginX * formatBpb};

        size_t xPerFirstGOB{(GobWidth - (originXBytes & (GobWidth - 1))) / formatBpb};
        size_t xPerLastGOB{((originXBytes + pitchTextureWidthBytes) & (GobWidth - 1)) / formatBpb};

        // Only accounts for GOBs containing padding.
        size_t GOBsPerRob{pitchTextureWidthBytes < GobWidth ? 0 : ((originXBytes + pitchTextureWidthBytes) - util::AlignUp(originXBytes, GobWidth)) / GobWidth};

        size_t pitchTextureHeight{util::DivideCeil<size_t>(pitchDimensions.height, formatBlockHeight)};
        size_t robHeight{gobBlockHeight * GobHeight};

        size_t originYOffset{util::DivideCeil<size_t>(originY, formatBlockHeight)};

        size_t alignedDepth{util::AlignUp(blockLinearDimensions.depth, gobBlockDepth)};

        size_t pitchBytes{pitchAmount ? pitchAmount : pitchTextureWidthBytes};

        size_t startingBlockOffset{(originXBytes / GobWidth) * robHeight * GobWidth * alignedDepth};
        size_t blockSize{robHeight * GobWidth * alignedDepth};

        u8 *pitchOffset{pitch};

        for (size_t slice{}; slice < blockLinearDimensions.depth; ++slice, blockLinear += (GobHeight * GobWidth * gobBlockHeight)) {
            for (size_t line{}; line < pitchTextureHeight; ++line, pitchOffset += pitchBytes) {
                u64 robOffset{blockLinearTextureWidthAlignedBytes * util::AlignDown(originYOffset + line, robHeight) * alignedDepth};
                u64 blockHeight{((originYOffset + line) & (robHeight - 1)) / GobHeight};
                // Y Offset in entire GOBs
                u64 GobYOffset{blockHeight * GobWidth * GobHeight};
                // Y Offset inside 1 GOB
                GobYOffset += (((line & 0x07) >> 1) << 6) + ((line & 0x01) << 4);

                u8 *deSwizzledOffset{pitchOffset};
                u8 *swizzledOffset{blockLinear + robOffset + GobYOffset + startingBlockOffset};

                // Copy per every element, don't use for copies larger than GobWidth
                auto elemCopy{[&](size_t from, size_t to) {
                    for (size_t pixel{from}; pixel < to; ++pixel, deSwizzledOffset += formatBpb) {
                        size_t xBytes{(actualOriginX + pixel) * formatBpb};

                        // Set offset on X
                        size_t GobXOffset{(((xBytes & 0x3F) >> 5) << 8) + (xBytes & 0xF) + (((xBytes & 0x1F) >> 4) << 5)};

                        if constexpr (BlockLinearToPitch)
                            std::memcpy(deSwizzledOffset, swizzledOffset + GobXOffset, formatBpb);
                        else
                            std::memcpy(swizzledOffset + GobXOffset, deSwizzledOffset, formatBpb);
                    }
                    swizzledOffset += blockSize;
                }};

                if (xPerFirstGOB != xPerGOB)
                    elemCopy(0, xPerFirstGOB);

                // Copy per every GOB width
                for (size_t GOB{}; GOB < GOBsPerRob; ++GOB, deSwizzledOffset += GobWidth, swizzledOffset += blockSize) {
                    if constexpr (BlockLinearToPitch) {
                        std::memcpy(deSwizzledOffset, swizzledOffset, SectorWidth);
                        std::memcpy(deSwizzledOffset + 16, swizzledOffset + 0x20, SectorWidth);
                        std::memcpy(deSwizzledOffset + 32, swizzledOffset + 0x100, SectorWidth);
                        std::memcpy(deSwizzledOffset + 48, swizzledOffset + 0x120, SectorWidth);
                    } else {
                        std::memcpy(swizzledOffset, deSwizzledOffset, SectorWidth);
                        std::memcpy(swizzledOffset + 0x20, deSwizzledOffset + 16, SectorWidth);
                        std::memcpy(swizzledOffset + 0x100, deSwizzledOffset + 32, SectorWidth);
                        std::memcpy(swizzledOffset + 0x120, deSwizzledOffset + 48, SectorWidth);
                    }
                }

                if (xPerLastGOB != xPerGOB)
                    elemCopy(pitchTextureWidth - xPerLastGOB, pitchTextureWidth);
            }
        }
    }

    void CopyBlockLinearToLinear(Dimensions dimensions, size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, size_t gobBlockHeight, size_t gobBlockDepth, u8 *blockLinear, u8 *linear) {
        CopyBlockLinearInternal<true>(
            dimensions,
            formatBlockWidth, formatBlockHeight, formatBpb, 0,
            gobBlockHeight, gobBlockDepth,
            blockLinear, linear
        );
    }

    void CopyBlockLinearToPitch(Dimensions dimensions,
                                size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                size_t gobBlockHeight, size_t gobBlockDepth,
                                u8 *blockLinear, u8 *pitch) {
        CopyBlockLinearInternal<true>(
            dimensions,
            formatBlockWidth, formatBlockHeight, formatBpb, pitchAmount,
            gobBlockHeight, gobBlockDepth,
            blockLinear, pitch
        );
    }

    void CopyBlockLinearToPitchSubrect(Dimensions pitchDimensions, Dimensions blockLinearDimensions,
                                       size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                       size_t gobBlockHeight, size_t gobBlockDepth,
                                       u8 *blockLinear, u8 *pitch,
                                       u32 originX, u32 originY) {
        CopyBlockLinearSubrectInternal<true>(pitchDimensions, blockLinearDimensions,
                                             formatBlockWidth, formatBlockHeight, formatBpb, pitchAmount,
                                             gobBlockHeight, gobBlockDepth,
                                             blockLinear, pitch,
                                             originX, originY
        );
    }

    void CopyBlockLinearToLinear(const GuestTexture &guest, u8 *blockLinear, u8 *linear) {
        CopyBlockLinearInternal<true>(
            guest.dimensions,
            guest.format->blockWidth, guest.format->blockHeight, guest.format->bpb, 0,
            guest.tileConfig.blockHeight, guest.tileConfig.blockDepth,
            blockLinear, linear
        );
    }

    void CopyLinearToBlockLinear(Dimensions dimensions,
                                 size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                 size_t gobBlockHeight, size_t gobBlockDepth,
                                 u8 *linear, u8 *blockLinear) {
        CopyBlockLinearInternal<false>(dimensions,
                                       formatBlockWidth, formatBlockHeight, formatBpb, 0,
                                       gobBlockHeight, gobBlockDepth,
                                       blockLinear, linear
        );
    }

    void CopyPitchToBlockLinear(Dimensions dimensions, size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount, size_t gobBlockHeight, size_t gobBlockDepth, u8 *pitch, u8 *blockLinear) {
        CopyBlockLinearInternal<false>(
            dimensions,
            formatBlockWidth, formatBlockHeight, formatBpb, pitchAmount,
            gobBlockHeight, gobBlockDepth,
            blockLinear, pitch
        );
    }

    void CopyLinearToBlockLinearSubrect(Dimensions linearDimensions, Dimensions blockLinearDimensions,
                                       size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                       size_t gobBlockHeight, size_t gobBlockDepth,
                                       u8 *linear, u8 *blockLinear,
                                       u32 originX, u32 originY) {
        CopyBlockLinearSubrectInternal<false>(linearDimensions, blockLinearDimensions,
                                              formatBlockWidth, formatBlockHeight,
                                              formatBpb, 0,
                                              gobBlockHeight, gobBlockDepth,
                                              blockLinear, linear,
                                              originX, originY
        );
    }

    void CopyPitchToBlockLinearSubrect(Dimensions pitchDimensions, Dimensions blockLinearDimensions,
                                       size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                       size_t gobBlockHeight, size_t gobBlockDepth,
                                       u8 *pitch, u8 *blockLinear,
                                       u32 originX, u32 originY) {
        CopyBlockLinearSubrectInternal<false>(pitchDimensions, blockLinearDimensions,
                                              formatBlockWidth, formatBlockHeight,
                                              formatBpb, pitchAmount,
                                              gobBlockHeight, gobBlockDepth,
                                              blockLinear, pitch,
                                              originX, originY
        );
    }

    void CopyLinearToBlockLinear(const GuestTexture &guest, u8 *linear, u8 *blockLinear) {
        CopyBlockLinearInternal<false>(
            guest.dimensions,
            guest.format->blockWidth, guest.format->blockHeight, guest.format->bpb, 0,
            guest.tileConfig.blockHeight, guest.tileConfig.blockDepth,
            blockLinear, linear
        );
    }

    void CopyPitchLinearToLinear(const GuestTexture &guest, u8 *guestInput, u8 *linearOutput) {
        auto sizeLine{guest.format->GetSize(guest.dimensions.width, 1)}; //!< The size of a single line of pixel data
        auto sizeStride{guest.tileConfig.pitch}; //!< The size of a single stride of pixel data

        auto inputLine{guestInput};
        auto outputLine{linearOutput};

        for (size_t line{}; line < guest.dimensions.height; line++) {
            std::memcpy(outputLine, inputLine, sizeLine);
            inputLine += sizeStride;
            outputLine += sizeLine;
        }
    }

    void CopyLinearToPitchLinear(const GuestTexture &guest, u8 *linearInput, u8 *guestOutput) {
        auto sizeLine{guest.format->GetSize(guest.dimensions.width, 1)}; //!< The size of a single line of pixel data
        auto sizeStride{guest.tileConfig.pitch}; //!< The size of a single stride of pixel data

        auto inputLine{linearInput};
        auto outputLine{guestOutput};

        for (size_t line{}; line < guest.dimensions.height; line++) {
            std::memcpy(outputLine, inputLine, sizeLine);
            inputLine += sizeLine;
            outputLine += sizeStride;
        }
    }
}
