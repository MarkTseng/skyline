// SPDX-License-Identifier: MPL-2.0
// Copyright © 2022 Skyline Team and Contributors (https://github.com/skyline-emu/)
// Copyright © 2022 yuzu Emulator Project (https://github.com/yuzu-emu/yuzu/)

#include <gpu/interconnect/command_executor.h>
#include <gpu/texture/format.h>
#include <gpu/texture/layout.h>
#include <soc.h>
#include <soc/gm20b/channel.h>
#include <soc/gm20b/gmmu.h>
#include "maxwell_dma.h"

namespace skyline::soc::gm20b::engine {
    MaxwellDma::MaxwellDma(const DeviceState &state, ChannelContext &channelCtx)
        : channelCtx{channelCtx},
          syncpoints{state.soc->host1x.syncpoints},
          interconnect{*state.gpu, channelCtx} {}

    __attribute__((always_inline)) void MaxwellDma::CallMethod(u32 method, u32 argument) {
        Logger::Verbose("Called method in Maxwell DMA: 0x{:X} args: 0x{:X}", method, argument);

        HandleMethod(method, argument);
    }

    void MaxwellDma::HandleMethod(u32 method, u32 argument) {
        registers.raw[method] = argument;

        if (method == ENGINE_OFFSET(launchDma))
            LaunchDma();
    }

    void MaxwellDma::LaunchDma() {
        if (registers.launchDma->remapEnable) [[unlikely]]
            Logger::Warn("Remapped DMA copies are unimplemented!");
        else
            DmaCopy();

        ReleaseSemaphore();
    }

    void MaxwellDma::DmaCopy() {
        if (registers.launchDma->multiLineEnable) {
            channelCtx.executor.Submit();

            if (registers.launchDma->srcMemoryLayout == registers.launchDma->dstMemoryLayout) [[unlikely]] {
                // Pitch to Pitch copy
                if (registers.launchDma->srcMemoryLayout == Registers::LaunchDma::MemoryLayout::Pitch) {
                    if ((*registers.pitchIn == *registers.pitchOut) && (*registers.pitchIn == *registers.lineLengthIn)) {
                        // Both Linear, copy as is.
                        interconnect.Copy(u64{*registers.offsetOut}, u64{*registers.offsetIn}, u64{*registers.lineLengthIn * *registers.lineCount});
                    } else {
                        u32 srcCopyOffset{0};
                        u32 dstCopyOffset{0};
                        for (u32 linesToCopy = *registers.lineCount; linesToCopy; --linesToCopy, srcCopyOffset += *registers.pitchIn, dstCopyOffset += *registers.pitchOut)
                            interconnect.Copy(u64{*registers.offsetOut + dstCopyOffset} , u64{*registers.offsetIn + srcCopyOffset}, u64{*registers.lineLengthIn});
                    }
                } else {
                    Logger::Warn("BlockLinear to BlockLinear DMA copies are unimplemented!");
                }

                return;
            } else if (registers.launchDma->srcMemoryLayout == Registers::LaunchDma::MemoryLayout::BlockLinear) {
                CopyBlockLinearToPitch();
            } else [[likely]] {
                CopyPitchToBlockLinear();
            }
        } else {
            // 1D copy
            interconnect.Copy(u64{*registers.offsetOut}, u64{*registers.offsetIn}, u64{*registers.lineLengthIn});
        }
    }

    void MaxwellDma::CopyBlockLinearToPitch() {
        if (registers.srcSurface->blockSize.Width() != 1) [[unlikely]] {
            Logger::Error("Blocklinear surfaces with a non-one block width are unsupported on the Tegra X1: {}", registers.srcSurface->blockSize.Width());
            return;
        }

        gpu::texture::Dimensions srcDimensions{registers.srcSurface->width, registers.srcSurface->height, registers.srcSurface->depth};
        size_t srcLayerStride{gpu::texture::GetBlockLinearLayerSize(srcDimensions, 1, 1, 1, registers.srcSurface->blockSize.Height(), registers.srcSurface->blockSize.Depth())};
        size_t srcLayerAddress{*registers.offsetIn + (registers.srcSurface->layer * srcLayerStride)};

        // Get source address
        auto srcMappings{channelCtx.asCtx->gmmu.TranslateRange(*registers.offsetIn, srcLayerStride)};

        gpu::texture::Dimensions dstDimensions{*registers.lineLengthIn, *registers.lineCount, registers.srcSurface->depth};
        u32 dstStride{*registers.pitchIn * dstDimensions.height * dstDimensions.depth};

        // Get destination address
        auto dstMappings{channelCtx.asCtx->gmmu.TranslateRange(*registers.offsetOut, dstStride)};

        if (srcMappings.size() != 1 || dstMappings.size() != 1) [[unlikely]] {
            Logger::Warn("DMA copies for split textures are unimplemented!");
            return;
        }

        Logger::Debug("{}x{}@0x{:X} -> {}x{}@0x{:X}", srcDimensions.width, srcDimensions.height, srcLayerAddress, dstDimensions.width, dstDimensions.height, u64{*registers.offsetOut});

        if ((srcDimensions.width != dstDimensions.width) || (srcDimensions.height != dstDimensions.height)) {
            gpu::texture::CopyBlockLinearToPitchSubrect(
                    srcDimensions, dstDimensions,
                    1, 1, 1, *registers.pitchIn,
                    registers.dstSurface->blockSize.Height(), registers.dstSurface->blockSize.Depth(),
                    srcMappings.front().data(), dstMappings.front().data(),
                    registers.dstSurface->origin.x, registers.dstSurface->origin.y
            );
        } else {
            gpu::texture::CopyBlockLinearToPitch(
                    dstDimensions,
                    1, 1, 1, *registers.pitchIn,
                    registers.dstSurface->blockSize.Height(), registers.dstSurface->blockSize.Depth(),
                    srcMappings.front().data(), dstMappings.front().data()
            );
        }
    }

    void MaxwellDma::CopyPitchToBlockLinear() {
        if (registers.dstSurface->blockSize.Width() != 1) [[unlikely]] {
            Logger::Error("Blocklinear surfaces with a non-one block width are unsupported on the Tegra X1: {}", registers.srcSurface->blockSize.Width());
            return;
        }

        gpu::texture::Dimensions srcDimensions{*registers.lineLengthIn, *registers.lineCount, registers.dstSurface->depth};
        u32 srcSize{*registers.pitchIn * srcDimensions.height * srcDimensions.depth};

        // Get source address
        auto srcMappings{channelCtx.asCtx->gmmu.TranslateRange(*registers.offsetIn, srcSize)};

        gpu::texture::Dimensions dstDimensions{registers.dstSurface->width, registers.dstSurface->height, registers.dstSurface->depth};
        size_t dstLayerStride{gpu::texture::GetBlockLinearLayerSize(dstDimensions, 1, 1, 1, registers.dstSurface->blockSize.Height(), registers.dstSurface->blockSize.Depth())};
        size_t dstLayerAddress{*registers.offsetOut + (registers.dstSurface->layer * dstLayerStride)};

        // Get destination address
        auto dstMappings{channelCtx.asCtx->gmmu.TranslateRange(*registers.offsetOut, dstLayerStride)};

        if (srcMappings.size() != 1 || dstMappings.size() != 1) [[unlikely]] {
            Logger::Warn("DMA copies for split textures are unimplemented!");
            return;
        }

        Logger::Debug("{}x{}@0x{:X} -> {}x{}@0x{:X}", srcDimensions.width, srcDimensions.height, u64{*registers.offsetIn}, dstDimensions.width, dstDimensions.height, dstLayerAddress);

        if ((srcDimensions.width != dstDimensions.width) || (srcDimensions.height != dstDimensions.height)) {
            gpu::texture::CopyPitchToBlockLinearSubrect(
                    srcDimensions, dstDimensions,
                    1, 1, 1, *registers.pitchIn,
                    registers.dstSurface->blockSize.Height(), registers.dstSurface->blockSize.Depth(),
                    srcMappings.front().data(), dstMappings.front().data(),
                    registers.dstSurface->origin.x, registers.dstSurface->origin.y
            );
        } else {
            gpu::texture::CopyPitchToBlockLinear(
                    dstDimensions,
                    1, 1, 1, *registers.pitchIn,
                    registers.dstSurface->blockSize.Height(), registers.dstSurface->blockSize.Depth(),
                    srcMappings.front().data(), dstMappings.front().data()
            );
        }
    }

    void MaxwellDma::ReleaseSemaphore() {
        if (registers.launchDma->reductionEnable) [[unlikely]]
            Logger::Warn("Semaphore reduction is unimplemented!");

        u64 address{registers.semaphore->address};
        u64 payload{registers.semaphore->payload};
        switch (registers.launchDma->semaphoreType) {
            case Registers::LaunchDma::SemaphoreType::ReleaseOneWordSemaphore:
                channelCtx.asCtx->gmmu.Write(address, payload);
                Logger::Debug("address: 0x{:X} payload: {}", address, payload);
                break;
            case Registers::LaunchDma::SemaphoreType::ReleaseFourWordSemaphore: {
                // Write timestamp first to ensure correct ordering
                u64 timestamp{GetGpuTimeTicks()};
                channelCtx.asCtx->gmmu.Write(address + 8, timestamp);
                channelCtx.asCtx->gmmu.Write(address, payload);
                Logger::Debug("address: 0x{:X} payload: {} timestamp: {}", address, payload, timestamp);
                break;
            }
            default:
                break;
        }
    }

    void MaxwellDma::CallMethodBatchNonInc(u32 method, span<u32> arguments) {
        for (u32 argument : arguments)
            HandleMethod(method, argument);
    }
}
