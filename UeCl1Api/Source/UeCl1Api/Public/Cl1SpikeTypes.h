// Copyright. Types shared between the Assembloid bridge subsystem and game/Blueprint code.

#pragma once

#include "CoreMinimal.h"
#include "Cl1SpikeTypes.generated.h"

/** CL1 frame clock: 25 kHz => 40 us per frame (CL paper Sec 6.1.5 / 6.5). */
namespace Cl1Time
{
	constexpr double FramesPerSecond = 25000.0;
	FORCEINLINE double FramesToSeconds(int64 Frames) { return double(Frames) / FramesPerSecond; }
	FORCEINLINE double FramesToMilliseconds(int64 Frames) { return (double(Frames) / FramesPerSecond) * 1000.0; }
}

/**
 * A single detected action potential, mirroring the CL API `Spike` object as
 * delivered over UDP. NOTE on units: the wire timestamp is a hardware FRAME
 * INDEX (40 us/frame), NOT milliseconds. The Assembloid paper's prose calls
 * spike timestamps "ms"; that is a convenience, not the wire value. Use
 * TimeSeconds / TimeMilliseconds for real time. The firehose carries no
 * voltage; the uV waveform lives in the CL API's Spike.samples, not here.
 */
USTRUCT(BlueprintType)
struct FCl1Spike
{
	GENERATED_BODY()

	/** Hardware frame-index timestamp (40 us/frame). */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	int64 Timestamp = 0;

	/** Electrode/channel index (0..63 on a 64-channel CL1). */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	int32 Channel = 0;

	/** Timestamp converted to seconds since the session clock origin. */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	double TimeSeconds = 0.0;
};

/** Per-channel firing rate snapshot for spike-to-axis mappings / visualization. */
USTRUCT(BlueprintType)
struct FCl1ChannelRate
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	int32 Channel = 0;

	/** Spikes per second over the query window. */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	float RateHz = 0.f;
};

/** Fired once per spike, on the game thread. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCl1OnSpike, FCl1Spike, Spike);

/** Fired once per received packet with every spike in it, on the game thread. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCl1OnSpikeBatch, const TArray<FCl1Spike>&, Spikes);
