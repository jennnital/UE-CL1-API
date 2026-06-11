#pragma once

#include "CoreMinimal.h"
#include "AssembloidAgencyTypes.generated.h"

/**
 * A batch of spikes reported by the CL1 (or the bridge simulator) in a single
 * UDP packet. Matches the Cortical Labs wire format: a 64-bit sample-index
 * timestamp plus the list of channels that fired.
 */
USTRUCT(BlueprintType)
struct FCLSpikeEvent
{
	GENERATED_BODY()

	/** CL sample index of the packet (40 µs per frame; 25 kHz sampling). */
	UPROPERTY(BlueprintReadOnly, Category = "Assembloid Agency")
	int64 Timestamp = 0;

	/** Channels (electrodes) that spiked. Default MEA range is 0–59. */
	UPROPERTY(BlueprintReadOnly, Category = "Assembloid Agency")
	TArray<int32> Channels;

	/** Source endpoint (e.g. "192.168.1.40"). */
	UPROPERTY(BlueprintReadOnly, Category = "Assembloid Agency")
	FString SourceIP;
};

/** Fired on the game thread for every spike packet received. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnSpikesReceived, const FCLSpikeEvent&, SpikeEvent);

/** Fired once the spike receive socket is listening. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnBridgeConnected, int32, ListenPort);
