// Copyright. GameInstance subsystem implementing the Assembloid Agency UE<->CL1 bridge.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Cl1SpikeTypes.h"
#include "Cl1BridgeSubsystem.generated.h"

class FSocket;
class FUdpSocketReceiver;

/**
 * UCl1BridgeSubsystem
 *
 * Substrate-agnostic UDP bridge (Assembloid Agency, Sec 3): the same wire
 * format talks to a CL1 (via bridge.py + CL-API) OR a NEST/SNN/EEG stand-in,
 * because both expose the spike-firehose + stim-command convention. UE just
 * sees UDP.
 *
 * Inbound  (substrate -> UE): uint64 LE frame-index timestamp + uint8 channels.
 * Outbound (UE -> substrate): the Assembloid 'AA' control packet (PROTOCOL.md),
 *   header '<2sBBBBHfHH'. Stim params map onto the CL paper's Sec 6 contract
 *   (StimDesign / BurstDesign / interrupt_then_stim) on the bridge side.
 *
 * Exposes the Assembloid Sec 3.3 API surface (C++ and Blueprint):
 *   SendStimulus, SendRewardSignal, InterruptStim, GetSpikeResponse,
 *   GetChannelRates, RecordSessionData, ExportToCSV.
 */
UCLASS()
class UECL1API_API UCl1BridgeSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// --- Connection --------------------------------------------------------

	/** Start listening for the spike firehose on the given UDP port. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Connection")
	bool StartReceiver(int32 ListenPort = 12345, const FString& BindAddress = TEXT("0.0.0.0"));

	/** Stop the spike receiver and release the socket. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Connection")
	void StopReceiver();

	/** Configure where control packets are sent (bridge.py control port). */
	UFUNCTION(BlueprintCallable, Category = "CL1|Connection")
	bool ConfigureControlTarget(const FString& Host, int32 Port);

	UFUNCTION(BlueprintCallable, Category = "CL1|Connection")
	bool IsReceiving() const { return Receiver != nullptr; }

	// --- Stimulation (Assembloid Sec 3.3 / CL paper Sec 6) -----------------

	/**
	 * Deliver targeted stimulation. Mirrors Assembloid SendStimulus():
	 * channels, firing frequency (Hz), pulse width (us), amplitude (uA),
	 * duration (ms). Duration is converted to a pulse count:
	 *   NumPulses = max(1, round(DurationMs/1000 * FreqHz)).
	 * Set bInterruptFirst to cleanly REPLACE ongoing activity (CL Sec 6.1.1)
	 * instead of appending to the per-channel queue - use this for continuous
	 * proximity->frequency control so latency does not build up.
	 */
	UFUNCTION(BlueprintCallable, Category = "CL1|Stim")
	bool SendStimulus(const TArray<int32>& Channels, float FreqHz, int32 PulseWidthUs,
	                  float AmplitudeUa, float DurationMs, bool bInterruptFirst = false);

	/**
	 * Reinforcement signal (Assembloid SendRewardSignal). In the DishBrain
	 * paradigm reward IS stimulation, so this is a thin, experiment-defined
	 * wrapper over SendStimulus. By convention here: positive => a predictable
	 * burst on RewardChannels; negative => an interrupt (silence). Override the
	 * mapping to match your protocol.
	 */
	UFUNCTION(BlueprintCallable, Category = "CL1|Stim")
	bool SendRewardSignal(bool bPositive, const TArray<int32>& RewardChannels,
	                      float FreqHz = 100.f, int32 PulseWidthUs = 200,
	                      float AmplitudeUa = 2.0f, float DurationMs = 50.f);

	/** Cleanly stop ongoing stimulation on channels (CL Sec 6.1.1 interrupt). */
	UFUNCTION(BlueprintCallable, Category = "CL1|Stim")
	bool InterruptStim(const TArray<int32>& Channels);

	// --- Recording (CL paper Sec 6.4) -------------------------------------

	/**
	 * Toggle CL1-side HDF5 recording through the bridge (RECORD message).
	 * This is the authoritative record: it captures raw samples, spikes, stims
	 * and data streams with one time base (Sec 6.4) - richer than any UE log.
	 * If bAlsoLogInUE is true, additionally writes a lightweight UE-side CSV of
	 * received spikes for quick game-side debugging (NOT the system of record).
	 */
	UFUNCTION(BlueprintCallable, Category = "CL1|Record")
	bool RecordSessionData(bool bStart, bool bAlsoLogInUE = false);

	/** Export the UE-side spike log to CSV. Returns the written path. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Record")
	bool ExportToCSV(const FString& FilePath);

	// --- Readback / visualization (Assembloid Sec 3.3 / 3.4) ---------------

	/** Most recent spike seen on a channel (frame index); -1 if none. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Readback")
	int64 GetSpikeResponse(int32 Channel) const;

	/**
	 * Per-channel firing rate (Hz) over the last WindowSeconds - the primitive
	 * behind Assembloid's spike-to-axis movement mappings (Sec 3.4).
	 */
	UFUNCTION(BlueprintCallable, Category = "CL1|Readback")
	float GetSpikeRateHz(int32 Channel, float WindowSeconds = 1.0f) const;

	/** Snapshot of all channels' rates for a visualizer (VisualizeSpikes). */
	UFUNCTION(BlueprintCallable, Category = "CL1|Readback")
	TArray<FCl1ChannelRate> GetChannelRates(float WindowSeconds = 1.0f) const;

	// --- Events ------------------------------------------------------------

	UPROPERTY(BlueprintAssignable, Category = "CL1")
	FCl1OnSpike OnSpike;

	UPROPERTY(BlueprintAssignable, Category = "CL1")
	FCl1OnSpikeBatch OnSpikeBatch;

private:
	void HandleReceivedData(const FArrayReaderPtr& Data, const FIPv4Endpoint& Endpoint);

	/** Build + send an 'AA' control packet. Returns false if no target set. */
	bool SendControlPacket(uint8 MsgType, uint8 Flags, const TArray<int32>& Channels,
	                       uint16 PulseWidthUs, float AmplitudeUa,
	                       uint16 NumPulses, uint16 FreqHz);

	/** Record a spike into the rolling buffer (game thread). */
	void TrackSpike(const FCl1Spike& Spike);

	FSocket* ListenSocket = nullptr;
	FUdpSocketReceiver* Receiver = nullptr;

	FSocket* ControlSocket = nullptr;
	TSharedPtr<class FInternetAddr> ControlAddr;

	// Rolling spike history for rate queries (game thread only).
	static constexpr int32 HistoryCapacity = 8192;
	TArray<FCl1Spike> SpikeHistory;   // ring buffer
	int32 HistoryHead = 0;
	int64 LatestFrame = 0;

	// Lightweight UE-side CSV log (debug convenience, not system of record).
	bool bUeLogging = false;
	TArray<FCl1Spike> UeLog;
};
