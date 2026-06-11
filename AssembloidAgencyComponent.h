#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AssembloidAgencyTypes.h"
#include "AssembloidAgencyComponent.generated.h"

class FUDPNative;

/**
 * UAssembloidAgencyComponent
 * --------------------------
 * Drop-in closed-loop interface between Unreal Engine and Cortical Labs' CL1
 * (or the NEST simulator / the bundled Python bridge in --simulate mode).
 *
 *   - Receives spike packets from the CL1 on a UDP port and surfaces them as
 *     FCLSpikeEvent via the OnSpikesReceived delegate, plus rolling spike-rate
 *     queries per channel.
 *   - Sends electrical stimulation to the CL1 with SendStimulus() / SendSinglePulse(),
 *     serialised as Assembloid Agency STIM packets (see PROTOCOL.md).
 *
 * Embeds getnamo's FUDPNative directly. Receive callbacks are marshalled to the
 * game thread by the wrapper, so it is safe to touch game state in the handler.
 *
 * Add this component to any Actor (e.g. a "NeuronBridge" actor in your level).
 */
UCLASS(ClassGroup = "Networking", meta = (BlueprintSpawnableComponent))
class ASSEMBLOIDAGENCY_API UAssembloidAgencyComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAssembloidAgencyComponent();

	// ---- Connection settings ------------------------------------------------

	/** IP of the CL1 / machine running the bridge (stim destination). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Connection")
	FString CLDeviceIP = TEXT("127.0.0.1");

	/** Port on the CL1/bridge that listens for our stim commands. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Connection")
	int32 StimSendPort = 12346;

	/** Local IP to bind for incoming spikes (0.0.0.0 = all interfaces). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Connection")
	FString SpikeListenIP = TEXT("0.0.0.0");

	/** Local port to receive spikes on (must match the bridge's --unreal-port). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Connection")
	int32 SpikeListenPort = 12345;

	/** Open sockets automatically on BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Connection")
	bool bAutoConnectOnBeginPlay = true;

	/** Window (seconds) used by GetSpikeRateHz(). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Assembloid Agency|Analysis")
	float SpikeRateWindowSeconds = 1.0f;

	// ---- Events -------------------------------------------------------------

	UPROPERTY(BlueprintAssignable, Category = "Assembloid Agency|Events")
	FOnSpikesReceived OnSpikesReceived;

	UPROPERTY(BlueprintAssignable, Category = "Assembloid Agency|Events")
	FOnBridgeConnected OnBridgeConnected;

	// ---- Connection control -------------------------------------------------

	/** Open the receive (spike) and send (stim) sockets. */
	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Connection")
	bool Connect();

	/** Close both sockets. */
	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Connection")
	void Disconnect();

	UFUNCTION(BlueprintPure, Category = "Assembloid Agency|Connection")
	bool IsConnected() const { return bConnected; }

	// ---- Stimulation --------------------------------------------------------

	/**
	 * Send a (possibly bursting) stimulation to the given channels.
	 * Charge-balanced biphasic by default (cathodic then anodic phase).
	 *
	 * @param Channels       Electrodes to stimulate (0–59 on the default MEA).
	 * @param FrequencyHz    Burst rate; ignored when only one pulse is produced.
	 * @param PulseWidthUs   Per-phase pulse width in microseconds.
	 * @param AmplitudeUA    Current magnitude in microamps.
	 * @param DurationMs     Burst duration; pulse count = round(DurationMs/1000 * FrequencyHz).
	 * @return true if the packet was emitted.
	 */
	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Stimulation")
	bool SendStimulus(const TArray<int32>& Channels, int32 FrequencyHz = 50,
	                  int32 PulseWidthUs = 180, float AmplitudeUA = 1.5f,
	                  int32 DurationMs = 0);

	/** Convenience: a single charge-balanced biphasic pulse. */
	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Stimulation")
	bool SendSinglePulse(const TArray<int32>& Channels,
	                     int32 PulseWidthUs = 180, float AmplitudeUA = 1.5f);

	/** Lowest-level form: explicit pulse count. */
	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Stimulation")
	bool SendStimulusPulses(const TArray<int32>& Channels, int32 NumPulses,
	                        int32 FrequencyHz, int32 PulseWidthUs,
	                        float AmplitudeUA, bool bBiphasic = true);

	// ---- Spike analysis -----------------------------------------------------

	/** Spikes/second on a channel over SpikeRateWindowSeconds. */
	UFUNCTION(BlueprintPure, Category = "Assembloid Agency|Analysis")
	float GetSpikeRateHz(int32 Channel) const;

	/** Total spikes counted on a channel since the last reset. */
	UFUNCTION(BlueprintPure, Category = "Assembloid Agency|Analysis")
	int32 GetTotalSpikeCount(int32 Channel) const;

	/** Most recent spike event received. */
	UFUNCTION(BlueprintPure, Category = "Assembloid Agency|Analysis")
	const FCLSpikeEvent& GetLastSpikeEvent() const { return LastSpikeEvent; }

	UFUNCTION(BlueprintCallable, Category = "Assembloid Agency|Analysis")
	void ResetSpikeCounts();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	void HandleReceivedBytes(const TArray<uint8>& Data, const FString& Endpoint, const int32& Port);
	void PruneOldSpikeTimes(double Now);

	TSharedPtr<FUDPNative> UDP;
	bool bConnected = false;

	UPROPERTY()
	FCLSpikeEvent LastSpikeEvent;

	// Rolling app-time of recent spikes per channel, for rate calculation.
	TMap<int32, TArray<double>> RecentSpikeTimes;
	TMap<int32, int32> TotalSpikeCounts;
};
