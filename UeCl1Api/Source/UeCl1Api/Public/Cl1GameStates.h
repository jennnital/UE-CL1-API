// Copyright. Game-state codec component: game state <-> CL1 stimulation/spikes.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Cl1GameStates.generated.h"

class UCl1BridgeSubsystem;

/**
 * Four discrete game states used to drive a pawn. The index (0..3) is the whole
 * contract: the ENCODER stimulates ClassElectrodes[index], and the DECODER maps
 * MEA band `index` back to this state. Rename the DisplayNames to suit your game
 * (e.g. Forward/Back/Left/Right) - only the ordering matters.
 */
UENUM(BlueprintType)
enum class ECl1GameState : uint8
{
	Up    = 0 UMETA(DisplayName = "Up"),
	Down  = 1 UMETA(DisplayName = "Down"),
	Left  = 2 UMETA(DisplayName = "Left"),
	Right = 3 UMETA(DisplayName = "Right")
};

/** Result of decoding recent spiking activity into one of the four game states. */
USTRUCT(BlueprintType)
struct FCl1DecodeResult
{
	GENERATED_BODY()

	/** Winning state = argmax over the four band scores. */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	ECl1GameState State = ECl1GameState::Up;

	/** Softmax probability of the winning band (0..1); a simple certainty gate. */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	float Confidence = 0.f;

	/** Per-band summed firing rate (Hz), one entry per state, in state order. */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	TArray<float> BandScores;

	/** False when the array was silent over the window (nothing to decode). */
	UPROPERTY(BlueprintReadOnly, Category = "CL1")
	bool bHasSignal = false;
};

/** Fired when auto-decode produces a result that passes the confidence gate. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCl1OnStateDecoded, FCl1DecodeResult, Result);

/**
 * UCl1GameStates
 *
 * A Blueprint-spawnable component that ports the biocompute closed loop
 * (cl1-callosum-biocompute: models.encoders.SymbolToFixedChannelEncoder +
 * models.decoders.LinearArgmaxDecoder, wired live in classify_server.py) onto
 * the UE CL1 bridge.
 *
 *   ENCODE  game state -> stimulation:
 *     each state is pinned to one fixed electrode (ClassElectrodes), stimulated
 *     with a fixed biphasic burst - "identity in signal, diversity in channel".
 *
 *   DECODE  spikes -> game state:
 *     the 64 channels pool into 4 spatial column-bands (band = channel*4/64 =
 *     channel/16 on an 8x8 MEA); the band with the highest summed firing rate
 *     wins (argmax). This is the weightless LinearArgmaxDecoder read-out.
 *
 * Drop this on a pawn, then EncodeState() to stimulate, and bind OnStateDecoded
 * (or poll DecodeState) to move the pawn. It finds the UCl1BridgeSubsystem on
 * the owning game instance automatically.
 */
UCLASS(ClassGroup = (CL1), meta = (BlueprintSpawnableComponent, DisplayName = "CL1 Game States"))
class UECL1API_API UCl1GameStates : public UActorComponent
{
	GENERATED_BODY()

public:
	UCl1GameStates();

	// --- Encoder config (game state -> stimulation) ------------------------

	/**
	 * One stimulation electrode per state, indexed by ECl1GameState. Defaults are
	 * the biocompute CLASS_ELECTRODES = {10, 26, 42, 50}: one live electrode in
	 * each of the four column-bands, so encode and decode share a band layout.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	TArray<int32> ClassElectrodes;

	/** Burst frequency in Hz (biocompute Signal.burst_rate_hz). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	float StimFreqHz = 200.f;

	/** Biphasic pulse width in microseconds (Signal.pulse_width_us). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	int32 PulseWidthUs = 400;

	/** Stimulation amplitude in microamps (Signal.amplitude). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	float AmplitudeUa = 2.8f;

	/** Burst duration in ms; NumPulses = round(s * Hz). 50 ms @ 200 Hz = 10 pulses. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	float StimDurationMs = 50.f;

	/** Replace ongoing stim rather than queueing it, so control latency can't build up. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Encode")
	bool bInterruptFirst = true;

	// --- Decoder config (spikes -> game state) -----------------------------

	/** Total electrodes on the array; band width = TotalChannels / 4. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Decode")
	int32 TotalChannels = 64;

	/** Rolling window (seconds) over which per-band firing rates are summed. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Decode")
	float DecodeWindowSeconds = 0.5f;

	/** If true, decode every DecodeIntervalSeconds and broadcast OnStateDecoded. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Decode")
	bool bAutoDecode = false;

	/** Auto-decode cadence in seconds. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Decode")
	float DecodeIntervalSeconds = 0.25f;

	/** Auto-decode only broadcasts when Confidence >= this (0 = always). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CL1|Decode")
	float MinConfidence = 0.f;

	// --- API ---------------------------------------------------------------

	/** ENCODE: stimulate the electrode pinned to State. False if no bridge/electrode. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Encode")
	bool EncodeState(ECl1GameState State);

	/** DECODE: pool recent spikes into 4 bands and return the argmax state. */
	UFUNCTION(BlueprintCallable, Category = "CL1|Decode")
	bool DecodeState(FCl1DecodeResult& Result);

	/** The electrode this component would stimulate for State (-1 if unset). */
	UFUNCTION(BlueprintPure, Category = "CL1|Encode")
	int32 StateToElectrode(ECl1GameState State) const;

	/** MEA band (0..3) a channel pools into: channel * 4 / TotalChannels. */
	UFUNCTION(BlueprintPure, Category = "CL1|Decode")
	int32 BandOfChannel(int32 Channel) const;

	/** Fired by auto-decode when a result passes the confidence gate. */
	UPROPERTY(BlueprintAssignable, Category = "CL1|Decode")
	FCl1OnStateDecoded OnStateDecoded;

protected:
	virtual void BeginPlay() override;

public:
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
	                           FActorComponentTickFunction* ThisTickFunction) override;

private:
	/** Resolve + cache the bridge subsystem from the owning game instance. */
	UCl1BridgeSubsystem* ResolveBridge();

	TWeakObjectPtr<UCl1BridgeSubsystem> CachedBridge;
	float TimeSinceDecode = 0.f;
};
