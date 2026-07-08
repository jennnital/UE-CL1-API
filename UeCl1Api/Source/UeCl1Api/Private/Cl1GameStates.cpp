// Copyright. Implementation of the game-state <-> stimulation/spike codec.

#include "Cl1GameStates.h"

#include "Cl1BridgeSubsystem.h"
#include "Cl1SpikeTypes.h"
#include "Engine/World.h"
#include "Engine/GameInstance.h"

UCl1GameStates::UCl1GameStates()
{
	PrimaryComponentTick.bCanEverTick = true;

	// biocompute CLASS_ELECTRODES: one live electrode per column-band (0..3).
	ClassElectrodes = { 10, 26, 42, 50 };
}

void UCl1GameStates::BeginPlay()
{
	Super::BeginPlay();
	ResolveBridge();
}

UCl1BridgeSubsystem* UCl1GameStates::ResolveBridge()
{
	if (CachedBridge.IsValid())
	{
		return CachedBridge.Get();
	}
	if (const UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			CachedBridge = GI->GetSubsystem<UCl1BridgeSubsystem>();
		}
	}
	return CachedBridge.Get();
}

int32 UCl1GameStates::StateToElectrode(ECl1GameState State) const
{
	const int32 Index = static_cast<int32>(State);
	return ClassElectrodes.IsValidIndex(Index) ? ClassElectrodes[Index] : -1;
}

int32 UCl1GameStates::BandOfChannel(int32 Channel) const
{
	// channel * 4 / TotalChannels -> {0,1,2,3}. On a 64-ch MEA this is channel/16,
	// matching biocompute band_of(channel) = channel // 16.
	const int32 Denom = FMath::Max(TotalChannels, 1);
	return FMath::Clamp((Channel * 4) / Denom, 0, 3);
}

// --------------------------------------------------------------------------- //
// ENCODE: game state -> stimulation
// --------------------------------------------------------------------------- //
bool UCl1GameStates::EncodeState(ECl1GameState State)
{
	UCl1BridgeSubsystem* Bridge = ResolveBridge();
	if (!Bridge)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] EncodeState: no bridge subsystem on this game instance"));
		return false;
	}

	const int32 Electrode = StateToElectrode(State);
	if (Electrode < 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] EncodeState: no electrode mapped for state %d"),
			static_cast<int32>(State));
		return false;
	}

	const TArray<int32> Channels = { Electrode };
	FString OutChannels, OutFreqHz, OutPulseWidth, OutAmplitude, OutDuration;
	return Bridge->SendStimulation(Channels, StimFreqHz, PulseWidthUs, AmplitudeUa,
		StimDurationMs, bInterruptFirst, OutChannels, OutFreqHz, OutPulseWidth,
		OutAmplitude, OutDuration);
}

// --------------------------------------------------------------------------- //
// DECODE: spikes -> game state (weightless band-pool argmax)
// --------------------------------------------------------------------------- //
bool UCl1GameStates::DecodeState(FCl1DecodeResult& Result)
{
	Result = FCl1DecodeResult();
	Result.BandScores.Init(0.f, 4);

	UCl1BridgeSubsystem* Bridge = ResolveBridge();
	if (!Bridge)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] DecodeState: no bridge subsystem on this game instance"));
		return false;
	}

	// Pool per-channel firing rates into the four spatial bands.
	const TArray<FCl1ChannelRate> Rates = Bridge->GetChannelRates(DecodeWindowSeconds);
	float Total = 0.f;
	for (const FCl1ChannelRate& R : Rates)
	{
		const int32 Band = BandOfChannel(R.Channel);
		if (Result.BandScores.IsValidIndex(Band))
		{
			Result.BandScores[Band] += R.RateHz;
			Total += R.RateHz;
		}
	}

	Result.bHasSignal = Total > KINDA_SMALL_NUMBER;

	// argmax over the four band scores.
	int32 Best = 0;
	for (int32 i = 1; i < 4; ++i)
	{
		if (Result.BandScores[i] > Result.BandScores[Best]) { Best = i; }
	}
	Result.State = static_cast<ECl1GameState>(Best);

	// Softmax certainty of the winner (shifted by the max for numerical stability).
	const float MaxScore = Result.BandScores[Best];
	double Sum = 0.0, Top = 0.0;
	for (int32 i = 0; i < 4; ++i)
	{
		const double E = FMath::Exp(static_cast<double>(Result.BandScores[i] - MaxScore));
		Sum += E;
		if (i == Best) { Top = E; }
	}
	Result.Confidence = (Sum > 0.0) ? static_cast<float>(Top / Sum) : 0.f;

	return Result.bHasSignal;
}

// --------------------------------------------------------------------------- //
// Optional auto-decode loop
// --------------------------------------------------------------------------- //
void UCl1GameStates::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!bAutoDecode) { return; }

	TimeSinceDecode += DeltaTime;
	if (TimeSinceDecode < DecodeIntervalSeconds) { return; }
	TimeSinceDecode = 0.f;

	FCl1DecodeResult Result;
	if (DecodeState(Result) && Result.Confidence >= MinConfidence)
	{
		OnStateDecoded.Broadcast(Result);
	}
}
