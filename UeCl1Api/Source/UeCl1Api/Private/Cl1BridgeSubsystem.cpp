// Copyright. Implementation of the Assembloid Agency UE<->CL1 UDP bridge subsystem.

#include "Cl1BridgeSubsystem.h"

#include "Common/UdpSocketBuilder.h"
#include "Common/UdpSocketReceiver.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "Serialization/ArrayReader.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "Math/UnrealMathUtility.h"

// 'AA' control packet message types / flags (must match bridge.py + PROTOCOL.md)
namespace AA
{
	constexpr uint8 MsgStim      = 1;
	constexpr uint8 MsgInterrupt = 2;
	constexpr uint8 MsgRecord    = 4;

	constexpr uint8 FlagBiphasic         = 0x01; // charge-balanced (RECORD: start/stop)
	constexpr uint8 FlagInterruptThenStim = 0x02;

	constexpr uint8 Version = 2;
}

void UCl1BridgeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	SpikeHistory.SetNum(HistoryCapacity);
}

void UCl1BridgeSubsystem::Deinitialize()
{
	StopReceiver();
	if (ControlSocket)
	{
		ControlSocket->Close();
		ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ControlSocket);
		ControlSocket = nullptr;
	}
	Super::Deinitialize();
}

// --------------------------------------------------------------------------- //
// Connection
// --------------------------------------------------------------------------- //
bool UCl1BridgeSubsystem::StartReceiver(int32 ListenPort, const FString& BindAddress)
{
	StopReceiver();

	FIPv4Address BindAddr = FIPv4Address::Any;
	FIPv4Address::Parse(BindAddress, BindAddr);

	ListenSocket = FUdpSocketBuilder(TEXT("Cl1SpikeListen"))
		.AsNonBlocking().AsReusable()
		.BoundToAddress(BindAddr).BoundToPort(ListenPort)
		.WithReceiveBufferSize(2 * 1024 * 1024)
		.Build();

	if (!ListenSocket)
	{
		UE_LOG(LogTemp, Error, TEXT("[CL1] Failed to bind spike listener on %s:%d"),
			*BindAddress, ListenPort);
		return false;
	}

	Receiver = new FUdpSocketReceiver(ListenSocket, FTimespan::FromMilliseconds(100),
		TEXT("Cl1SpikeReceiver"));
	Receiver->OnDataReceived().BindUObject(this, &UCl1BridgeSubsystem::HandleReceivedData);
	Receiver->Start();

	UE_LOG(LogTemp, Log, TEXT("[CL1] Spike receiver on %s:%d"), *BindAddress, ListenPort);
	return true;
}

void UCl1BridgeSubsystem::StopReceiver()
{
	if (Receiver) { delete Receiver; Receiver = nullptr; }
	if (ListenSocket)
	{
		ListenSocket->Close();
		ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListenSocket);
		ListenSocket = nullptr;
	}
}

bool UCl1BridgeSubsystem::ConfigureControlTarget(const FString& Host, int32 Port)
{
	ISocketSubsystem* SocketSub = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (!SocketSub) { return false; }

	if (!ControlSocket)
	{
		ControlSocket = FUdpSocketBuilder(TEXT("Cl1ControlSend")).AsReusable().Build();
		if (!ControlSocket)
		{
			UE_LOG(LogTemp, Error, TEXT("[CL1] Failed to create control socket"));
			return false;
		}
	}

	bool bValid = false;
	ControlAddr = SocketSub->CreateInternetAddr();
	ControlAddr->SetIp(*Host, bValid);
	ControlAddr->SetPort(Port);
	if (!bValid)
	{
		UE_LOG(LogTemp, Error, TEXT("[CL1] Invalid control host: %s"), *Host);
		ControlAddr.Reset();
		return false;
	}
	UE_LOG(LogTemp, Log, TEXT("[CL1] Control target %s:%d"), *Host, Port);
	return true;
}

// --------------------------------------------------------------------------- //
// Inbound spike handling
// --------------------------------------------------------------------------- //
void UCl1BridgeSubsystem::HandleReceivedData(const FArrayReaderPtr& Data,
                                             const FIPv4Endpoint& /*Endpoint*/)
{
	const int32 Num = Data->Num();
	if (Num < 9) { return; } // 8-byte timestamp + >=1 channel

	const uint8* Bytes = Data->GetData();

	// Explicit little-endian uint64 (matches struct '<Q'), host-order independent.
	uint64 Timestamp = 0;
	for (int32 i = 0; i < 8; ++i)
	{
		Timestamp |= (static_cast<uint64>(Bytes[i]) << (8 * i));
	}

	TArray<FCl1Spike> Spikes;
	Spikes.Reserve(Num - 8);
	for (int32 i = 8; i < Num; ++i)
	{
		FCl1Spike S;
		S.Timestamp = static_cast<int64>(Timestamp);
		S.Channel = static_cast<int32>(Bytes[i]);
		S.TimeSeconds = Cl1Time::FramesToSeconds(S.Timestamp);
		Spikes.Add(S);
	}

	TWeakObjectPtr<UCl1BridgeSubsystem> WeakThis(this);
	AsyncTask(ENamedThreads::GameThread, [WeakThis, Spikes = MoveTemp(Spikes)]()
	{
		UCl1BridgeSubsystem* Self = WeakThis.Get();
		if (!Self) { return; }
		for (const FCl1Spike& S : Spikes) { Self->TrackSpike(S); }
		Self->OnSpikeBatch.Broadcast(Spikes);
		for (const FCl1Spike& S : Spikes) { Self->OnSpike.Broadcast(S); }
	});
}

void UCl1BridgeSubsystem::TrackSpike(const FCl1Spike& Spike)
{
	LatestFrame = FMath::Max(LatestFrame, Spike.Timestamp);
	SpikeHistory[HistoryHead] = Spike;
	HistoryHead = (HistoryHead + 1) % HistoryCapacity;
	if (bUeLogging) { UeLog.Add(Spike); }
}

// --------------------------------------------------------------------------- //
// Outbound control packet ('AA')
// --------------------------------------------------------------------------- //
bool UCl1BridgeSubsystem::SendControlPacket(uint8 MsgType, uint8 Flags,
	const TArray<int32>& Channels, uint16 PulseWidthUs, float AmplitudeUa,
	uint16 NumPulses, uint16 FreqHz)
{
	if (!ControlSocket || !ControlAddr.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] Control packet before ConfigureControlTarget"));
		return false;
	}
	if (Channels.Num() > 255)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] Too many channels (%d) for one packet"),
			Channels.Num());
		return false;
	}

	// Header '<2sBBBBHfHH' (16 bytes) + channel bytes. Little-endian throughout.
	TArray<uint8> Packet;
	Packet.Reserve(16 + Channels.Num());

	auto PushU8  = [&](uint8 V){ Packet.Add(V); };
	auto PushU16 = [&](uint16 V){ Packet.Add(uint8(V & 0xFF)); Packet.Add(uint8((V >> 8) & 0xFF)); };
	auto PushF32 = [&](float V){ uint32 B; FMemory::Memcpy(&B, &V, 4);
		Packet.Add(uint8(B & 0xFF)); Packet.Add(uint8((B >> 8) & 0xFF));
		Packet.Add(uint8((B >> 16) & 0xFF)); Packet.Add(uint8((B >> 24) & 0xFF)); };

	PushU8('A'); PushU8('A');           // magic
	PushU8(AA::Version);                // version
	PushU8(MsgType);                    // msg_type
	PushU8(Flags);                      // flags
	PushU8(uint8(Channels.Num()));      // num_channels
	PushU16(PulseWidthUs);              // pulse_width_us
	PushF32(AmplitudeUa);               // amplitude_uA
	PushU16(NumPulses);                 // num_pulses
	PushU16(FreqHz);                    // freq_hz
	for (int32 C : Channels) { PushU8(uint8(FMath::Clamp(C, 0, 255))); }

	int32 Sent = 0;
	const bool bOk = ControlSocket->SendTo(Packet.GetData(), Packet.Num(), Sent, *ControlAddr);
	if (!bOk || Sent != Packet.Num())
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] Control send failed (%d/%d) - verify bridge.py listening and ConfigureControlTarget() called"), Sent, Packet.Num());
		return false;
	}
	return true;
}

// --------------------------------------------------------------------------- //
// Stimulation API (Assembloid Sec 3.3)
// --------------------------------------------------------------------------- //
bool UCl1BridgeSubsystem::SendStimulation(const TArray<int32>& Channels, float FreqHz,
	int32 PulseWidthUs, float AmplitudeUa, float DurationMs, bool bInterruptFirst,
	FString& OutChannels, FString& OutFreqHz, FString& OutPulseWidth,
	FString& OutAmplitude, FString& OutDuration)
{
	if (Channels.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CL1] SendStimulation: no channels specified"));
		return false;
	}

	// Build output strings for Blueprint debugging/logging
	FString ChannelStr = TEXT("[");
	for (int32 i = 0; i < Channels.Num(); ++i)
	{
		if (i > 0) ChannelStr += TEXT(", ");
		ChannelStr += FString::Printf(TEXT("%d"), Channels[i]);
	}
	ChannelStr += TEXT("]");
	OutChannels = ChannelStr;
	OutFreqHz = FString::Printf(TEXT("%.1f Hz"), FreqHz);
	OutPulseWidth = FString::Printf(TEXT("%d us"), PulseWidthUs);
	OutAmplitude = FString::Printf(TEXT("%.2f uA"), AmplitudeUa);
	OutDuration = FString::Printf(TEXT("%.1f ms"), DurationMs);

	// Duration -> pulse count (PROTOCOL.md): NumPulses = max(1, round(s * Hz)).
	int32 NumPulses = 1;
	if (FreqHz > 0.f && DurationMs > 0.f)
	{
		NumPulses = FMath::Max(1, FMath::RoundToInt((DurationMs / 1000.f) * FreqHz));
	}

	uint8 Flags = AA::FlagBiphasic;
	if (bInterruptFirst) { Flags |= AA::FlagInterruptThenStim; }

	return SendControlPacket(AA::MsgStim, Flags, Channels,
		uint16(FMath::Clamp(PulseWidthUs, 0, 65535)),
		AmplitudeUa,
		uint16(FMath::Clamp(NumPulses, 0, 65535)),
		uint16(FMath::Clamp(FMath::RoundToInt(FreqHz), 0, 65535)));
}

bool UCl1BridgeSubsystem::SendRewardSignal(bool bPositive, const TArray<int32>& RewardChannels,
	float FreqHz, int32 PulseWidthUs, float AmplitudeUa, float DurationMs)
{
	// DishBrain-style convention (override to match your protocol):
	//   positive => predictable burst; negative => silence (interrupt).
	if (bPositive)
	{
		return SendStimulation(RewardChannels, FreqHz, PulseWidthUs, AmplitudeUa,
			DurationMs, /*bInterruptFirst*/ true);
	}
	return InterruptStim(RewardChannels);
}

bool UCl1BridgeSubsystem::InterruptStim(const TArray<int32>& Channels)
{
	if (Channels.Num() == 0) { return false; }
	return SendControlPacket(AA::MsgInterrupt, 0, Channels, 0, 0.f, 0, 0);
}

// --------------------------------------------------------------------------- //
// Recording (CL paper Sec 6.4)
// --------------------------------------------------------------------------- //
bool UCl1BridgeSubsystem::RecordSessionData(bool bStart, bool bAlsoLogInUE)
{
	bUeLogging = bAlsoLogInUE && bStart;
	if (bStart) { UeLog.Reset(); }

	// RECORD message: reuse flags bit0 as start/stop. Empty channel list.
	const uint8 Flags = bStart ? AA::FlagBiphasic : 0;
	return SendControlPacket(AA::MsgRecord, Flags, TArray<int32>(), 0, 0.f, 0, 0);
}

bool UCl1BridgeSubsystem::ExportToCSV(const FString& FilePath)
{
	FString Csv = TEXT("frame,time_seconds,channel\n");
	for (const FCl1Spike& S : UeLog)
	{
		Csv += FString::Printf(TEXT("%lld,%.6f,%d\n"), S.Timestamp, S.TimeSeconds, S.Channel);
	}
	const bool bOk = FFileHelper::SaveStringToFile(Csv, *FilePath);
	UE_LOG(LogTemp, Log, TEXT("[CL1] ExportToCSV %s (%d rows) -> %s"),
		*FilePath, UeLog.Num(), bOk ? TEXT("ok") : TEXT("FAILED"));
	return bOk;
}

// --------------------------------------------------------------------------- //
// Readback / visualization
// --------------------------------------------------------------------------- //
int64 UCl1BridgeSubsystem::GetSpikeResponse(int32 Channel) const
{
	int64 Best = -1;
	for (const FCl1Spike& S : SpikeHistory)
	{
		if (S.Channel == Channel && S.Timestamp > Best) { Best = S.Timestamp; }
	}
	return Best;
}

float UCl1BridgeSubsystem::GetSpikeRateHz(int32 Channel, float WindowSeconds) const
{
	if (WindowSeconds <= 0.f) { return 0.f; }
	const int64 WindowFrames = int64(WindowSeconds * Cl1Time::FramesPerSecond);
	const int64 Cutoff = LatestFrame - WindowFrames;
	int32 Count = 0;
	for (const FCl1Spike& S : SpikeHistory)
	{
		if (S.Channel == Channel && S.Timestamp > 0 && S.Timestamp >= Cutoff) { ++Count; }
	}
	return float(Count) / WindowSeconds;
}

TArray<FCl1ChannelRate> UCl1BridgeSubsystem::GetChannelRates(float WindowSeconds) const
{
	TMap<int32, int32> Counts;
	const int64 WindowFrames = int64(FMath::Max(WindowSeconds, KINDA_SMALL_NUMBER) * Cl1Time::FramesPerSecond);
	const int64 Cutoff = LatestFrame - WindowFrames;
	for (const FCl1Spike& S : SpikeHistory)
	{
		if (S.Timestamp > 0 && S.Timestamp >= Cutoff) { Counts.FindOrAdd(S.Channel)++; }
	}
	TArray<FCl1ChannelRate> Out;
	Out.Reserve(Counts.Num());
	for (const TPair<int32, int32>& Pair : Counts)
	{
		FCl1ChannelRate R;
		R.Channel = Pair.Key;
		R.RateHz = float(Pair.Value) / FMath::Max(WindowSeconds, KINDA_SMALL_NUMBER);
		Out.Add(R);
	}
	return Out;
}
