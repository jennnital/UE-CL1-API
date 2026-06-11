#include "AssembloidAgencyComponent.h"
#include "UDPComponent.h"   // brings in FUDPNative / FUDPSettings from the UDP-Unreal plugin

// ----------------------------------------------------------------------------
//  Wire-format constants (see PROTOCOL.md)
// ----------------------------------------------------------------------------
namespace
{
	constexpr uint8  STIM_MAGIC_0   = 'A';
	constexpr uint8  STIM_MAGIC_1   = 'A';
	constexpr uint8  STIM_VERSION   = 1;
	constexpr uint8  STIM_MSG_STIM  = 1;
	constexpr uint8  STIM_FLAG_BIPHASIC = 0x01;
	constexpr int32  STIM_HEADER_SIZE   = 16;
	constexpr int32  SPIKE_MIN_SIZE     = 9;   // uint64 timestamp + >=1 channel

	// Append a little-endian uint16.
	FORCEINLINE void AppendU16LE(TArray<uint8>& Out, uint16 Value)
	{
		Out.Add(static_cast<uint8>(Value & 0xFF));
		Out.Add(static_cast<uint8>((Value >> 8) & 0xFF));
	}

	// Append a little-endian float32.
	FORCEINLINE void AppendF32LE(TArray<uint8>& Out, float Value)
	{
		uint32 Bits;
		FMemory::Memcpy(&Bits, &Value, sizeof(Bits));
		Out.Add(static_cast<uint8>(Bits & 0xFF));
		Out.Add(static_cast<uint8>((Bits >> 8) & 0xFF));
		Out.Add(static_cast<uint8>((Bits >> 16) & 0xFF));
		Out.Add(static_cast<uint8>((Bits >> 24) & 0xFF));
	}
}

UAssembloidAgencyComponent::UAssembloidAgencyComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAssembloidAgencyComponent::BeginPlay()
{
	Super::BeginPlay();
	if (bAutoConnectOnBeginPlay)
	{
		Connect();
	}
}

void UAssembloidAgencyComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
}

bool UAssembloidAgencyComponent::Connect()
{
	if (bConnected)
	{
		return true;
	}

	UDP = MakeShared<FUDPNative>();

	// Configure the embedded native UDP wrapper.
	UDP->Settings.SendIP    = CLDeviceIP;
	UDP->Settings.SendPort  = StimSendPort;
	UDP->Settings.ReceiveIP = SpikeListenIP;
	UDP->Settings.ReceivePort = SpikeListenPort;
	UDP->Settings.bReceiveDataOnGameThread = true;  // safe to touch game state in the handler
	UDP->Settings.bShouldAutoOpenSend = false;
	UDP->Settings.bShouldAutoOpenReceive = false;

	TWeakObjectPtr<UAssembloidAgencyComponent> WeakThis(this);
	UDP->OnReceivedBytes =
		[WeakThis](const TArray<uint8>& Data, const FString& Endpoint, const int32& Port)
		{
			if (UAssembloidAgencyComponent* Self = WeakThis.Get())
			{
				Self->HandleReceivedBytes(Data, Endpoint, Port);
			}
		};

	const bool bRecv = UDP->OpenReceiveSocket(SpikeListenIP, SpikeListenPort);
	const int32 BoundSendPort = UDP->OpenSendSocket(CLDeviceIP, StimSendPort);
	const bool bSend = BoundSendPort != 0;

	bConnected = bRecv && bSend;
	if (bConnected)
	{
		UE_LOG(LogTemp, Log,
			TEXT("[AssembloidAgency] Listening for spikes on %s:%d, sending stim to %s:%d"),
			*SpikeListenIP, SpikeListenPort, *CLDeviceIP, StimSendPort);
		OnBridgeConnected.Broadcast(SpikeListenPort);
	}
	else
	{
		UE_LOG(LogTemp, Error,
			TEXT("[AssembloidAgency] Connect failed (recv=%d send=%d)"), bRecv, bSend);
		Disconnect();
	}
	return bConnected;
}

void UAssembloidAgencyComponent::Disconnect()
{
	if (UDP.IsValid())
	{
		UDP->ClearReceiveCallbacks();
		UDP->CloseReceiveSocket();
		UDP->CloseSendSocket();
		UDP.Reset();
	}
	bConnected = false;
}

// ----------------------------------------------------------------------------
//  Spike reception (game thread)
// ----------------------------------------------------------------------------
void UAssembloidAgencyComponent::HandleReceivedBytes(
	const TArray<uint8>& Data, const FString& Endpoint, const int32& /*Port*/)
{
	if (Data.Num() < SPIKE_MIN_SIZE)
	{
		return; // too small to be a valid spike packet
	}

	// 8-byte little-endian timestamp.
	uint64 Timestamp = 0;
	for (int32 i = 0; i < 8; ++i)
	{
		Timestamp |= static_cast<uint64>(Data[i]) << (8 * i);
	}

	FCLSpikeEvent Event;
	Event.Timestamp = static_cast<int64>(Timestamp);
	Event.SourceIP = Endpoint;
	Event.Channels.Reserve(Data.Num() - 8);

	const double Now = FPlatformTime::Seconds();
	for (int32 i = 8; i < Data.Num(); ++i)
	{
		const int32 Channel = static_cast<int32>(Data[i]);
		Event.Channels.Add(Channel);

		RecentSpikeTimes.FindOrAdd(Channel).Add(Now);
		TotalSpikeCounts.FindOrAdd(Channel)++;
	}

	PruneOldSpikeTimes(Now);

	LastSpikeEvent = Event;
	OnSpikesReceived.Broadcast(Event);
}

void UAssembloidAgencyComponent::PruneOldSpikeTimes(double Now)
{
	const double Cutoff = Now - FMath::Max(0.01f, SpikeRateWindowSeconds);
	for (auto& Pair : RecentSpikeTimes)
	{
		TArray<double>& Times = Pair.Value;
		int32 KeepFrom = 0;
		while (KeepFrom < Times.Num() && Times[KeepFrom] < Cutoff)
		{
			++KeepFrom;
		}
		if (KeepFrom > 0)
		{
			Times.RemoveAt(0, KeepFrom);  // version-agnostic across UE5 releases
		}
	}
}

// ----------------------------------------------------------------------------
//  Stimulation (build + emit STIM packets)
// ----------------------------------------------------------------------------
bool UAssembloidAgencyComponent::SendStimulusPulses(
	const TArray<int32>& Channels, int32 NumPulses, int32 FrequencyHz,
	int32 PulseWidthUs, float AmplitudeUA, bool bBiphasic)
{
	if (!bConnected || !UDP.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("[AssembloidAgency] SendStimulus while disconnected"));
		return false;
	}
	if (Channels.Num() == 0 || Channels.Num() > 255)
	{
		UE_LOG(LogTemp, Warning, TEXT("[AssembloidAgency] Invalid channel count: %d"), Channels.Num());
		return false;
	}

	NumPulses    = FMath::Clamp(NumPulses, 1, 65535);
	FrequencyHz  = FMath::Clamp(FrequencyHz, 0, 65535);
	PulseWidthUs = FMath::Clamp(PulseWidthUs, 0, 65535);

	TArray<uint8> Packet;
	Packet.Reserve(STIM_HEADER_SIZE + Channels.Num());

	Packet.Add(STIM_MAGIC_0);
	Packet.Add(STIM_MAGIC_1);
	Packet.Add(STIM_VERSION);
	Packet.Add(STIM_MSG_STIM);
	Packet.Add(bBiphasic ? STIM_FLAG_BIPHASIC : 0);
	Packet.Add(static_cast<uint8>(Channels.Num()));
	AppendU16LE(Packet, static_cast<uint16>(PulseWidthUs));
	AppendF32LE(Packet, FMath::Abs(AmplitudeUA));   // magnitude; bridge sets phase polarity
	AppendU16LE(Packet, static_cast<uint16>(NumPulses));
	AppendU16LE(Packet, static_cast<uint16>(FrequencyHz));

	for (int32 Ch : Channels)
	{
		Packet.Add(static_cast<uint8>(FMath::Clamp(Ch, 0, 255)));
	}

	return UDP->EmitBytes(Packet);
}

bool UAssembloidAgencyComponent::SendStimulus(
	const TArray<int32>& Channels, int32 FrequencyHz, int32 PulseWidthUs,
	float AmplitudeUA, int32 DurationMs)
{
	int32 NumPulses = 1;
	if (FrequencyHz > 0 && DurationMs > 0)
	{
		NumPulses = FMath::Max(1, FMath::RoundToInt((DurationMs / 1000.0f) * FrequencyHz));
	}
	return SendStimulusPulses(Channels, NumPulses, FrequencyHz, PulseWidthUs, AmplitudeUA, true);
}

bool UAssembloidAgencyComponent::SendSinglePulse(
	const TArray<int32>& Channels, int32 PulseWidthUs, float AmplitudeUA)
{
	return SendStimulusPulses(Channels, 1, 0, PulseWidthUs, AmplitudeUA, true);
}

// ----------------------------------------------------------------------------
//  Spike analysis
// ----------------------------------------------------------------------------
float UAssembloidAgencyComponent::GetSpikeRateHz(int32 Channel) const
{
	const TArray<double>* Times = RecentSpikeTimes.Find(Channel);
	if (!Times || Times->Num() == 0)
	{
		return 0.0f;
	}
	const double Now = FPlatformTime::Seconds();
	const double Window = FMath::Max(0.01f, SpikeRateWindowSeconds);
	int32 Count = 0;
	for (double T : *Times)
	{
		if (T >= Now - Window)
		{
			++Count;
		}
	}
	return static_cast<float>(Count / Window);
}

int32 UAssembloidAgencyComponent::GetTotalSpikeCount(int32 Channel) const
{
	const int32* Count = TotalSpikeCounts.Find(Channel);
	return Count ? *Count : 0;
}

void UAssembloidAgencyComponent::ResetSpikeCounts()
{
	RecentSpikeTimes.Reset();
	TotalSpikeCounts.Reset();
}
